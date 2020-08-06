#!/usr/bin/env python3
import os
import cv2
import random
from random import shuffle
import os.path
import nori2
import numpy as np
import pickle as pkl
from PIL import Image
from queue import Queue
from common import Config
from argparse import ArgumentParser
import sys
from tqdm import tqdm
from lib.utils.my_utils import my_utils
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MeanShift
import concurrent.futures
# from neupeak.utils.webcv2 import imshow, waitKey

from lib.utils.icp.icp import my_icp, best_fit_transform
import numpy as np


cls_type = open('./cls_type.txt').readline().strip()
config = Config(cls_type)
DEBUG=False #True
SHOW=False

tst_nid_lst = my_utils.read_lines(config.val_nid_ptn.format('allobj'))

xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

n_sample_points = 2000
mininum_cnt = 1500
nf = nori2.Fetcher()

cls_lst = my_utils.read_lines(config.ycb_cls_lst_p)
obj_dict = {}
for cls_id, cls in enumerate(cls_lst, start=1):
    obj_dict[cls] = cls_id

pvn3d_poses = np.load(open('./pvn3d_poses.npy', 'rb'))

n_cls = 22
gb_cls_add_dis = [list() for i in range(n_cls)]
gb_cls_adds_dis = [list() for i in range(n_cls)]
gb_cls_add_dis_icp = [list() for i in range(n_cls)]
gb_cls_adds_dis_icp = [list() for i in range(n_cls)]

radius = 0.06


def get_cld_bigest_clus(p3ds):
    n_clus_jobs = 8
    ms = MeanShift(
        bandwidth=radius, bin_seeding=True, n_jobs=n_clus_jobs
    )
    ms.fit(p3ds)
    clus_labels = ms.labels_
    bg_clus = p3ds[np.where(clus_labels == 0)[0], :]
    return bg_clus


def cal_adds_dis(cls_ptsxyz, pred_pose, gt_pose):
    pred_pts = np.dot(cls_ptsxyz.copy(), pred_pose[:, :3].T) + pred_pose[:, 3]
    gt_pts = np.dot(cls_ptsxyz.copy(), gt_pose[:, :3].T) + gt_pose[:, 3]
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(gt_pts)
    distances, _ = neigh.kneighbors(pred_pts, return_distance=True)
    return np.mean(distances)


def cal_add_dis(cls_ptsxyz, pred_pose, gt_pose):
    pred_pts = np.dot(cls_ptsxyz.copy(), pred_pose[:, :3].T) + pred_pose[:, 3]
    gt_pts = np.dot(cls_ptsxyz.copy(), gt_pose[:, :3].T) + gt_pose[:, 3]
    mean_dist = np.mean(np.linalg.norm(pred_pts - gt_pts, axis=-1))
    return mean_dist


sv_icp_msk_dir = 'train_log/eval_result/icp'
if not os.path.exists(sv_icp_msk_dir):
    os.mkdir(sv_icp_msk_dir)


def sv_mesh(p3ds, sv_pth):
    with open(sv_pth, 'w') as f:
        for p3d in p3ds:
            print('v', p3d[0], p3d[1], p3d[2], file=f)


def eval_item(nid_pvn3d_poses):
    nid, pvn3d_poses = nid_pvn3d_poses[0], nid_pvn3d_poses[1]
    # print(nid, pvn3d_poses)
    data = pkl.loads(nf.get(nid))
    obj_info_lst = data['obj_info_lst']
    dpt = data['depth'].astype(np.float32).copy()
    labels = data['label']
    cam_scale = data['meta']['factor_depth'].astype(np.float32)[0][0]
    msk_dp = dpt > 1e-6

    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    cls_add_dis_icp = [list() for i in range(n_cls)]
    cls_adds_dis_icp = [list() for i in range(n_cls)]

    K = obj_info_lst[0]['K']

    for i, obj_info in enumerate(obj_info_lst):
        cls_id = obj_dict[obj_info['cls_typ']]

        has_pose = False
        for cid, pose in pvn3d_poses:
            if cid == cls_id:
                has_pose = True
                break
        if not has_pose:
            pose = np.zeros((3, 4), dtype=np.float32)
        init_pose = np.identity(4, dtype=np.float32)
        init_pose[:3, :] = pose

        cls_msk = msk_dp & (labels == cls_id)
        if DEBUG and SHOW:
            cv2.imshow('cls_msk', cls_msk.astype('uint8') * 255)
        # if cls_msk.sum() < n_sample_points:
        #     print("num pts:", cls_msk.sum())

        choose = cls_msk.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) > n_sample_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        # else:
        #     choose = np.pad(choose, (0, n_sample_points-len(choose)), 'wrap')

        cls_ptsxyz = my_utils.get_pointxyz(cls_lst[cls_id-1])
        adds_dis = cal_adds_dis(cls_ptsxyz.copy(), pose, obj_info['RT'])
        add_dis = cal_add_dis(cls_ptsxyz.copy(), pose, obj_info['RT'])
        cls_adds_dis[cls_id].append(adds_dis)
        cls_add_dis[cls_id].append(add_dis)
        cls_adds_dis[0].append(adds_dis)
        cls_add_dis[0].append(add_dis)
        if len(choose) < mininum_cnt:
            cls_adds_dis_icp[cls_id].append(adds_dis)
            cls_add_dis_icp[cls_id].append(add_dis)
            cls_adds_dis_icp[0].append(adds_dis)
            cls_add_dis_icp[0].append(add_dis)
            continue

        if DEBUG:
            pvn3d_p3d = np.dot(cls_ptsxyz.copy(), pose[:3, :3].T) + pose[:3, 3]
            pvn3d_p2d = my_utils.project_p3d(pvn3d_p3d, 1)
            show_pvn3d_pose = np.zeros((480, 640, 3), dtype='uint8')
            show_pvn3d_pose = my_utils.draw_p2ds(show_pvn3d_pose, pvn3d_p2d)
            if SHOW:
                cv2.imshow('pvn3d', show_pvn3d_pose)

        dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32).copy()
        xmap_mskd = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_mskd = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

        pt2 = dpt_mskd / cam_scale
        cam_cx, cam_cy = K[0][2], K[1][2]
        cam_fx, cam_fy = K[0][0], K[1][1]
        pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
        cld = np.concatenate( (pt0, pt1, pt2), axis=1 )
        cld = get_cld_bigest_clus(cld)

        if DEBUG:
            cld_p2d = my_utils.project_p3d(cld, 1)
            show_cld = np.zeros((480, 640, 3), dtype='uint8')
            show_cld = my_utils.draw_p2ds(show_cld, cld_p2d)
            if SHOW:
                cv2.imshow('cld', show_cld)


        icp_pose, dis, _ = my_icp(
            cls_ptsxyz.copy(), cld, init_pose=init_pose,
            max_iterations=500,
            tolerance=1e-9
        )
        # print('dis final icp:', np.mean(dis))

        pose = icp_pose[:3, :]
        adds_dis_icp = cal_adds_dis(cls_ptsxyz.copy(), pose, obj_info['RT'])
        add_dis_icp = cal_add_dis(cls_ptsxyz.copy(), pose, obj_info['RT'])
        cls_adds_dis_icp[cls_id].append(adds_dis_icp)
        cls_add_dis_icp[cls_id].append(add_dis_icp)
        cls_adds_dis_icp[0].append(adds_dis_icp)
        cls_add_dis_icp[0].append(add_dis_icp)
        # print(adds_dis, adds_dis_icp, add_dis, add_dis_icp)
        if DEBUG:
            icp_p3d = np.dot(cls_ptsxyz.copy(), pose[:3, :3].T) + pose[:3, 3]
            icp_p2d = my_utils.project_p3d(icp_p3d, 1)
            show_icp_pose = np.zeros((480, 640, 3), dtype='uint8')
            show_icp_pose = my_utils.draw_p2ds(show_icp_pose, icp_p2d)
            if adds_dis_icp - adds_dis > 0.05:
                item_name = '{}_{}_{}_'.format(nid, adds_dis_icp, adds_dis)
                sv_mesh(
                    icp_p3d,
                    os.path.join(sv_icp_msk_dir, item_name+'icp.obj'),
                )
                sv_mesh(
                    pvn3d_p3d,
                    os.path.join(sv_icp_msk_dir, item_name+'pvn3d.obj'),
                )
                sv_mesh(
                    cld,
                    os.path.join(sv_icp_msk_dir, item_name+'cld.obj'),
                )
            if SHOW:
                cv2.imshow('icp', show_icp_pose)
                cmd = cv2.waitKey(0)
                if cmd == ord('q'):
                    exit()

    return (cls_add_dis, cls_adds_dis, cls_add_dis_icp, cls_adds_dis_icp)


def eval_item_pvn3d_msk(ipic_nid):
    ipic, nid, pvn3d_poses = ipic_nid[0], ipic_nid[1], ipic_nid[2]
    info_fd = 'train_log/eval_result/004_sugar_box/torch_res/our_msk_info/'
    data = pkl.loads(nf.get(nid))
    dpt = data['depth'].astype(np.float32).copy()
    obj_info_lst = data['obj_info_lst']
    labels = cv2.imread(info_fd+'{}_fillpredmsk.png'.format(ipic))[:, :, 0]

    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    cls_add_dis_icp = [list() for i in range(n_cls)]
    cls_adds_dis_icp = [list() for i in range(n_cls)]

    K = obj_info_lst[0]['K']
    cam_scale = data['meta']['factor_depth'].astype(np.float32)[0][0]
    all_cld, choose = my_utils.dpt_2_cld(dpt, cam_scale, K)
    labels = labels.reshape(-1)[choose]
    for i, obj_info in enumerate(obj_info_lst):
        cls_id = obj_dict[obj_info['cls_typ']]

        for cid, pose in pvn3d_poses:
            if cid == cls_id:
                has_pose = True
                break
        if not has_pose:
            pose = np.zeros((3, 4), dtype=np.float32)
        init_pose = np.identity(4, dtype=np.float32)
        init_pose[:3, :] = pose

        cls_ptsxyz = my_utils.get_pointxyz(cls_lst[cls_id-1])
        adds_dis = cal_adds_dis(cls_ptsxyz.copy(), pose, obj_info['RT'])
        add_dis = cal_add_dis(cls_ptsxyz.copy(), pose, obj_info['RT'])
        cls_adds_dis[cls_id].append(adds_dis)
        cls_add_dis[cls_id].append(add_dis)
        cls_adds_dis[0].append(adds_dis)
        cls_add_dis[0].append(add_dis)

        choose = np.where(labels == cls_id)[0]
        if len(choose) > n_sample_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        cld = all_cld[choose, :]
        if cld.shape[0] < 1500:
            cls_adds_dis_icp[cls_id].append(adds_dis)
            cls_add_dis_icp[cls_id].append(add_dis)
            cls_adds_dis_icp[0].append(adds_dis)
            cls_add_dis_icp[0].append(add_dis)
            continue

        cld = get_cld_bigest_clus(cld)

        icp_pose, dis, _ = my_icp(
            cls_ptsxyz.copy(), cld, init_pose=init_pose,
            max_iterations=500,
            tolerance=1e-9
        )

        pose = icp_pose[:3, :]
        adds_dis_icp = cal_adds_dis(cls_ptsxyz.copy(), pose, obj_info['RT'])
        add_dis_icp = cal_add_dis(cls_ptsxyz.copy(), pose, obj_info['RT'])
        cls_adds_dis_icp[cls_id].append(adds_dis_icp)
        cls_add_dis_icp[cls_id].append(add_dis_icp)
        cls_adds_dis_icp[0].append(adds_dis_icp)
        cls_add_dis_icp[0].append(add_dis_icp)

    return (cls_add_dis, cls_adds_dis, cls_add_dis_icp, cls_adds_dis_icp)


max_workers= 10
label_type='predmsk'
# label_type='gtmsk'
def cal_pose_icp():
    pvn3d_poses = np.load(open('./pvn3d_poses.npy', 'rb'))
    idx = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        if label_type == 'predmsk':
            exc_map = executor.map(
                eval_item_pvn3d_msk, tqdm(
                    zip(list(range(len(tst_nid_lst))), tst_nid_lst, pvn3d_poses)
                )
            )
        else:
            exc_map = executor.map(
                eval_item, tqdm(zip(tst_nid_lst, pvn3d_poses))
            )
        for data in exc_map:
            """
            data: (cls_add_dis, cls_adds_dis, cls_add_dis_icp, cls_adds_dis_icp)
            """
            for cls_id in range(n_cls):
                gb_cls_add_dis[cls_id] += data[0][cls_id]
                gb_cls_adds_dis[cls_id] += data[1][cls_id]
                gb_cls_add_dis_icp[cls_id] += data[2][cls_id]
                gb_cls_adds_dis_icp[cls_id] += data[3][cls_id]
            idx += 1
            print(idx)

    cls_add_auc_icp = []
    cls_add_auc = []
    cls_adds_auc_icp = []
    cls_add_s_auc_icp = []
    cls_adds_auc = []
    gb_cls_add_s_dis_icp = [list() for i in range(22)]
    for cls_id in range(1, 22):
        if cls_id in config.ycb_sym_cls_ids:
            gb_cls_add_s_dis_icp[cls_id] = gb_cls_adds_dis_icp[cls_id]
        else:
            gb_cls_add_s_dis_icp[cls_id] = gb_cls_add_dis_icp[cls_id]
        gb_cls_add_s_dis_icp[0] += gb_cls_add_s_dis_icp[cls_id]
    for cls_id in range(0, 22):
        cls_add_auc_icp.append(my_utils.cal_auc(gb_cls_add_dis_icp[cls_id]))
        cls_add_auc.append(my_utils.cal_auc(gb_cls_add_dis[cls_id]))
        cls_adds_auc_icp.append(my_utils.cal_auc(gb_cls_adds_dis_icp[cls_id]))
        cls_adds_auc.append(my_utils.cal_auc(gb_cls_adds_dis[cls_id]))
        cls_add_s_auc_icp.append(my_utils.cal_auc(gb_cls_add_s_dis_icp[cls_id]))
        if cls_id == 0:
            print("all obj:")
        else:
            print(cls_lst[cls_id-1], ":")
        print(
            "########## add_icp:\t", cls_add_auc_icp[-1], "\n",
            "########## add:\t", cls_add_auc[-1], "\n",
            "########## adds_icp:\t", cls_adds_auc_icp[-1], "\n",
            "########## adds:\t", cls_adds_auc[-1], "\n"
        )

    print("icp:")
    print_screen("icp_add_auc: ", cls_add_auc_icp)
    print_screen("icp_adds_auc: ", cls_adds_auc_icp)
    print_screen("icp_add_s_auc: ", cls_add_s_auc_icp)

    sv_info = dict(
        adds_auc_icp=cls_adds_auc_icp,
        adds_auc=cls_adds_auc,
        add_auc_icp=cls_add_auc_icp,
        add_auc=cls_add_auc,
        cls_add_dis_icp=gb_cls_add_dis_icp,
        cls_adds_dis_icp=gb_cls_adds_dis_icp,
        cls_add_dis=gb_cls_add_dis,
        cls_adds_dis=gb_cls_adds_dis,
    )
    pkl.dump(
        sv_info,
        open(
            './train_log/eval_result/icp_sv_info_{}_{}_{}_{}_{}_{}.pkl'.format(
                n_sample_points, radius,
                cls_adds_auc_icp[0], cls_add_auc_icp[0],
                label_type, mininum_cnt
            ),
            'wb'
        )
    )


def print_screen(title, aucs):
    print(title)
    for i in range(22):
        print(aucs[i])


def fill_label_item(ipic_nid):
    ipic, nid= ipic_nid[0], ipic_nid[1]
    info_fd = 'train_log/eval_result/004_sugar_box/torch_res/our_msk_info/'
    info_ptn = info_fd + '{}.pkl'
    data = pkl.loads(nf.get(nid))
    obj_info_lst = data['obj_info_lst']
    dpt = data['depth'].astype(np.float32).copy()
    cam_scale = data['meta']['factor_depth'].astype(np.float32)[0][0]
    K = obj_info_lst[0]['K']
    all_cld, all_choose = my_utils.dpt_2_cld(dpt, cam_scale, K)

    data_pred = pkl.load(open(info_ptn.format(ipic), 'rb'))
    if 'p3ds' in data_pred.keys():
        key_pcld = 'p3ds'
    else:
        key_pcld = 'pcld'
    sample_cld = data_pred[key_pcld]
    if 'pred_label' in data_pred.keys():
        key_lb = 'pred_label'
    else:
        key_lb = 'labels'
    pred_labels = data_pred[key_lb]
    if type(sample_cld) != np.ndarray:
        sample_cld = sample_cld.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(sample_cld)
    distances, indices = neigh.kneighbors(all_cld, return_distance=True)
    all_labels = pred_labels[indices]
    all_msk = np.zeros((480, 640), dtype="uint8")
    all_msk = all_msk.reshape(-1)
    all_msk[all_choose] = all_labels[:, 0]
    all_msk = all_msk.reshape((480, 640))
    cv2.imwrite(info_fd+'{}_fillpredmsk.png'.format(ipic), all_msk)
    # cv2.imshow("pred_msk", all_msk * (255 // 22))
    # cmd = cv2.waitKey(0)
    # if cmd == ord('q'):
    #     exit()


def fill_label():
    idx = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
        # max_workers=1
    ) as executor:
        exc_map = executor.map(
            fill_label_item, tqdm(enumerate(tst_nid_lst))
        )
        for data in exc_map:
            idx += 1
            print(idx)


def main():
    cal_pose_icp()
    # fill_label()

if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
