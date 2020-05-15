#!/usr/bin/env python3
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import concurrent.futures
import numpy as np
import pickle as pkl
from common import Config
from lib.utils.basic_utils import Basic_Utils
from lib.utils.meanshift_pytorch import MeanShiftTorch


config = Config(dataset_name='ycb')
bs_utils = Basic_Utils(config)
config_lm = Config(dataset_name="linemod")
bs_utils_lm = Basic_Utils(config_lm)
cls_lst = config.ycb_cls_lst


class VotingType:
    BB8=0
    BB8C=1
    BB8S=2
    VanPts=3
    Farthest=5
    Farthest4=6
    Farthest12=7
    Farthest16=8
    Farthest20=9


def cal_frame_poses(
    pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter
):
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius=0.08
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
    else:
        cls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    pred_cls_ids = np.unique(mask[mask>0].contiguous().cpu().numpy())
    if use_ctr_clus_flter:
       ctrs = []
       for icls, cls_id in enumerate(pred_cls_ids):
           cls_msk = (mask == cls_id)
           ms = MeanShiftTorch(bandwidth=radius)
           ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
           ctrs.append(ctr.detach().contiguous().cpu().numpy())
       ctrs = torch.from_numpy(np.array(ctrs).astype(np.float32)).cuda()
       n_ctrs, _ = ctrs.size()
       pred_ctr_rp = pred_ctr.view(n_pts, 1, 3).repeat(1, n_ctrs, 1)
       ctrs_rp = ctrs.view(1, n_ctrs, 3).repeat(n_pts, 1, 1)
       ctr_dis = torch.norm((pred_ctr_rp - ctrs_rp), dim=2)
       min_dis, min_idx = torch.min(ctr_dis, dim=1)
       msk_closest_ctr = torch.LongTensor(pred_cls_ids).cuda()[min_idx]
       new_msk = mask.clone()
       for cls_id in pred_cls_ids:
           if cls_id == 0:
               break
           min_msk = min_dis < config.ycb_r_lst[cls_id-1] * 0.8
           update_msk = (mask > 0) & (msk_closest_ctr == cls_id) & min_msk
           new_msk[update_msk] = msk_closest_ctr[update_msk]
       mask = new_msk

    pred_pose_lst = []
    for icls, cls_id in enumerate(pred_cls_ids):
        if cls_id == 0:
            break
        cls_msk = mask == cls_id
        if cls_msk.sum() < 1:
            pred_pose_lst.append(np.identity(4)[:3,:])
            continue

        cls_voted_kps = pred_kp[:, cls_msk, :]
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id, n_kps, :] = ctr

        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id, ikp, :], _ = ms.fit(kps3d)

        mesh_kps = bs_utils.get_kps(cls_lst[cls_id-1])
        if use_ctr:
            mesh_ctr = bs_utils.get_ctr(cls_lst[cls_id-1]).reshape(1,3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        mesh_kps = torch.from_numpy(mesh_kps.astype(np.float32)).cuda()
        pred_RT = bs_utils.best_fit_transform(
            mesh_kps.contiguous().cpu().numpy(),
            cls_kps[cls_id].squeeze().contiguous().cpu().numpy()
        )
        pred_pose_lst.append(pred_RT)

    return (pred_cls_ids, pred_pose_lst)


def eval_metric(cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label):
    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break
        cls_msk = mask == cls_id

        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0:
            pred_RT = torch.zeros(3,4).cuda()
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
            pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
        gt_RT = RTs[icls]
        mesh_pts = bs_utils.get_pointxyz_cuda(cls_lst[cls_id-1]).clone()
        add = bs_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = bs_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        cls_add_dis[cls_id].append(add.item())
        cls_adds_dis[cls_id].append(adds.item())
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis)


def eval_one_frame_pose(
    item
):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs = item

    pred_cls_ids, pred_pose_lst = cal_frame_poses(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter
    )

    cls_add_dis, cls_adds_dis = eval_metric(
        cls_ids, pred_pose_lst, pred_cls_ids, RTs, mask, label
    )
    return (cls_add_dis, cls_adds_dis)


def cal_frame_poses_lm(
    pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter, obj_id
):
    n_kps, n_pts, _ = pred_kp_of.size()
    pred_ctr = pcld - ctr_of[0]
    pred_kp = pcld.view(1, n_pts, 3).repeat(n_kps, 1, 1) - pred_kp_of

    radius=0.08
    if use_ctr:
        cls_kps = torch.zeros(n_cls, n_kps+1, 3).cuda()
    else:
        ls_kps = torch.zeros(n_cls, n_kps, 3).cuda()

    pred_pose_lst = []
    cls_id = 1
    cls_msk = mask == cls_id
    if cls_msk.sum() < 1:
        pred_pose_lst.append(np.identity(4)[:3,:])
    else:
        cls_voted_kps = pred_kp[:, cls_msk, :]
        ms = MeanShiftTorch(bandwidth=radius)
        ctr, ctr_labels = ms.fit(pred_ctr[cls_msk, :])
        if ctr_labels.sum() < 1:
            ctr_labels[0] = 1
        if use_ctr:
            cls_kps[cls_id, n_kps, :] = ctr

        if use_ctr_clus_flter:
            in_pred_kp = cls_voted_kps[:, ctr_labels, :]
        else:
            in_pred_kp = cls_voted_kps

        for ikp, kps3d in enumerate(in_pred_kp):
            cls_kps[cls_id, ikp, :], _ = ms.fit(kps3d)

        mesh_kps = bs_utils_lm.get_kps(obj_id, ds_type="linemod")
        if use_ctr:
            mesh_ctr = bs_utils_lm.get_ctr(obj_id, ds_type="linemod").reshape(1,3)
            mesh_kps = np.concatenate((mesh_kps, mesh_ctr), axis=0)
        mesh_kps = torch.from_numpy(mesh_kps.astype(np.float32)).cuda()
        pred_RT = bs_utils_lm.best_fit_transform(
            mesh_kps.contiguous().cpu().numpy(),
            cls_kps[cls_id].squeeze().contiguous().cpu().numpy()
        )
        pred_pose_lst.append(pred_RT)
    return pred_pose_lst


def eval_metric_lm(cls_ids, pred_pose_lst, RTs, mask, label, obj_id):
    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]

    pred_RT = pred_pose_lst[0]
    pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()
    gt_RT = RTs[0]
    mesh_pts = bs_utils_lm.get_pointxyz_cuda(obj_id, ds_type="linemod").clone()
    add = bs_utils_lm.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
    adds = bs_utils_lm.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
    cls_add_dis[obj_id].append(add.item())
    cls_adds_dis[obj_id].append(adds.item())
    cls_add_dis[0].append(add.item())
    cls_adds_dis[0].append(adds.item())

    return (cls_add_dis, cls_adds_dis)


def eval_one_frame_pose_lm(
    item
):
    pcld, mask, ctr_of, pred_kp_of, RTs, cls_ids, use_ctr, n_cls, \
        min_cnt, use_ctr_clus_flter, label, epoch, ibs, obj_id = item
    pred_pose_lst = cal_frame_poses_lm(
        pcld, mask, ctr_of, pred_kp_of, use_ctr, n_cls, use_ctr_clus_flter,
        obj_id
    )

    cls_add_dis, cls_adds_dis = eval_metric_lm(
        cls_ids, pred_pose_lst, RTs, mask, label, obj_id
    )
    return (cls_add_dis, cls_adds_dis)


class TorchEval():

    def __init__(self):
        n_cls = 22
        self.n_cls = 22
        self.cls_add_dis = [list() for i in range(n_cls)]
        self.cls_adds_dis = [list() for i in range(n_cls)]
        self.cls_add_s_dis = [list() for i in range(n_cls)]
        self.sym_cls_ids = []

    def cal_auc(self):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        for cls_id in range(1, self.n_cls):
            if (cls_id) in config.ycb_sym_cls_ids:
                self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
            else:
                self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
            self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        for i in range(self.n_cls):
            add_auc = bs_utils.cal_auc(self.cls_add_dis[i])
            adds_auc = bs_utils.cal_auc(self.cls_adds_dis[i])
            add_s_auc = bs_utils.cal_auc(self.cls_add_s_dis[i])
            add_auc_lst.append(add_auc)
            adds_auc_lst.append(adds_auc)
            add_s_auc_lst.append(add_s_auc)
            if i == 0:
                continue
            print(cls_lst[i-1])
            print("***************add:\t", add_auc)
            print("***************adds:\t", adds_auc)
            print("***************add(-s):\t", add_s_auc)

        print("Average of all object:")
        print("***************add:\t", np.mean(add_auc_lst[1:]))
        print("***************adds:\t", np.mean(adds_auc_lst[1:]))
        print("***************add(-s):\t", np.mean(add_s_auc_lst[1:]))

        print("All object (following PoseCNN):")
        print("***************add:\t", add_auc_lst[0])
        print("***************adds:\t", adds_auc_lst[0])
        print("***************add(-s):\t", add_s_auc_lst[0])

        sv_info = dict(
            add_dis_lst = self.cls_add_dis,
            adds_dis_lst = self.cls_adds_dis,
            add_auc_lst = add_auc_lst,
            adds_auc_lst = adds_auc_lst,
            add_s_auc_lst = add_s_auc_lst,
        )
        sv_pth = os.path.join(
            config.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}.pkl'.format(
                adds_auc_lst[0], add_auc_lst[0], add_s_auc_lst[0]
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))

    def cal_lm_add(self, obj_id, test_occ=False):
        add_auc_lst = []
        adds_auc_lst = []
        add_s_auc_lst = []
        cls_id = obj_id
        if (obj_id) in config_lm.lm_sym_cls_ids:
            self.cls_add_s_dis[cls_id] = self.cls_adds_dis[cls_id]
        else:
            self.cls_add_s_dis[cls_id] = self.cls_add_dis[cls_id]
        self.cls_add_s_dis[0] += self.cls_add_s_dis[cls_id]
        add_auc = bs_utils_lm.cal_auc(self.cls_add_dis[cls_id])
        adds_auc = bs_utils_lm.cal_auc(self.cls_adds_dis[cls_id])
        add_s_auc = bs_utils_lm.cal_auc(self.cls_add_s_dis[cls_id])
        add_auc_lst.append(add_auc)
        adds_auc_lst.append(adds_auc)
        add_s_auc_lst.append(add_s_auc)
        d = config_lm.lm_r_lst[obj_id]['diameter'] / 1000.0 * 0.1
        print("obj_id: ", obj_id, "0.1 diameter: ", d)
        add = np.mean(np.array(self.cls_add_dis[cls_id]) < d) * 100
        adds = np.mean(np.array(self.cls_adds_dis[cls_id]) < d) * 100

        cls_type = config_lm.lm_id2obj_dict[obj_id]
        print(obj_id, cls_type)
        print("***************add auc:\t", add_auc)
        print("***************adds auc:\t", adds_auc)
        print("***************add(-s) auc:\t", add_s_auc)
        print("***************add < 0.1 diameter:\t", add)
        print("***************adds < 0.1 diameter:\t", adds)

        sv_info = dict(
            add_dis_lst = self.cls_add_dis,
            adds_dis_lst = self.cls_adds_dis,
            add_auc_lst = add_auc_lst,
            adds_auc_lst = adds_auc_lst,
            add_s_auc_lst = add_s_auc_lst,
            add = add,
            adds = adds,
        )
        occ = "occlusion" if test_occ else ""
        sv_pth = os.path.join(
            config_lm.log_eval_dir,
            'pvn3d_eval_cuda_{}_{}_{}_{}.pkl'.format(
                cls_type, occ, add, adds
            )
        )
        pkl.dump(sv_info, open(sv_pth, 'wb'))

    def eval_pose_parallel(
        self, pclds, rgbs, masks, pred_ctr_ofs, gt_ctr_ofs, labels, cnt,
        cls_ids, RTs, pred_kp_ofs, min_cnt=20, merge_clus=False, bbox=False, ds='YCB',
        cls_type=None, use_p2d = False, vote_type=VotingType.Farthest,
        use_ctr_clus_flter=True, use_ctr=True, ds_type="ycb", obj_id=0
    ):
        bs, n_kps, n_pts, c = pred_kp_ofs.size()
        masks = masks.long()
        cls_ids = cls_ids.long()
        use_ctr_lst = [use_ctr for i in range(bs)]
        n_cls_lst = [self.n_cls for i in range(bs)]
        min_cnt_lst = [min_cnt for i in range(bs)]
        epoch_lst = [cnt*bs for i in range(bs)]
        bs_lst = [i for i in range(bs)]
        use_ctr_clus_flter_lst = [use_ctr_clus_flter for i in range(bs)]
        obj_id_lst = [obj_id for i in range(bs)]
        if ds_type == "ycb":
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst
            )
        else:
            data_gen = zip(
                pclds, masks, pred_ctr_ofs, pred_kp_ofs, RTs,
                cls_ids, use_ctr_lst, n_cls_lst, min_cnt_lst, use_ctr_clus_flter_lst,
                labels, epoch_lst, bs_lst, obj_id_lst
            )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers= bs
        ) as executor:
            if ds_type == "ycb":
                eval_func = eval_one_frame_pose
            else:
                eval_func = eval_one_frame_pose_lm
            for res in executor.map(eval_func, data_gen):
                cls_add_dis_lst, cls_adds_dis_lst = res
                self.cls_add_dis = self.merge_lst(
                    self.cls_add_dis, cls_add_dis_lst
                )
                self.cls_adds_dis = self.merge_lst(
                    self.cls_adds_dis, cls_adds_dis_lst
                )

    def merge_lst(self, targ, src):
        for i in range(len(targ)):
            targ[i] += src[i]
        return targ

# vim: ts=4 sw=4 sts=4 expandtab
