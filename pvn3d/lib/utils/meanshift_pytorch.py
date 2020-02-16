#!/usr/bin/env python3
import os
import torch
import cv2
import numpy as np
import pickle as pkl
import time
from sklearn.cluster import MeanShift
from neupeak.utils.webcv2 import imshow, waitKey
# from lib.utils.my_utils import my_utils


def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * torch.sqrt(2 * torch.tensor(np.pi)))) \
        * torch.exp(-0.5 * ((distance / bandwidth)) ** 2)


class MeanShiftTorch():
    def __init__(self, bandwidth=0.05, max_iter=300):
        self.bandwidth = bandwidth
        self.stop_thresh = bandwidth * 1e-3
        self.max_iter = max_iter

    def fit(self, A):
        """
        params: A: [N, 3]
        """
        N, c = A.size()
        it = 0
        C = A.clone()
        while True:
            it += 1
            Ar = A.view(1, N, c).repeat(N, 1, 1)
            Cr = C.view(N, 1, c).repeat(1, N, 1)
            dis = torch.norm(Cr - Ar, dim=2)
            w = gaussian_kernel(dis, self.bandwidth).view(N, N, 1)
            new_C = torch.sum(w * Ar, dim=1) / torch.sum(w, dim=1)
            # new_C = C + shift_offset
            Adis = torch.norm(new_C - C, dim=1)
            # print(C, new_C)
            C = new_C
            if torch.max(Adis) < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        Cr = A.view(N, 1, c).repeat(1, N, 1)
        dis = torch.norm(Ar - Cr, dim=2)
        num_in = torch.sum(dis < self.bandwidth, dim=1)
        max_num, max_idx = torch.max(num_in, 0)
        labels = dis[max_idx] < self.bandwidth
        return C[max_idx, :], labels

    def fit_batch_npts(self, A):
        """
        params: A: [bs, n_kps, pts, 3]
        """
        bs, n_kps, N, cn = A.size()
        it = 0
        C = A.clone()
        while True:
            it += 1
            Ar = A.view(bs, n_kps, 1, N, cn).repeat(1, 1, N, 1, 1)
            Cr = C.view(bs, n_kps, N, 1, cn).repeat(1, 1, 1, N, 1)
            dis = torch.norm(Cr - Ar, dim=4)
            w = gaussian_kernel(dis, self.bandwidth).view(bs, n_kps, N, N, 1)
            new_C = torch.sum(w * Ar, dim=3) / torch.sum(w, dim=3)
            # new_C = C + shift_offset
            Adis = torch.norm(new_C - C, dim=3)
            # print(C, new_C)
            C = new_C
            if torch.max(Adis) < self.stop_thresh or it > self.max_iter:
                # print("torch meanshift total iter:", it)
                break
        # find biggest cluster
        Cr = A.view(N, 1, c).repeat(1, N, 1)
        dis = torch.norm(Ar - Cr, dim=4)
        num_in = torch.sum(dis < self.bandwidth, dim=3)
        # print(num_in.size())
        max_num, max_idx = torch.max(num_in, 2)
        dis = torch.gather(dis, 2, max_idx.reshape(bs, n_kps, 1))
        labels = dis < self.bandwidth
        ctrs = torch.gather(
            C, 2, max_idx.reshape(bs, n_kps, 1, 1).repeat(1, 1, 1, cn)
        )
        return ctrs, labels


def test():
    while True:
        a = np.random.rand(1000, 2)
        ta = torch.from_numpy(a.astype(np.float32)).cuda()
        ms = MeanShiftTorch(0.05)
        ctr, _ = ms.fit(ta)
        a_idx = (a * 480).astype("uint8")
        show_a = np.zeros((480, 480, 3), dtype="uint8")
        show_a[a_idx[:, 0], a_idx[:, 1], :] = np.array([255, 255, 255])
        ctr = (ctr.cpu().numpy() * 480).astype("uint8")
        show_a = cv2.circle(show_a, (ctr[1], ctr[0]), 3, (0, 0, 255), -1)

        ms_cpu = MeanShift(
            bandwidth=0.05, n_jobs=8
        )
        ms_cpu.fit(a)
        clus_ctrs = np.array(ms_cpu.cluster_centers_)
        clus_labels = ms_cpu.labels_
        ctr = (clus_ctrs[0] * 480).astype("uint8")
        show_a = cv2.circle(show_a, (ctr[1], ctr[0]), 3, (255, 0, 0), -1)
        imshow('show_a', show_a)
        waitKey(0)
        print(clus_ctrs[0])


def test2():
    sv_ptn = '/data/workspace/3D_Point_Det/config/ycb.onestage.rs14.nofarflatFocalls/train_log/eval_result/051_large_clamp/mask_res_pic/{}sv_info_1.pkl'
    for i in range(2000):
        data = pkl.load(open(sv_ptn.format(i), 'rb'))
        all_p3ds = data['p3ds']

        for cls_id in data['gt_cls_ids'][0]:
            if cls_id == 0:
                break
            p3ds = all_p3ds[np.where(data['labels'] == cls_id)[0], :]
            show_img = np.zeros((480, 640, 3), dtype="uint8")
            p2ds = my_utils.project_p3d(p3ds, 1.0)
            show_img[p2ds[:, 1], p2ds[:, 0], :] = np.array([255, 255, 255])
            gpu_label = np.zeros((480, 640, 3), dtype="uint8")
            cpu_label = gpu_label.copy()
            p3ds_cu = torch.from_numpy(p3ds).cuda()
            ms_gpu = MeanShiftTorch(0.05)

            start = time.time()
            ctr, labels = ms_gpu.fit(p3ds_cu)
            ctr = ctr.cpu().numpy().reshape(1, 3)
            labels = labels.cpu().numpy()
            p2ds_gt_lb = p2ds[np.where(labels==1)[0], :]
            gpu_label[p2ds_gt_lb[:, 1], p2ds_gt_lb[:, 0], :] = np.array(
                [255, 255, 255]
            )
            end = time.time()
            print("gpu time:\t", end - start)
            ctr_2d = my_utils.project_p3d(ctr, 1.0)
            show_img = cv2.circle(
                show_img, (ctr_2d[0][0], ctr_2d[0][1]), 3, (0, 0, 255), -1
            )

            ms_cpu = MeanShift(
                bandwidth=0.05, n_jobs=40
            )
            start = time.time()
            ms_cpu.fit(p3ds)
            end = time.time()
            print("sklearn cpu time:\t", end - start)
            clus_ctrs = np.array(ms_cpu.cluster_centers_)
            clus_labels = ms_cpu.labels_
            ctr_2d = my_utils.project_p3d(clus_ctrs[0].reshape(1, 3), 1.0)
            show_img = cv2.circle(
                show_img, (ctr_2d[0][0], ctr_2d[0][1]), 3, (255, 0, 0), -1
            )
            p2ds_gt_lb = p2ds[np.where(clus_labels==0)[0], :]
            cpu_label[p2ds_gt_lb[:, 1], p2ds_gt_lb[:, 0], :] = np.array(
                [255, 255, 255]
            )
            imshow('show_img', show_img)
            imshow('gpu', gpu_label)
            imshow('cpu', cpu_label)
            waitKey(0)


def main():
    test2()


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
