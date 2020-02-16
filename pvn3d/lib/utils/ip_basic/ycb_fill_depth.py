#!/usr/bin/env python3
import os
import time

import cv2
import numpy as np
import pickle as pkl
import png

import nori2
from ip_basic import depth_map_utils, depth_map_utils_ycb
from ip_basic import vis_utils
import sys
sys.path.append('..')
from lib.utils.my_utils import my_utils
from neupeak.utils.webcv2 import imshow, waitKey
from tqdm import tqdm
import concurrent.futures


nf = nori2.Fetcher()


def show_depth(name, dpt):
    dpt = (dpt / np.max(dpt) * 255).astype(np.uint8)
    imshow(name, dpt)


def get_one_show(nid):
    fill_type = 'multiscale'
    # fill_type = 'fast'
    show_process = False
    extrapolate = True# False # True
    # blur_type = 'gaussian'
    blur_type = 'bilateral'

    data = pkl.loads(nf.get(nid))
    bk_label = data['label']
    bk_label = bk_label <= 0
    bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
    rgb_back = data['rgb'][:, :, :3] * bk_label_3c
    dpt_back = data['depth'].astype(np.float32) # * bk_label.astype(np.float32)
    cam_scale = data['meta']['factor_depth'].astype(np.float32)[0][0]
    scale_2_80 = 1 #80 / 4.6 # test
    scale_2_80 = 1 #80 / 6.6 # train_real

    dpt_back = dpt_back / cam_scale * scale_2_80

    pcld, choose = my_utils.dpt_2_cld(
        data['depth'], cam_scale, data['obj_info_lst'][0]['K']
    )
    nrm = my_utils.get_normal(pcld)
    nrm_map = my_utils.get_normal_map(nrm, choose)

    print('dpt range(min, max): ', np.min(dpt_back), np.max(dpt_back), cam_scale)

    projected_depth = dpt_back.copy()
    start_fill_time = time.time()
    if fill_type == 'fast':
        final_dpt = depth_map_utils_ycb.fill_in_fast(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            # max_depth=120.0
        )
    elif fill_type == 'multiscale':
        final_dpt, process_dict = depth_map_utils_ycb.fill_in_multiscale(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            show_process=show_process,
            # max_depth=120.0
        )
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))
    end_fill_time = time.time()
    pcld, choose = my_utils.dpt_2_cld(
        final_dpt, scale_2_80, data['obj_info_lst'][0]['K']
    )
    nrm = my_utils.get_normal(pcld)
    nrm_map_final = my_utils.get_normal_map(nrm, choose)

    show_dict = dict(
        ori_dpt = dpt_back,
        final_dpt = final_dpt,
        rgb = data['rgb'][:, :, :3][...,::-1].astype(np.uint8),
        nrm_map = nrm_map,
        nrm_map_final = nrm_map_final,
    )
    return show_dict

def complete_dpt(nid_p):
    nid_lst = my_utils.read_lines(nid_p)
    # fill_type = 'fast'
    cnt = 0
    import random
    # random.shuffle(nid_lst)
    with concurrent.futures.ProcessPoolExecutor(15) as executor:
        for info in executor.map(get_one_show, nid_lst):
            print(np.min(info['final_dpt']), np.max(info['final_dpt']))
            show_depth('ori_dpth', info['ori_dpt'])
            show_depth('cmplt_dpth', info['final_dpt'])
            imshow('rgb', info['rgb'])
            imshow('nrm_map', info['nrm_map'])
            imshow('nrm_map_final', info['nrm_map_final'])
            if cnt == 0:
                cmd = waitKey(0)
                # cnt += 1
            else:
                cmd = waitKey(2)


def get_one_depth(nid):
    data = pkl.loads(nf.get(nid))
    dpt_back = data['depth'].astype(np.float32) # * bk_label.astype(np.float32)
    cam_scale = data['meta']['factor_depth'].astype(np.float32)[0][0]
    # K = data['obj_info_lst'][0]['K']
    # print(K)
    dpt_back = dpt_back / cam_scale
    dpt_back = dpt_back.reshape(-1)
    max_dpt = dpt_back[np.argpartition(dpt_back, -100)[-100:]]
    return np.mean(max_dpt)


def get_depth_max_statics(nid_p):
    print(nid_p)
    nid_lst = my_utils.read_lines(nid_p)
    # nid_lst = nid_lst[:2] + nid_lst[-2:]
    max_dp = 0.0
    with concurrent.futures.ProcessPoolExecutor(15) as executor:
        for dpt in tqdm(executor.map(get_one_depth, nid_lst)):
            if dpt > max_dp:
                max_dp = dpt
    print("max_dp: ", max_dp)


def main():
    nid_lst_p_lst = [
        '/data/ycb_linemod_datasets/ycb/pose_nori_lists/allobj_test_real.nori.list',
        '/data/ycb_linemod_datasets/ycb/pose_nori_lists/allobj_train_real.nori.list',
        '/data/ycb_linemod_datasets/ycb/pose_nori_lists/allobj_train_syn.nori.list',
        # '/data/ycb_linemod_datasets/ycb/ycb_train_rdlm.nori.list',
        # '/data/ycb_linemod_datasets/ycb/ycb_train_syn_rdlm.nori.list',
        # '/data/ycb_linemod_datasets/ycb/ycb_train_real_rdlm.nori.list',
        # '/data/ycb_linemod_datasets/ycb/ycb_test_rdlm.nori.list'
    ]
    complete_dpt(nid_lst_p_lst[0])
    for nid_lst_p in nid_lst_p_lst:
        get_depth_max_statics(nid_lst_p)


if __name__ == "__main__":
    main()


# vim: ts=4 sw=4 sts=4 expandtab

