#!/usr/bin/env python3
import os
import cv2
import random
from random import shuffle
import os.path
import nori2
import numpy as np
import pickle as pkl
from meghair.train.base import DatasetMinibatch
from neupeak.dataset.meta import (
    GeneratorDataset, EpochDataset, StackMinibatchDataset
)
from neupeak.dataset.server import create_servable_dataset
from neupeak.utils.misc import stable_rng
import torchvision.transforms as transforms
from PIL import Image
from common import Config
from argparse import ArgumentParser
import sys
from tqdm import tqdm
import pcl

cls_type = 'cat'
config = Config(cls_type)


class LM_Dataset():

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.diameters = {}

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
        self.obj_dict={
            'ape':1,
            'cam':2,
            'cat':3,
            'duck':4,
            'glue':5,
            'iron':6,
            'phone':7,
            'benchvise':8,
            'can':9,
            'driller':10,
            'eggbox':11,
            'holepuncher':12,
            'lamp':13,
        }

        self.rng = stable_rng(random.random())
        self.nf = nori2.Fetcher()
        self.cls_id = self.obj_dict[cls_type]

        if dataset_name == 'train':
            self.nid_path = config.train_nid_path
            config.add_noise = False
        elif dataset_name == 'val':
            self.nid_path = config.validation_nid_path
            config.add_noise = False
        else:
            self.nid_path = config.test_nid_path
            config.add_noise = False

        if 'zbuf' in self.nid_path:
            self.depth_scale = 1.0
            self.mm2m = 1.0
        else:
            self.depth_scale = 2.0
            self.mm2m = 1000.0
        dataset_items = self.read_nid_list(self.nid_path)
        if len(config.fuse_nid_path) > 0:
            fuse_nid_list = self.read_nid_list(config.fuse_nid_path)
        if dataset_name == 'validation':
            if os.path.exists(config.validation_preproc_nid_path):
                self.dataset_items = self.read_nid_list(config.validation_preproc_nid_path)
                print('{} val nori exists'.format(cls_type))
            else:
                print('packing {} val nori'.format(cls_type))
                self.dataset_items = self.pack_preproc(dataset_items, "val")
                self.save_nid_list(config.validation_preproc_nid_path, self.dataset_items)
        elif dataset_name == 'test':
            if os.path.exists(config.test_preproc_nid_path):
                self.dataset_items = self.read_nid_list(config.test_preproc_nid_path)
                print('{} test nori exists'.format(cls_type))
            else:
                print('packing {} test nori'.format(cls_type))
                self.dataset_items = self.pack_preproc(dataset_items, "test")
                self.save_nid_list(config.test_preproc_nid_path, self.dataset_items)
        else:
            if len(fuse_nid_list) > 0:
                self.dataset_items = dataset_items + fuse_nid_list
                shuffle(self.dataset_items)
            else:
                self.dataset_items = dataset_items

    def save_nid_list(self, path, nid_list):
        with open(path, 'w') as f:
            for nid in nid_list:
                print(nid, file=f)

    def read_nid_list(self, p):
        nid_list = [
            line.strip() for line in open(p, 'r').readlines()
        ]
        return nid_list

    def msk_depth_to_pcld(self, msk, dpt, K):
        ys, xs = np.nonzero(msk)
        dpts = dpt[ys, xs]
        xs, ys = np.asarray(xs, np.float32), np.asarray(ys, np.float32)
        xys = np.concatenate([xs[:, None], ys[:, None]], 1)
        xys *= dpts[:, None]
        xyds = np.concatenate([xys, dpts[:, None]], 1)
        pts = np.matmul(xyds, np.linalg.inv(K).transpose())
        return pts

    def __get_item__(self, data):
        if data['rnd_typ'] == 'render':
            data['depth'] *= self.depth_scale
        elif data['rnd_typ'] in ['real', 'fuse']:
            data['rgb'] = data['rgb'][...,::-1]
            if data['rnd_typ'] == 'fuse':
                msk = (data['mask']==self.cls_id).astype(data['mask'].dtype)
                msk_3c = np.repeat(msk[:,:,None], 3, 2)
            else:
                msk_3c = data['mask']
                msk_3c = (msk_3c > 0).astype(msk_3c.dtype)
                msk = msk_3c[:, :, 0]

        rgb = data['rgb']
        dpt = data['depth']

        flter = np.where(msk > 0)
        if len(flter) == 0 or len(flter[0]) < 3 or len(flter[1]) < 3:
            return None, None, None, None, None, \
                None, None, None, None, None
        rmin, rmax, cmin, cmax = flter[0].min(), flter[0].max(), flter[1].min(), flter[1].max()

        if config.add_noise:
            rgb = Image.fromarray(np.uint8(rgb))
            rgb = self.trancolor(rgb)
            rgb = np.asarray(rgb)

        rgb = rgb * msk_3c.astype(rgb.dtype)

        rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw

        choose = msk.flatten().nonzero()[0]
        if len(choose) == 0:
            return None, None, None, None, None, \
                None, None, None, None, None
        if len(choose) > config.n_sample_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, config.n_sample_points-len(choose)), 'wrap')

        dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)

        xmap_mskd = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_mskd = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

        choose = np.array([choose])

        K = data['K']
        RT_m = data['RT']
        T = RT_m[:3, 3]
        cam_scale = 1.0
        pt2 = dpt_mskd / cam_scale
        cam_cx, cam_cy = K[0][2], K[1][2]
        cam_fx, cam_fy = K[0][0], K[1][1]
        pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
        cld = np.concatenate( (pt0, pt1, pt2), axis=1 )
        cld = np.add(cld, -1.0 * T * self.mm2m) / self.mm2m
        cld = np.add(cld, T)
        if 'center_3d' not in data.keys():
            tmp_d = np.ones((1, 1))
            data['center_3d'] = np.concatenate((data['center'], tmp_d), axis=1)

        key_kpts = ''
        if config.n_keypoints == 8:
            key_kpts = 'farthest_3d'
        else:
            key_kpts = 'farthest{}_3d'.format(config.n_keypoints)
        kpts = data[key_kpts]
        kpts = np.array(kpts)

        target = []
        for kpt in kpts:
            tmp_cld = cld.copy()
            target.append(np.add(tmp_cld, -1.0*kpt))

        target_offset = np.array(target)

        bbx = np.array((rmin, rmax, cmin, cmax))
        idx = np.array(self.obj_dict[data['cls_typ']])
        return rgb, cld, choose, kpts, target_offset, idx, bbx, \
               np.array(data['center_3d']), np.array(data['center']), \
               np.array(data['farthest'])

    def get_once(self):
        config.add_noise = False
        for nid in self.dataset_items:
            data = pkl.loads(self.nf.get(nid))
            # data = self.clip_outlier(data)
            if 'RT' in data.keys():
                RT = data['RT']
            else:
                RT = np.zeros(3, 4)
            yield dict(
                rgb = data['rgb'],
                pcld = data['pcld'],
                choose = data['choose'],
                kpts = data['kpts'],
                target_offsets = data['target_offsets'],
                idx = data['idx'],
                bbx = data['bbx'],
                ctr_3d = data['ctr_3d'],
                ctr_2d = data['ctr_2d'],
                kpts_2d = data['kpts_2d'],
                RT = RT
            )

    def get_diameter(self, class_type):
        if class_type in self.diameters:
            return self.diameters[class_type]
        self.diameter_pattern = os.path.join('/data/6D_Pose_Data/LINEMOD_ORIG','{}/distance.txt')
        diameter_path = self.diameter_pattern.format(class_type)
        diameter = np.loadtxt(diameter_path) / 100.
        self.diameters[class_type] = diameter
        return diameter

    def clip_outlier(self, data):
        pcld = np.array(data['pcld']),
        choose = np.array(data['choose']),
        kpts = np.array(data['kpts']),
        target_offsets = np.array(data['target_offsets']),
        bbx = data['bbx'],
        diameter = self.get_diameter(cls_type)
        dpts = pcld[0][:,2].copy()
        pcld_sorted = sorted(list(dpts))
        n_pts = len(pcld_sorted)
        n_out = int(n_pts*0.4/2.0)
        mid = np.array(pcld_sorted[n_out:-n_out]).mean()
        interval = diameter / 2.0 * 1.2
        front = mid - interval
        back = mid + interval
        pcld, choose, target_offsets = pcld[0], choose[0][0], target_offsets[0]
        print(pcld.shape, choose.shape, target_offsets.shape)
        good = (pcld[:,2] > front) & (pcld[:,2] < back)
        selected = np.where(good)
        data['pcld'] = pcld[selected, :]
        data['choose'] = choose[selected]
        data['target_offsets'] = target_offsets[:, selected, :]
        return data


    def pack_preproc(self, dataset_items, ds_type):
        if ds_type == 'test':
            nr_nm = config.test_preproc_nid_path.split('/')[-1].replace(".list","")
        else:
            nr_nm = config.validation_preproc_nid_path.split('/')[-1].replace(".list","")

        nori_p = config.val_test_pp_nori_p + "/" + nr_nm
        os.system("oss rm --recursive {}".format(nori_p))
        nw = nori2.open(nori_p, 'w')
        new_nid_list = []
        for item in tqdm(dataset_items):
            data = pkl.loads(self.nf.get(item))
            rgb, pcld, choose, kpts, target_offset, idx, bbx, ctr_3d, ctr_2d, kpts_2d = self.__get_item__(data)
            if rgb is None:
                continue
            data = dict(
                rgb = rgb,
                pcld = pcld,
                choose = choose,
                kpts = kpts,
                target_offsets = target_offset,
                idx = idx,
                bbx = bbx,
                ctr_3d = ctr_3d,
                ctr_2d = ctr_2d,
                kpts_2d= kpts_2d,
                RT = data['RT']
            )
            new_nid_list.append(nw.put(pkl.dumps(data)))
        nw.close()
        os.system("nori speedup --on {}".format(nori_p))
        return new_nid_list

    def get(self):
        if self.dataset_name == 'train':
            mini_batch_size = config.mini_batch_size
            num_mini_batch_per_epoch = config.num_mini_batch_per_epoch
            num_images_per_epoch = config.num_images_per_epoch
        else:
            mini_batch_size = config.val_mini_batch_size
            num_mini_batch_per_epoch = config.val_num_mini_batch_per_epoch
            num_images_per_epoch = mini_batch_size * num_mini_batch_per_epoch

        def generator():
            if self.dataset_name == 'train':
                while True:
                    item = self.dataset_items[self.rng.randint(len(self.dataset_items))]
                    data = pkl.loads(self.nf.get(item))
                    rgb, pcld, choose, kpts, target_offset, idx, bbx, ctr_3d, ctr_2d, kpts_2d = self.__get_item__(data)
                    if rgb is None:
                        continue
                    yield DatasetMinibatch(
                        rgb=rgb,
                        pcld=pcld,
                        choose=choose,
                        kpts=kpts,
                        target_offsets=target_offset,
                        idx=idx,
                        bbx=bbx,
                        ctr_3d = ctr_3d,
                        ctr_2d = ctr_2d,
                        kpts_2d= kpts_2d,
                        RT = data['RT'],
                        check_minibatch_size=False
                    )
            else:
                while True:
                    for nid in self.dataset_items:
                        data = pkl.loads(self.nf.get(nid))
                        if 'RT' in data.keys():
                            RT = data['RT']
                        else:
                            RT = np.zeros(3, 4)
                        # data = self.clip_outlier(data)
                        yield DatasetMinibatch(
                            rgb = data['rgb'],
                            pcld = data['pcld'],
                            choose = data['choose'],
                            kpts = data['kpts'],
                            target_offsets = data['target_offsets'],
                            idx = data['idx'],
                            bbx = data['bbx'],
                            ctr_3d = data['ctr_3d'],
                            ctr_2d = data['ctr_2d'],
                            kpts_2d = data['kpts_2d'],
                            RT = RT,
                            check_minibatch_size=False
                        )

        service_name = config.make_service_name(self.dataset_name)
        dataset = GeneratorDataset(generator)
        dataset = EpochDataset(dataset, num_images_per_epoch)
        dataset = StackMinibatchDataset(dataset, mini_batch_size)
        dataset = create_servable_dataset(
            dataset, service_name,
            num_mini_batch_per_epoch,
            serve_type='combiner'
        )
        return dataset


def get(dataset_name):
    ds = Dataset(dataset_name)
    return ds.get()


def main():
    config.mini_batch_size = 1
    ds = {}
    ds['train'] = Dataset('train')
    ds['val'] = Dataset('validation')
    ds['test'] = Dataset('test')
    while True:
        for cat in ['train', 'val', 'test']:
            ds_ = ds[cat].get()
            gen = ds_.get_epoch_minibatch_iter()
            datum = next(gen)
            brgb = datum['rgb']
            bs, c, h, w = brgb.shape
            for i in range(bs):
                rgb = brgb[i, :, :, :].transpose(1, 2, 0).copy()
                cv2.imshow('rgb', rgb)
                cmd = cv2.waitKey(10)
                if cmd == ord('q'):
                    exit()


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
