#!/usr/bin/env python3
import os
import cv2
import tqdm
import torch
import os.path
import numpy as np
from common import Config
import pickle as pkl
from lib.utils.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc
from datasets.linemod.linemod_dataset import LM_Dataset
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--cls_type", default="duck")
args = parser.parse_args()

config = Config(dataset_name='linemod', cls_type=args.cls_type)
bs_utils = Basic_Utils(config)
torch.multiprocessing.set_sharing_strategy('file_system')


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def pack_all():
    obj_lst = [
        'ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck',
        'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone',
    ]
    for cls_type in obj_lst:
        # test_ds = LM_Dataset('test', cls_type=args.cls_type)
        test_ds = LM_Dataset('test', cls_type=cls_type)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=config.test_mini_batch_size, shuffle=False,
            num_workers=40, worker_init_fn=worker_init_fn
        )
        data_lst = []
        for i, data in tqdm.tqdm(
            enumerate(test_loader), leave=False, desc='Preprocessing valtestset'
        ):
            bs, _, _, _ = data[0].shape
            for ibs in range(bs):
                # rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, ctr_targ_ofst, cls_ids, RTs, labels, kp_3ds, ctr_3ds
                i_data = [item[ibs] for item in data]
                data_lst.append(i_data)

        pkl.dump(
            data_lst,
            open(config.preprocessed_testset_ptn.format(cls_type), 'wb')
        )


def pack_arg():
    if os.path.exists(config.preprocessed_testset_pth):
        return
    cls_type = args.cls_type
    test_ds = LM_Dataset('test', cls_type=cls_type)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.test_mini_batch_size, shuffle=False,
        num_workers=40, worker_init_fn=worker_init_fn
    )
    data_lst = []
    for i, data in tqdm.tqdm(
        enumerate(test_loader), leave=False, desc='Preprocessing valtestset'
    ):
        bs, _, _, _ = data[0].shape
        for ibs in range(bs):
            # rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, ctr_targ_ofst, cls_ids, RTs, labels, kp_3ds, ctr_3ds
            i_data = [item[ibs].numpy() for item in data]
            data_lst.append(i_data)

            # Dubug
            # rgb = i_data[0].transpose((1, 2, 0)).astype("uint8")[:,:,::-1].copy()
            # labels = i_data[-1].astype("uint8")
            # labels = np.repeat(labels[:, :, None], 3, 2)
            # msked_rgb = rgb * labels
            # imshow("msked_rgb", msked_rgb)
            # imshow("rgb", rgb.astype("uint8"))
            # waitKey(0)

    pkl.dump(
        data_lst,
        open(config.preprocessed_testset_ptn.format(cls_type), 'wb')
    )


def main():
    pack_all()
    # pack_arg()


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
