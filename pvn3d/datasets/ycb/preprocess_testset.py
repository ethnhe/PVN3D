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
from datasets.ycb.ycb_dataset import YCB_Dataset

config = Config(dataset_name='ycb')
bs_utils = Basic_Utils(config)
torch.multiprocessing.set_sharing_strategy('file_system')

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def main():
    if os.path.exists(config.preprocessed_testset_pth):
        return
    test_ds = YCB_Dataset('test')
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
            i_data = [item[ibs] for item in data]
            if len(i_data) < 11:
                print(len(i_data))
            data_lst.append(i_data)
    pkl.dump(data_lst, open(config.preprocessed_testset_pth, 'wb'))


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
