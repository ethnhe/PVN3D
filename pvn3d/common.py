#!/usr/bin/env python3
import os
import yaml
import numpy as np


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))


class Config:
    def __init__(self, dataset_name='ycb', cls_type=''):
        self.dataset_name = dataset_name
        self.exp_dir = os.path.dirname(__file__)
        self.exp_name = os.path.basename(self.exp_dir)
        self.resnet_ptr_mdl_p = os.path.abspath(
            os.path.join(
                self.exp_dir,
                'lib/ResNet_pretrained_mdl'
            )
        )
        ensure_fd(self.resnet_ptr_mdl_p)

        # log folder
        self.cls_type = cls_type
        self.log_dir = os.path.abspath(
            os.path.join(self.exp_dir, 'train_log', dataset_name)
        )
        ensure_fd(self.log_dir)
        self.log_model_dir = os.path.join(self.log_dir, 'checkpoints', self.cls_type)
        ensure_fd(self.log_model_dir)
        self.log_eval_dir = os.path.join(self.log_dir, 'eval_results', self.cls_type)
        ensure_fd(self.log_eval_dir)

        self.n_total_epoch = 25
        self.mini_batch_size = 24
        self.num_mini_batch_per_epoch = 4000
        self.val_mini_batch_size = 24
        self.val_num_mini_batch_per_epoch = 125
        self.test_mini_batch_size = 1

        self.n_sample_points = 8192 + 4096
        self.n_keypoints = 8
        self.n_min_points = 400

        self.noise_trans = 0.05 # range of the random noise of translation added to the training data

        self.preprocessed_testset_pth = ''
        if self.dataset_name == 'ycb':
            self.n_objects = 21 + 1
            self.n_classes = 21 + 1
            self.ycb_cls_lst_p = os.path.abspath(
                os.path.join(
                    self.exp_dir, 'datasets/ycb/dataset_config/classes.txt'
                )
            )
            self.ycb_root = os.path.abspath(
                os.path.join(
                    self.exp_dir, 'datasets/ycb/YCB_Video_Dataset'
                )
            )
            self.ycb_kps_dir = os.path.abspath(
                os.path.join(
                    self.exp_dir, 'datasets/ycb/ycb_object_kps/'
                )
            )
            ycb_r_lst_p = os.path.abspath(
                os.path.join(
                    self.exp_dir, 'datasets/ycb/dataset_config/radius.txt'
                )
            )
            self.preprocessed_testset_pth = os.path.abspath(
                os.path.join(
                    self.exp_dir,
                    'datasets/ycb/YCB_Video_Dataset/preprocessed_valtestset.pkl'
                )
            )
            self.use_preprocess = True
            self.ycb_r_lst = list(np.loadtxt(ycb_r_lst_p))
            self.ycb_cls_lst = self.read_lines(self.ycb_cls_lst_p)
            self.ycb_sym_cls_ids = [13, 16, 19, 20, 21]
            self.val_test_pkl_p = os.path.join(
                self.exp_dir,
                'datasets/ycb/test_val_data_pts{}.pkl'.format(self.n_sample_points),
            )
        else: # linemod
            self.n_objects = 1 + 1
            self.n_classes = 1 + 1
            self.lm_cls_lst = [
                1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15
            ]
            self.lm_sym_cls_ids = [10, 11]
            self.lm_obj_dict={
                'ape':1,
                'benchvise':2,
                'cam':4,
                'can':5,
                'cat':6,
                'driller':8,
                'duck':9,
                'eggbox':10,
                'glue':11,
                'holepuncher':12,
                'iron':13,
                'lamp':14,
                'phone':15,
            }
            self.lm_id2obj_dict = dict(
                zip(self.lm_obj_dict.values(), self.lm_obj_dict.keys())
            )
            self.lm_root = os.path.abspath(
                os.path.join(self.exp_dir, 'datasets/linemod/')
            )
            self.lm_kps_dir = os.path.abspath(
                os.path.join(self.exp_dir, 'datasets/linemod/lm_obj_kps/')
            )
            self.lm_sym_cls_ids = [7, 8]
            self.val_test_pkl_p = os.path.join(
                self.exp_dir, 'datasets/linemod/test_val_data.pkl',
            )
            prep_fd = os.path.join(
                self.lm_root, "preprocess_testset"
            )
            ensure_fd(prep_fd)
            self.preprocessed_testset_ptn = os.path.abspath(
                os.path.join(prep_fd, '{}_pp_vts.pkl')
            )
            self.preprocessed_testset_pth = self.preprocessed_testset_ptn.format(cls_type)
            self.use_preprocess = False

            lm_r_pth = os.path.join(self.lm_root, "dataset_config/models_info.yml")
            lm_r_file = open(os.path.join(lm_r_pth), "r")
            self.lm_r_lst = yaml.load(lm_r_file)

            self.val_nid_ptn = "/data/6D_Pose_Data/datasets/LINEMOD/pose_nori_lists/{}_real_val.nori.list"

        self.intrinsic_matrix = {
            'linemod': np.array([[572.4114, 0.,         325.2611],
                                [0.,        573.57043,  242.04899],
                                [0.,        0.,         1.]]),
            'blender': np.array([[700.,     0.,     320.],
                                 [0.,       700.,   240.],
                                 [0.,       0.,     1.]]),
            'ycb_K1': np.array([[1066.778, 0.        , 312.9869],
                                [0.      , 1067.487  , 241.3109],
                                [0.      , 0.        , 1.0]], np.float32),
            'ycb_K2': np.array([[1077.836, 0.        , 323.7872],
                                [0.      , 1078.189  , 279.6921],
                                [0.      , 0.        , 1.0]], np.float32)
        }

    def read_lines(self, p):
        with open(p, 'r') as f:
            return [
                line.strip() for line in f.readlines()
            ]


config = Config()
# vim: ts=4 sw=4 sts=4 expandtab
