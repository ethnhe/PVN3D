# PVN3D
This is the source code for PVN3D: A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose Estimation ([PDF](https://arxiv.org/abs/1911.04231), [Video](https://www.bilibili.com/video/av89408773/)).

## Installation
- Install CUDA9.0/CUDA10.0
- Set up python environment from requirement.txt:
  ```shell
  pip3 install -r requirement.txt 
  ```
- Install tkinter through ``sudo apt install python3-tk``
- Install [python-pcl](https://github.com/strawlab/python-pcl).
- Install PointNet++:
  ```shell
  python3 setup.py build_ext
  ```

## Datasets
- Download the YCB-Video Dataset from [PoseCNN](https://rse-lab.cs.washington.edu/projects/posecnn/). Unzip it and link the unzipped```YCB_Video_Dataset``` to ```pvn3d/datasets/ycb/YCB_Video_Dataset```:

  ```
  ln -s path_to_unziped_YCB_Video_Dataset pvn3d/datasets/ycb
  ```


## Training and evaluating
### Training on the YCB-Video Dataset
- Preprocess the validation set to speed up training:
  ```shell
  cd pvn3d
  python3 -m datasets.ycb.preprocess_testset
  ```
- Start training on the YCB-Video Dataset by:
  ```shell
  chmod +x ./train_ycb.sh
  ./train_ycb
  ```
  The trained model checkpoints are stored in ``train_log/ycb/checkpoints/``
### Evaluating on the YCB-Video Dataset
- Start evaluating by:
  ```shell
  chmod +x ./eval_ycb.sh
  ./eval_ycb.sh
  ```
  You can evaluate different checkpoint by revising the ``tst_mdl`` in ``eval_ycb.sh`` to path of your target model.
- We provide our pre-trained models [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yhebk_connect_ust_hk/ElLgjzbbENZGhf-Sn8e4CMgBzd9zDjcJpCXFmB4n0WVw_w?e=IHMkvh). Download the ycb pre-trained model, move it to ``train_log/ycb/checkpoints/`` and modify ``tst_mdl`` in ``eval_ycb.sh`` for testing.

## Ongoing
- [ ] Scripts for synthesis data in LineMOD dataset.
- [ ] Training code and pre-trained models for the LineMOD dataset.

## Citations:
Please cite [PVN3D](https://arxiv.org/abs/1911.04231) if you use this repository in your publications:
```
@article{he2019pvn3d,
  title={PVN3D: A Deep Point-wise 3D Keypoints Voting Network for 6DoF Pose Estimation},
  author={He, Yisheng and Sun, Wei and Huang, Haibin and Liu, Jianran and Fan, Haoqiang and Sun, Jian},
  journal={arXiv preprint arXiv:1911.04231},
  year={2019}
}
```
