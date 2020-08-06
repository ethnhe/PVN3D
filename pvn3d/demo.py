#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import tqdm
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
import pickle as pkl
from common import Config
from lib import PVN3D
from datasets.ycb.ycb_dataset import YCB_Dataset
from datasets.linemod.linemod_dataset import LM_Dataset
from lib.utils.sync_batchnorm import convert_model
from lib.utils.pvn3d_eval_utils import cal_frame_poses, cal_frame_poses_lm
from lib.utils.basic_utils import Basic_Utils
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except:
    from cv2 import imshow, waitKey


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to eval"
)
parser.add_argument(
    "-dataset", type=str, default="linemod",
    help="Target dataset, ycb or linemod. (linemod as default)."
)
parser.add_argument(
    "-cls", type=str, default="ape",
    help="Target object to eval in LineMOD dataset. (ape, benchvise, cam, can," +
    "cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)"
)
args = parser.parse_args()

if args.dataset == "ycb":
    config = Config(dataset_name=args.dataset)
else:
    config = Config(dataset_name=args.dataset, cls_type=args.cls)
bs_utils = Basic_Utils(config)


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))


def checkpoint_state(model=None, optimizer=None, best_prec=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None
    return {
        "epoch": epoch,
        "it": it,
        "best_prec": best_prec,
        "model_state": model_state,
        "optimizer_state": optim_state,
    }


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(filename))
        try:
            checkpoint = torch.load(filename)
        except:
            checkpoint = pkl.load(open(filename, "rb"))
        epoch = checkpoint["epoch"]
        it = checkpoint.get("it", 0.0)
        best_prec = checkpoint["best_prec"]
        if model is not None and checkpoint["model_state"] is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("==> Done")
        return it, epoch, best_prec
    else:
        print("==> Checkpoint '{}' not found".format(filename))
        return None


def cal_view_pred_pose(model, data, epoch=0, obj_id=-1):
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = [item.to("cuda", non_blocking=True) for item in data]
        rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, ctr_targ_ofst, \
            cls_ids, rts, labels, kp_3ds, ctr_3ds = cu_dt

        pred_kp_of, pred_rgbd_seg, pred_ctr_of = model(
            cld_rgb_nrm, rgb, choose
        )
        _, classes_rgbd = torch.max(pred_rgbd_seg, -1)

        if args.dataset == "ycb":
            pred_cls_ids, pred_pose_lst = cal_frame_poses(
                pcld[0], classes_rgbd[0], pred_ctr_of[0], pred_kp_of[0], True,
                config.n_objects, True
            )
        else:
            pred_pose_lst = cal_frame_poses_lm(
                pcld[0], classes_rgbd[0], pred_ctr_of[0], pred_kp_of[0], True,
                config.n_objects, False, obj_id
            )
            pred_cls_ids = np.array([[1]])

        np_rgb = rgb.cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
        if args.dataset == "ycb":
            np_rgb = np_rgb[:, :, ::-1].copy()
        ori_rgb = np_rgb.copy()
        for cls_id in cls_ids[0].cpu().numpy():
            idx = np.where(pred_cls_ids == cls_id)[0]
            if len(idx) == 0:
                continue
            pose = pred_pose_lst[idx[0]]
            if args.dataset == "ycb":
                obj_id = int(cls_id[0])
            mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type=args.dataset).copy()
            mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
            if args.dataset == "ycb":
                K = config.intrinsic_matrix["ycb_K1"]
            else:
                K = config.intrinsic_matrix["linemod"]
            mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
            color = bs_utils.get_label_color(obj_id, n_obj=22, mode=1)
            np_rgb = bs_utils.draw_p2ds(np_rgb, mesh_p2ds, color=color)
        vis_dir = os.path.join(config.log_eval_dir, "pose_vis")
        ensure_fd(vis_dir)
        f_pth = os.path.join(vis_dir, "{}.jpg".format(epoch))
        cv2.imwrite(f_pth, np_rgb)
        # imshow("projected_pose_rgb", np_rgb)
        # imshow("ori_rgb", ori_rgb)
        # waitKey(1)
    if epoch == 0:
        print("\n\nResults saved in {}".format(vis_dir))


def main():
    if args.dataset == "ycb":
        test_ds = YCB_Dataset('test')
        obj_id = -1
    else:
        test_ds = LM_Dataset('test', cls_type=args.cls)
        obj_id = config.lm_obj_dict[args.cls]
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.test_mini_batch_size, shuffle=False,
        num_workers=20
    )

    model = PVN3D(
        num_classes=config.n_objects, pcld_input_channels=6, pcld_use_xyz=True,
        num_points=config.n_sample_points
    ).cuda()
    model = convert_model(model)
    model.cuda()

    # load status from checkpoint
    if args.checkpoint is not None:
        checkpoint_status = load_checkpoint(
            model, None, filename=args.checkpoint[:-8]
        )
    model = nn.DataParallel(model)

    for i, data in tqdm.tqdm(
        enumerate(test_loader), leave=False, desc="val"
    ):
        cal_view_pred_pose(model, data, epoch=i, obj_id=obj_id)


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
