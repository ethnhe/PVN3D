#!/usr/bin/env python3
import os
import numpy as np
import glob
from plyfile import PlyData
from tqdm import tqdm
import pickle as pkl
from fps.fps_utils import farthest_point_sampling
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("ply_pth", type=str, help="path to the input ply mesh model.")
parser.add_argument("sv_fd", type=str, help="path to save the generated mesh info.")
parser.print_help()
args = parser.parse_args()


# Read object vertexes from ply file
def get_p3ds_from_ply(ply_pth):
    print("loading p3ds from ply:", ply_pth)
    ply = PlyData.read(ply_pth)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    p3ds = np.stack([x, y, z], axis=-1)
    print("finish loading ply.")
    return p3ds


# Read object vertexes from text file
def get_p3ds_from_txt(pxyz_pth):
    pointxyz = np.loadtxt(pxyz_pth, dtype=np.float32)
    return pointxyz


# Compute the 3D bounding box from object vertexes
def get_corners_3d(p3ds, small=False):
    x = p3ds[:, 0]
    min_x, max_x = np.min(x), np.max(x)
    y = p3ds[:, 1]
    min_y, max_y = np.min(y), np.max(y)
    z = p3ds[:, 2]
    min_z, max_z = np.min(z), np.max(z)
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    if small:
        center = np.mean(corners_3d, 0)
        corners_3d = (corners_3d - center[None,:]) * 2.0 / 3.0 + center[None,:]
    return corners_3d


# Compute the radius of object
def get_radius(corners_3d):
    radius = np.linalg.norm(np.max(corners_3d, 0)-np.min(corners_3d, 0)) / 2.0
    return radius


# Compute the center of object
def get_centers_3d(corners_3d):
    centers_3d=(np.max(corners_3d, 0) + np.min(corners_3d, 0)) / 2
    return centers_3d


# Select keypoint with Farthest Point Sampling (FPS) algorithm
def get_farthest_3d(p3ds, num=8, init_center=False):
    fps = farthest_point_sampling(p3ds, num, init_center=init_center)
    return fps


# Compute and save all mesh info
def gen_one_mesh_info(ply_pth, sv_fd):
    if not os.path.exists(sv_fd):
        os.system("mkdir -p %s" % sv_fd)
    p3ds = get_p3ds_from_ply(ply_pth)

    c3ds = get_corners_3d(p3ds)
    c3ds_pth = os.path.join(sv_fd, "corners.txt")
    with open(c3ds_pth, 'w') as of:
        for p3d in c3ds:
            print(p3d[0], p3d[1], p3d[2], file=of)

    radius = get_radius(c3ds)
    r_pth = os.path.join(sv_fd, "radius.txt")
    with open(r_pth, 'w') as of:
        print(radius, file=of)

    ctr = get_centers_3d(c3ds)
    ctr_pth = os.path.join(sv_fd, "center.txt")
    with open(ctr_pth, 'w') as of:
        print(ctr[0], ctr[1], ctr[2], file=of)

    fps = get_farthest_3d(p3ds, num=8)
    fps_pth = os.path.join(sv_fd, "farthest.txt")
    with open(fps_pth, 'w') as of:
        for p3d in fps:
            print(p3d[0], p3d[1], p3d[2], file=of)


def test():
    ply_pth = '../../../datasets/ycb/YCB_Video_Dataset/models/002_master_chef_can/textured.ply'
    gen_one_mesh_info(ply_pth, 'mesh_info/002_master_chef_can')


def main():
    # test()
    gen_one_mesh_info(args.ply_pth, args.sv_fd)


if __name__ == "__main__":
    main()


# vim: ts=4 sw=4 sts=4 expandtab
