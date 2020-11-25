#!/usr/bin/env python3
import os
import numpy as np
import cv2
from lib.utils.ip_basic.ip_basic import depth_map_utils_ycb as depth_map_utils
from lib.utils.ip_basic.ip_basic import vis_utils
from plyfile import PlyData
import random
import torch


intrinsic_matrix = {
    'linemod': np.array([[572.4114, 0.,         325.2611],
                        [0.,        573.57043,  242.04899],
                        [0.,        0.,         1.]]),
    'blender': np.array([[700.,     0.,     320.],
                         [0.,       700.,   240.],
                         [0.,       0.,     1.]]),
    'pascal': np.asarray([[-3000.0, 0.0,    0.0],
                         [0.0,      3000.0, 0.0],
                         [0.0,      0.0,    1.0]]),
    'ycb_K1': np.array([[1066.778, 0.        , 312.9869],
                        [0.      , 1067.487  , 241.3109],
                        [0.      , 0.        , 1.0]], np.float32),
    'ycb_K2': np.array([[1077.836, 0.        , 323.7872],
                        [0.      , 1078.189  , 279.6921],
                        [0.      , 0.        , 1.0]], np.float32)
}



def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
        A: Nxm numpy array of corresponding points, usually points on mdl
        B: Nxm numpy array of corresponding points, usually points on camera axis
    Returns:
    T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
    R: mxm rotation matrix
    t: mx1 translation vector
    '''

    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matirx
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    return  T


class PoseTransformer(object):
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {}
    class_type_to_number = {
        'ape': '001',
        'can': '004',
        'cat': '005',
        'driller': '006',
        'duck': '007',
        'eggbox': '008',
        'glue': '009',
        'holepuncher': '010'
    }
    blender_models={}

    def __init__(self, class_type):
        self.class_type = class_type
        lm_pth = 'datasets/linemod/LINEMOD'
        lm_occ_pth = 'datasets/linemod/OCCLUSION_LINEMOD'
        self.blender_model_path = os.path.join(lm_pth,'{}/{}.ply'.format(class_type, class_type))
        self.xyz_pattern = os.path.join(lm_occ_pth,'models/{}/{}.xyz')

    @staticmethod
    def load_ply_model(model_path):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        return np.stack([x, y, z], axis=-1)

    def get_blender_model(self):
        if self.class_type in self.blender_models:
            return self.blender_models[self.class_type]

        blender_model = self.load_ply_model(self.blender_model_path.format(self.class_type, self.class_type))
        self.blender_models[self.class_type] = blender_model

        return blender_model

    def get_translation_transform(self):
        if self.class_type in self.translation_transforms:
            return self.translation_transforms[self.class_type]

        model = self.get_blender_model()
        xyz = np.loadtxt(self.xyz_pattern.format(
            self.class_type.title(), self.class_type_to_number[self.class_type]))
        rotation = np.array([[0., 0., 1.],
                             [1., 0., 0.],
                             [0., 1., 0.]])
        xyz = np.dot(xyz, rotation.T)
        translation_transform = np.mean(xyz, axis=0) - np.mean(model, axis=0)
        self.translation_transforms[self.class_type] = translation_transform

        return translation_transform

    def occlusion_pose_to_blender_pose(self, pose):
        rot, tra = pose[:, :3], pose[:, 3]
        rotation = np.array([[0., 1., 0.],
                             [0., 0., 1.],
                             [1., 0., 0.]])
        rot = np.dot(rot, rotation)

        tra[1:] *= -1
        translation_transform = np.dot(rot, self.get_translation_transform())
        rot[1:] *= -1
        translation_transform[1:] *= -1
        tra += translation_transform
        pose = np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

        return pose


class Basic_Utils():

    def __init__(self, config):
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.config = config
        if config.dataset_name == "ycb":
            self.ycb_cls_lst = config.ycb_cls_lst
        self.ycb_cls_ptsxyz_dict = {}
        self.ycb_cls_ptsxyz_cuda_dict = {}
        self.ycb_cls_kps_dict = {}
        self.ycb_cls_ctr_dict = {}
        self.lm_cls_ptsxyz_dict = {}
        self.lm_cls_ptsxyz_cuda_dict = {}
        self.lm_cls_kps_dict = {}
        self.lm_cls_ctr_dict = {}

    def read_lines(self, p):
        with open(p, 'r') as f:
            lines = [
                line.strip() for line in f.readlines()
            ]
        return lines

    def sv_lines(self, p, line_lst):
        with open(p, 'w') as f:
            for line in line_lst:
                print(line, file=f)

    def cal_frustum_RT(self, ctr):
        # rotate through axis z to x-z plane
        sign = -1.0 if ctr[1] * ctr[0] < 0 else 1.0
        anglez = -1.0 * sign * np.arctan2(abs(ctr[1]), abs(ctr[0]))
        Rz = np.array([
            [np.cos(anglez),    -1.0*np.sin(anglez),   0],
            [np.sin(anglez),    np.cos(anglez),        0],
            [0,                 0,                     1]
        ])
        # rotate through axis y to axis z
        ctr = np.dot(ctr, Rz.T)
        sign = -1.0 if ctr[0] * ctr[2] < 0 else 1.0
        angley = -1.0 * sign * np.arctan2(abs(ctr[0]), abs(ctr[2]))
        Ry = np.array([
            [np.cos(angley),        0.0,    np.sin(angley)],
            [0.0,                   1.0,    0.0           ],
            [-1.0*np.sin(angley),   0.0,    np.cos(angley)]
        ])
        ctr = np.dot(ctr, Ry.T)
        R = np.dot(Ry, Rz)
        T = -1.0 * ctr
        RT = np.zeros((3, 4))
        RT[:3, :3] = R
        RT[:, 3] = T
        return RT, R, -1.0 * ctr

    def cal_frustum_RT_RAug(self, ctr):
        RT, R, T = self.cal_frustum_RT(ctr)
        if random.random() > 0.5:
            rand_ang = random.random() * 2.0 * np.pi
            Rz = np.array([
                [np.cos(rand_ang),    -1.0*np.sin(rand_ang),   0],
                [np.sin(rand_ang),    np.cos(rand_ang),        0],
                [0,                   0,                       1]
            ])
            R = np.dot(Rz, R)
            RT[:3, :3] = R
        return RT, R, T

    def translate(self, img, x, y):
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return shifted

    def rotate(self, img, angle, ctr=None, scale=1.0):
        (h, w) = img.shape[:2]
        if ctr is None:
            ctr = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(ctr, -1.0 * angle, scale)
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated

    def cal_degree_from_vec(self, v1, v2):
        cos = np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if abs(cos) > 1.0:
            cos = 1.0 * (-1.0 if cos < 0 else 1.0)
            print(cos, v1, v2)
        dg = np.arccos(cos) / np.pi * 180
        return dg

    def cal_directional_degree_from_vec(self, v1, v2):
        dg12 = self.cal_degree_from_vec(v1, v2)
        cross = v1[0] * v2[1] - v2[0] * v1[1]
        if cross < 0:
            dg12 = 360 - dg12

        return dg12

    def mean_shift(self, data, radius=5.0):
        clusters = []
        for i in range(len(data)):
            cluster_centroid = data[i]
            cluster_frequency = np.zeros(len(data))
            # Search points in circle
            while True:
                temp_data = []
                for j in range(len(data)):
                    v = data[j]
                    # Handle points in the circles
                    if np.linalg.norm(v - cluster_centroid) <= radius:
                        temp_data.append(v)
                        cluster_frequency[i] += 1
                # Update centroid
                old_centroid = cluster_centroid
                new_centroid = np.average(temp_data, axis=0)
                cluster_centroid = new_centroid
                # Find the mode
                if np.array_equal(new_centroid, old_centroid):
                    break
            # Combined 'same' clusters
            has_same_cluster = False
            for cluster in clusters:
                if np.linalg.norm(cluster['centroid'] - cluster_centroid) <= radius:
                    has_same_cluster = True
                    cluster['frequency'] = cluster['frequency'] + cluster_frequency
                    break
            if not has_same_cluster:
                clusters.append({
                    'centroid': cluster_centroid,
                    'frequency': cluster_frequency
                })

        print('clusters (', len(clusters), '): ', clusters)
        self.clustering(data, clusters)
        return clusters

    # Clustering data using frequency
    def clustering(self, data, clusters):
        t = []
        for cluster in clusters:
            cluster['data'] = []
            t.append(cluster['frequency'])
        t = np.array(t)
        # Clustering
        for i in range(len(data)):
            column_frequency = t[:, i]
            cluster_index = np.where(column_frequency == np.max(column_frequency))[0][0]
            clusters[cluster_index]['data'].append(data[i])

    def project_p3d(self, p3d, cam_scale, K=intrinsic_matrix['ycb_K1']):
        p3d = p3d * cam_scale
        p2d = np.dot(p3d, K.T)
        p2d_3 = p2d[:, 2]
        p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
        p2d[:, 2] = p2d_3
        p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
        return p2d

    def draw_p2ds(self, img, p2ds, r=1, color=(255, 0, 0)):
        h, w = img.shape[0], img.shape[1]
        for pt_2d in p2ds:
            pt_2d[0] = np.clip(pt_2d[0], 0, w)
            pt_2d[1] = np.clip(pt_2d[1], 0, h)
            img = cv2.circle(
                img, (pt_2d[0], pt_2d[1]), r, color, -1
            )
        return img

    def get_show_label_img(self, labels, mode=1):
        cls_ids = np.unique(labels)
        n_obj = np.max(cls_ids)
        if len(labels.shape) > 2:
            labels = labels[:, :, 0]
        h, w = labels.shape
        show_labels = np.zeros(
            (h, w, 3), dtype='uint8'
        )
        labels = labels.reshape(-1)
        show_labels = show_labels.reshape(-1, 3)
        for cls_id in cls_ids:
            if cls_id == 0:
                continue
            cls_color = np.array(
                self.get_label_color(cls_id, n_obj=n_obj, mode=mode)
            )
            show_labels[labels == cls_id, :] = cls_color
        show_labels = show_labels.reshape(h, w, 3)
        return show_labels

    def get_label_color(self, cls_id, n_obj=22, mode=0):
        if mode == 0:
            cls_color = [
                255, 255, 255,  # 0
                180, 105, 255,   # 194, 194, 0,    # 1 # 194, 194, 0
                0, 255, 0,      # 2
                0, 0, 255,      # 3
                0, 255, 255,    # 4
                255, 0, 255,    # 5
                180, 105, 255,  # 128, 128, 0,    # 6
                128, 0, 0,      # 7
                0, 128, 0,      # 8
                0, 165, 255,    # 0, 0, 128,      # 9
                128, 128, 0,    # 10
                0, 0, 255,      # 11
                255, 0, 0,      # 12
                0, 194, 0,      # 13
                0, 194, 0,      # 14
                255, 255, 0,    # 15 # 0, 194, 194
                64, 64, 0,      # 16
                64, 0, 64,      # 17
                185, 218, 255,  # 0, 0, 64,       # 18
                0, 0, 255,      # 19
                0, 64, 0,       # 20
                0, 0, 192       # 21
            ]
            cls_color = np.array(cls_color).reshape(-1, 3)
            color = cls_color[cls_id]
            bgr = (int(color[0]), int(color[1]), int(color[2]))
        else:
            mul_col = 255 * 255 * 255 // n_obj * cls_id
            r, g, b= mul_col // 255 // 255, (mul_col // 255) % 255, mul_col % 255
            bgr = (int(r), int(g) , int(b))
        return bgr

    def dpt_2_cld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        msk_dp = dpt > 1e-6
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
        if len(choose) < 1:
            return None, None

        dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_mskd = self.xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_mskd = self.ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

        pt2 = dpt_mskd / cam_scale
        cam_cx, cam_cy = K[0][2], K[1][2]
        cam_fx, cam_fy = K[0][0], K[1][1]
        pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
        cld = np.concatenate((pt0, pt1, pt2), axis=1)
        return cld, choose

    def get_normal(self, cld):
        import pcl
        cloud = pcl.PointCloud()
        cld = cld.astype(np.float32)
        cloud.from_array(cld)
        ne = cloud.make_NormalEstimation()
        kdtree = cloud.make_kdtree()
        ne.set_SearchMethod(kdtree)
        ne.set_KSearch(50)
        n = ne.compute()
        n = n.to_array()
        return n

    def get_normal_map(self, nrm, choose):
        nrm_map = np.zeros((480, 640, 3), dtype=np.uint8)
        nrm = nrm[:, :3]
        nrm[np.isnan(nrm)] = 0.0
        nrm[np.isinf(nrm)] = 0.0
        nrm_color = ((nrm + 1.0) * 127).astype(np.uint8)
        nrm_map = nrm_map.reshape(-1, 3)
        nrm_map[choose, :] = nrm_color
        nrm_map = nrm_map.reshape((480, 640, 3))
        return nrm_map

    def get_rgb_pts_map(self, pts, choose):
        pts_map = np.zeros((480, 640, 3), dtype=np.uint8)
        pts = pts[:, :3]
        pts[np.isnan(pts)] = 0.0
        pts[np.isinf(pts)] = 0.0
        pts_color = pts.astype(np.uint8)
        pts_map = pts_map.reshape(-1, 3)
        pts_map[choose, :] = pts_color
        pts_map = pts_map.reshape((480, 640, 3))
        return pts_map

    def fill_missing(
            self, dpt, cam_scale, scale_2_80m, fill_type='multiscale',
            extrapolate=False, show_process=False, blur_type='bilateral'
    ):
        dpt = dpt / cam_scale * scale_2_80m
        projected_depth = dpt.copy()
        if fill_type == 'fast':
            final_dpt = depth_map_utils.fill_in_fast(
                projected_depth, extrapolate=extrapolate, blur_type=blur_type,
                # max_depth=2.0
            )
        elif fill_type == 'multiscale':
            final_dpt, process_dict = depth_map_utils.fill_in_multiscale(
                projected_depth, extrapolate=extrapolate, blur_type=blur_type,
                show_process=show_process,
                max_depth=3.0
            )
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))
        dpt = final_dpt / scale_2_80m * cam_scale
        return dpt

    def rand_range(self, lo, hi):
        return random.random()*(hi-lo)+lo

    def get_ycb_ply_mdl(
        self, cls
    ):
        ply_pattern = os.path.join(
            self.config.ycb_root, '/models',
            '{}/textured.ply'
        )
        ply = PlyData.read(ply_pattern.format(cls, cls))
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        model = np.stack([x, y, z], axis=-1)
        return model

    def get_cls_name(self, cls, ds_type):
        if type(cls) is int:
            if ds_type == 'ycb':
                cls = self.ycb_cls_lst[cls - 1]
            else:
                cls = self.lm_cls_lst[cls - 1]
        return cls

    def ply_vtx(self, pth):
        f = open(pth)
        assert f.readline().strip() == "ply"
        f.readline()
        f.readline()
        N = int(f.readline().split()[-1])
        while f.readline().strip() != "end_header":
            continue
        pts = []
        for _ in range(N):
            pts.append(np.float32(f.readline().split()[:3]))
        return np.array(pts)

    def get_pointxyz(
        self, cls, ds_type='ycb'
    ):
        if ds_type == "ycb":
            cls = self.get_cls_name(cls, ds_type)
            if cls in self.ycb_cls_ptsxyz_dict.keys():
                return self.ycb_cls_ptsxyz_dict[cls]
            ptxyz_ptn = os.path.join(
                self.config.ycb_root, 'models',
                '{}/points.xyz'.format(cls),
            )
            pointxyz = np.loadtxt(ptxyz_ptn.format(cls), dtype=np.float32)
            self.ycb_cls_ptsxyz_dict[cls] = pointxyz
            return pointxyz
        else:
            ptxyz_pth = os.path.join(
                'datasets/linemod/Linemod_preprocessed/models',
                'obj_%02d.ply' % cls
            )
            pointxyz = self.ply_vtx(ptxyz_pth) / 1000.0
            dellist = [j for j in range(0, len(pointxyz))]
            dellist = random.sample(dellist, len(pointxyz) - 2000)
            pointxyz = np.delete(pointxyz, dellist, axis=0)
            self.lm_cls_ptsxyz_dict[cls] = pointxyz
            return pointxyz

    def get_pointxyz_cuda(
        self, cls, ds_type='ycb'
    ):
        if ds_type == "ycb":
            if cls in self.ycb_cls_ptsxyz_cuda_dict.keys():
                return self.ycb_cls_ptsxyz_cuda_dict[cls].clone()
            ptsxyz = self.get_pointxyz(cls, ds_type)
            ptsxyz_cu = torch.from_numpy(ptsxyz.astype(np.float32)).cuda()
            self.ycb_cls_ptsxyz_cuda_dict[cls] = ptsxyz_cu
            return ptsxyz_cu.clone()
        else:
            if cls in self.lm_cls_ptsxyz_cuda_dict.keys():
                return self.lm_cls_ptsxyz_cuda_dict[cls].clone()
            ptsxyz = self.get_pointxyz(cls, ds_type)
            ptsxyz_cu = torch.from_numpy(ptsxyz.astype(np.float32)).cuda()
            self.lm_cls_ptsxyz_cuda_dict[cls] = ptsxyz_cu
            return ptsxyz_cu.clone()

    def get_kps(
        self, cls, kp_type='farthest', ds_type='ycb', kp_pth=None
    ):
        if kp_pth:
            return np.loadtxt(kp_pth)
        if type(cls) is int:
            if ds_type == 'ycb':
                cls = self.ycb_cls_lst[cls - 1]
            else:
                cls = self.config.lm_id2obj_dict[cls]
        if ds_type == "ycb":
            if cls in self.ycb_cls_kps_dict.keys():
                return self.ycb_cls_kps_dict[cls].copy()
            kps_pattern = os.path.join(
                self.config.ycb_kps_dir, '{}/{}.txt'.format(cls, kp_type)
            )
            kps = np.loadtxt(kps_pattern.format(cls), dtype=np.float32)
            self.ycb_cls_kps_dict[cls] = kps
        else:
            if cls in self.lm_cls_kps_dict.keys():
                return self.lm_cls_kps_dict[cls].copy()
            kps_pattern = os.path.join(
                self.config.lm_kps_dir, "{}/{}.txt".format(cls, kp_type)
            )
            kps = np.loadtxt(kps_pattern.format(cls), dtype=np.float32)
            self.lm_cls_kps_dict[cls] = kps
        return kps.copy()

    def get_ctr(self, cls, ds_type='ycb', ctr_pth=None):
        if ctr_pth:
            return np.loadtxt(ctr_pth)
        if type(cls) is int:
            if ds_type == 'ycb':
                cls = self.ycb_cls_lst[cls - 1]
            else:
                cls = self.config.lm_id2obj_dict[cls]
        if ds_type == "ycb":
            if cls in self.ycb_cls_ctr_dict.keys():
                return self.ycb_cls_ctr_dict[cls].copy()
            cor_pattern = os.path.join(
                self.config.ycb_kps_dir, '{}/corners.txt'.format(cls),
            )
            cors = np.loadtxt(cor_pattern.format(cls), dtype=np.float32)
            ctr = cors.mean(0)
            self.ycb_cls_ctr_dict[cls] = ctr
        else:
            if cls in self.lm_cls_ctr_dict.keys():
                return self.lm_cls_ctr_dict[cls].copy()
            cor_pattern = os.path.join(
                self.config.lm_kps_dir, '{}/corners.txt'.format(cls),
            )
            cors = np.loadtxt(cor_pattern.format(cls), dtype=np.float32)
            ctr = cors.mean(0)
            self.lm_cls_ctr_dict[cls] = ctr
        return ctr.copy()

    def cal_auc(self, add_dis, max_dis=0.1):
        D = np.array(add_dis)
        D[np.where(D > max_dis)] = np.inf;
        D = np.sort(D)
        n = len(add_dis)
        acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
        aps = VOCap(D, acc)
        return aps * 100

    def cal_pose_from_kp(
            self, cls_id, pred_kps, ds_type='ycb', kp_type='farthest'
    ):
        if ds_type == 'ycb':
            cls_nm = self.ycb_cls_lst[cls_id-1]
        else:
            cls_nm = self.lm_cls_lst[cls_id-1]
        kp_on_mesh = self.get_kps(cls_nm, kp_type=kp_type)
        RT = best_fit_transform(kp_on_mesh, pred_kps)
        return RT

    def cal_add_cuda(
        self, pred_RT, gt_RT, p3ds
    ):
        pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        gt_p3ds = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        dis = torch.norm(pred_p3ds - gt_p3ds, dim=1)
        return torch.mean(dis)

    def cal_adds_cuda(
        self, pred_RT, gt_RT, p3ds
    ):
        N, _ = p3ds.size()
        pd = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
        pd = pd.view(1, N, 3).repeat(N, 1, 1)
        gt = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
        gt = gt.view(N, 1, 3).repeat(1, N, 1)
        dis = torch.norm(pd - gt, dim=2)
        mdis = torch.min(dis, dim=1)[0]
        return torch.mean(mdis)

    def best_fit_transform_torch(self, A, B):
        '''
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
            A: Nxm numpy array of corresponding points, usually points on mdl
            B: Nxm numpy array of corresponding points, usually points on camera axis
        Returns:
        T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R: mxm rotation matrix
        t: mx1 translation vector
        '''
        assert A.size() == B.size()
        # get number of dimensions
        m = A.size()[1]
        # translate points to their centroids
        centroid_A = torch.mean(A, dim=0)
        centroid_B = torch.mean(B, dim=0)
        AA = A - centroid_A
        BB = B - centroid_B
        # rotation matirx
        H = torch.mm(AA.transpose(1, 0), BB)
        U, S, Vt = torch.svd(H)
        R = torch.mm(Vt.transpose(1, 0), U.transpose(1, 0))
        # special reflection case
        if torch.det(R) < 0:
            Vt[m-1, :] *= -1
            R = torch.mm(Vt.transpose(1, 0), U.transpose(1, 0))
        # translation
        t = centroid_B - torch.mm(R, centroid_A.view(3, 1))[:, 0]
        T = torch.zeros(3, 4).cuda()
        T[:, :3] = R
        T[:, 3] = t
        return  T

    def best_fit_transform(self, A, B):
        return best_fit_transform(A, B)


if __name__ == "__main__":

    pass
# vim: ts=4 sw=4 sts=4 expandtab
