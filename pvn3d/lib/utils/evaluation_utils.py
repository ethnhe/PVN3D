import time

import scipy
import sys
sys.path.append('.')

from lib.utils.config import cfg
from lib.utils.data_utils import LineModModelDB, Projector, YCBModelDB
from plyfile import PlyData
import numpy as np
import cv2
import os
import uuid
import itertools
from queue import Queue

# from lib.utils.extend_utils.extend_utils import uncertainty_pnp, find_nearest_point_idx, uncertainty_pnp_v2

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
    # print("H: \n", H)
    U, S, Vt = np.linalg.svd(H)
    # print(
    #     "U: \n", U, "\n",
    #     "S: \n", S, "\n",
    #     "V: \n", Vt, "\n",
    # )

    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    # T = np.identity(m+1)
    # T[:m, :m] = R
    # T[:m, m] = t
    T = np.zeros((3, 4))
    T[:, :3] = R
    T[:, 3] = t
    # R, _ = cv2.Rodrigues(R)

    # print(R.shape, t.shape)
    return  T# np.concatenate([R, t[:,None]], axis=-1)# T, R, t

class VotingType:
    BB8=0
    BB8C=1
    BB8S=2
    VanPts=3
    Farthest=5
    Farthest4=6
    Farthest12=7
    Farthest16=8
    Farthest20=9

    @staticmethod
    def get_data_pts_2d(vote_type,data):
        if vote_type==VotingType.BB8:
            cor = data['corners'].copy()  # note the copy here!!!
            hcoords=np.concatenate([cor,np.ones([8,1],np.float32)],1) # [8,3]
        elif vote_type==VotingType.BB8C:
            cor = data['corners'].copy()
            cen = data['center'].copy()
            hcoords = np.concatenate([cor,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([9,1],np.float32)],1)
        elif vote_type==VotingType.BB8S:
            cor = data['small_bbox'].copy()
            cen = data['center'].copy()
            hcoords = np.concatenate([cor,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([9,1],np.float32)],1)
        elif vote_type==VotingType.VanPts:
            cen = data['center'].copy()
            van = data['van_pts'].copy()
            hcoords = np.concatenate([cen,np.ones([1,1],np.float32)],1)
            hcoords = np.concatenate([van,hcoords],0)
        elif vote_type==VotingType.Farthest:
            cen = data['center'].copy()
            far = data['farthest'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.Farthest4:
            cen = data['center'].copy()
            far = data['farthest4'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.Farthest12:
            cen = data['center'].copy()
            far = data['farthest12'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.Farthest16:
            cen = data['center'].copy()
            far = data['farthest16'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)
        elif vote_type==VotingType.Farthest20:
            cen = data['center'].copy()
            far = data['farthest20'].copy()
            hcoords = np.concatenate([far,cen],0)
            hcoords = np.concatenate([hcoords,np.ones([hcoords.shape[0],1],np.float32)],1)

        return hcoords

    @staticmethod
    def get_pts_3d(vote_type,class_type):
        linemod_db=LineModModelDB()
        if vote_type==VotingType.BB8C:
            points_3d = linemod_db.get_corners_3d(class_type)
            points_3d = np.concatenate([points_3d,linemod_db.get_centers_3d(class_type)[None,:]],0)
        elif vote_type==VotingType.BB8S:
            points_3d = linemod_db.get_small_bbox(class_type)
            points_3d = np.concatenate([points_3d,linemod_db.get_centers_3d(class_type)[None,:]],0)
        elif vote_type==VotingType.Farthest:
            points_3d = linemod_db.get_farthest_3d(class_type)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.Farthest4:
            points_3d = linemod_db.get_farthest_3d(class_type,4)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.Farthest12:
            points_3d = linemod_db.get_farthest_3d(class_type,12)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.Farthest16:
            points_3d = linemod_db.get_farthest_3d(class_type,16)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        elif vote_type==VotingType.Farthest20:
            points_3d = linemod_db.get_farthest_3d(class_type,20)
            points_3d = np.concatenate([points_3d, linemod_db.get_centers_3d(class_type)[None, :]], 0)
        else: # BB8
            points_3d = linemod_db.get_corners_3d(class_type)

        return points_3d

    @staticmethod
    def ycb_get_pts_3d(vote_type,class_type, use_ctr=False):
        ycb_db=YCBModelDB()
        if vote_type==VotingType.BB8C:
            points_3d = ycb_db.get_corners_3d(class_type)
            if use_ctr:
                points_3d = np.concatenate(
                    [points_3d,ycb_db.get_centers_3d(class_type)[None,:]], 0
                )
        elif vote_type==VotingType.BB8S:
            points_3d = ycb_db.get_small_bbox(class_type)
            if use_ctr:
                points_3d = np.concatenate(
                    [points_3d,ycb_db.get_centers_3d(class_type)[None,:]], 0
                )
        elif vote_type==VotingType.Farthest:
            points_3d = ycb_db.get_farthest_3d(class_type)
            if use_ctr:
                points_3d = np.concatenate(
                    [points_3d, ycb_db.get_centers_3d(class_type)[None, :]], 0
                )
        elif vote_type==VotingType.Farthest4:
            points_3d = ycb_db.get_farthest_3d(class_type, 4)
            if use_ctr:
                points_3d = np.concatenate(
                    [points_3d, ycb_db.get_centers_3d(class_type)[None, :]], 0
                )
        elif vote_type==VotingType.Farthest12:
            points_3d = ycb_db.get_farthest_3d(class_type, 12)
            if use_ctr:
                points_3d = np.concatenate(
                    [points_3d, ycb_db.get_centers_3d(class_type)[None, :]], 0
                )
        elif vote_type==VotingType.Farthest16:
            points_3d = ycb_db.get_farthest_3d(class_type, 16)
            if use_ctr:
                points_3d = np.concatenate(
                    [points_3d, ycb_db.get_centers_3d(class_type)[None, :]], 0
                )
        elif vote_type==VotingType.Farthest20:
            points_3d = ycb_db.get_farthest_3d(class_type, 20)
            if use_ctr:
                points_3d = np.concatenate(
                    [points_3d, ycb_db.get_centers_3d(class_type)[None, :]], 0
                )
        else: # BB8
            points_3d = ycb_db.get_corners_3d(class_type)

        return points_3d

def pnp(points_3d, points_2d, camera_matrix,method=cv2.SOLVEPNP_ITERATIVE):
    try:
        dist_coeffs = pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')

    assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
    if method==cv2.SOLVEPNP_EPNP:
        points_3d=np.expand_dims(points_3d, 0)
        points_2d=np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    # _, R_exp, t = cv2.solvePnP(points_3d,
    #                            points_2d,
    #                            camera_matrix,
    #                            dist_coeffs,
    #                            flags=method)
    #                           # , None, None, False, cv2.SOLVEPNP_UPNP)

    _, R_exp, t, _ = cv2.solvePnPRansac(points_3d,
                               points_2d,
                               camera_matrix,
                               dist_coeffs,
                               )

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t], axis=-1)

def find_nearest_point_distance(pts1,pts2):
    '''

    :param pts1:  pn1,2 or 3
    :param pts2:  pn2,2 or 3
    :return:
    '''
    idxs=find_nearest_point_idx(pts1,pts2)
    return np.linalg.norm(pts1[idxs]-pts2,2,1)

class Evaluator(object):
    def __init__(self):
        self.linemod_db = LineModModelDB()
        self.ycb_db = YCBModelDB()
        self.projector=Projector()
        self.projection_2d_recorder = Queue() # []
        self.add_recorder = Queue() # []
        self.cm_degree_5_recorder = Queue() # []
        self.proj_mean_diffs= Queue() # []
        self.add_dists= Queue() # []
        self.uncertainty_pnp_cost= Queue() # []
        self.add_dis = Queue() # []

    def projection_2d(self, pose_pred, pose_targets, model, K, threshold=5):
        model_2d_pred = self.projector.project_K(model, pose_pred, K)
        model_2d_targets = self.projector.project_K(model, pose_targets, K)
        proj_mean_diff=np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))

        self.proj_mean_diffs.put(proj_mean_diff)
        self.projection_2d_recorder.put(proj_mean_diff < threshold)

    def projection_2d_sym(self, pose_pred, pose_targets, model, K, threshold=5):
        model_2d_pred = self.projector.project_K(model, pose_pred, K)
        model_2d_targets = self.projector.project_K(model, pose_targets, K)
        proj_mean_diff=np.mean(find_nearest_point_distance(model_2d_pred,model_2d_targets))

        self.proj_mean_diffs.put(proj_mean_diff)
        self.projection_2d_recorder.put(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, model, diameter, percentage=0.1):
        """ ADD metric
        1. compute the average of the 3d distances between the transformed vertices
        2. pose_pred is considered correct if the distance is less than 10% of the object's diameter
        """
        diameter = diameter * percentage
        model_pred = np.dot(model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(model, pose_targets[:, :3].T) + pose_targets[:, 3]

        mean_dist=np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        self.add_dis.put(mean_dist)
        self.add_recorder.put(mean_dist < diameter)
        self.add_dists.put(mean_dist)
        return mean_dist

    def add_metric_sym(self, pose_pred, pose_targets, model, diameter, percentage=0.1):
        """ ADD metric
        1. compute the average of the 3d distances between the transformed vertices
        2. pose_pred is considered correct if the distance is less than 10% of the object's diameter
        """
        diameter = diameter * percentage
        model_pred = np.dot(model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(model, pose_targets[:, :3].T) + pose_targets[:, 3]

        mean_dist=np.mean(find_nearest_point_distance(model_pred,model_targets))
        self.add_dis.put(mean_dist)
        self.add_recorder.put(mean_dist < diameter)
        self.add_dists.put(mean_dist)
        return mean_dist

    def cm_degree_5_metric(self, pose_pred, pose_targets):
        """ 5 cm 5 degree metric
        1. pose_pred is considered correct if the translation and rotation errors are below 5 cm and 5 degree respectively
        """
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cm_degree_5_recorder.put(translation_distance < 5 and angular_distance < 5)

    def evaluate_3dkp_adds(
            self, dt_p3d, pose_targets, class_type, intri_type='blender',
            vote_type=VotingType.Farthest, intri_matrix=None, use_ctr=False
    ):
        mdl_p3d = VotingType.get_pts_3d(vote_type, class_type)

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        if use_ctr:
            pose_pred = best_fit_transform(mdl_p3d, dt_p3d)
        else:
            pose_pred = best_fit_transform(mdl_p3d[:8], dt_p3d)
        # pose_pred = pnp(points_3d, points_2d, K)
        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)

        self.add_metric_sym(pose_pred, pose_targets, model, diameter)

        self.projection_2d(pose_pred, pose_targets, model, K)
        self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred

    def evaluate_3dkp(
            self, dt_p3d, pose_targets, class_type, intri_type='blender',
            vote_type=VotingType.Farthest, intri_matrix=None, use_ctr=False
    ):
        mdl_p3d = VotingType.get_pts_3d(vote_type, class_type)

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        if use_ctr:
            pose_pred = best_fit_transform(mdl_p3d, dt_p3d)
        else:
            pose_pred = best_fit_transform(mdl_p3d[:8], dt_p3d)
        # pose_pred = pnp(points_3d, points_2d, K)
        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)

        if class_type in ['eggbox','glue']:
            self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        else:
            self.add_metric(pose_pred, pose_targets, model, diameter)

        self.projection_2d(pose_pred, pose_targets, model, K)
        self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred

    def ycb_evaluate_RT(
            self, pose_pred, pose_targets, class_type, intri_type='ycb_K1',
            vote_type=VotingType.Farthest, intri_matrix=None
    ):
        mdl_p3d = VotingType.ycb_get_pts_3d(vote_type, class_type, use_ctr=True)

        model = self.ycb_db.get_pointxyz(class_type)
        diameter = 0.02 * 10.0# self.ycb_db.get_diameter(class_type)

        mean_dis = self.add_metric_sym(pose_pred, pose_targets, model, diameter)

        return pose_pred, mean_dis, np.dot(mdl_p3d, pose_targets[:, :3].T) + pose_targets[:, 3]

    def ycb_evaluate_add_RT(
            self, pose_pred, pose_targets, class_type, intri_type='ycb_K1',
            vote_type=VotingType.Farthest, intri_matrix=None
    ):
        mdl_p3d = VotingType.ycb_get_pts_3d(vote_type, class_type, use_ctr=True)

        model = self.ycb_db.get_pointxyz(class_type)
        diameter = 0.02 * 10.0# self.ycb_db.get_diameter(class_type)
        mean_dis = self.add_metric(pose_pred, pose_targets, model, diameter)

        return pose_pred, mean_dis, np.dot(mdl_p3d, pose_targets[:, :3].T) + pose_targets[:, 3]

    def ycb_evaluate_3dkp_ctr(
            self, dt_p3d, pose_targets, class_type, intri_type='ycb_K1',
            vote_type=VotingType.Farthest, intri_matrix=None, use_ctr=True
    ):
        mdl_p3d = VotingType.ycb_get_pts_3d(
            vote_type, class_type, use_ctr=use_ctr
        )

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        pose_pred = best_fit_transform(mdl_p3d, dt_p3d)
        # pose_pred = pnp(points_3d, points_2d, K)
        model = self.ycb_db.get_pointxyz(class_type)
        # model = self.ycb_db.get_ply_model(class_type)
        diameter = 0.02 * 10.0# self.ycb_db.get_diameter(class_type)
        symetry_ycb_cls = [
            '024_bowl','036_wood_block', '051_large_clamp',
            '052_extra_large_clamp', '061_foam_brick'
        ]

        mean_dis = self.add_metric_sym(pose_pred, pose_targets, model, diameter)

        # self.projection_2d(pose_pred, pose_targets, model, K)
        # self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred, mean_dis, np.dot(mdl_p3d, pose_targets[:, :3].T) + pose_targets[:, 3]

    def ycb_evaluate_add_3dkp_ctr(
            self, dt_p3d, pose_targets, class_type, intri_type='ycb_K1',
            vote_type=VotingType.Farthest, intri_matrix=None, use_ctr=True
    ):
        mdl_p3d = VotingType.ycb_get_pts_3d(
            vote_type, class_type, use_ctr=use_ctr
        )

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        pose_pred = best_fit_transform(mdl_p3d, dt_p3d)
        # pose_pred = pnp(points_3d, points_2d, K)
        model = self.ycb_db.get_pointxyz(class_type)
        # model = self.ycb_db.get_ply_model(class_type)
        diameter = 0.02 * 10.0# self.ycb_db.get_diameter(class_type)

        mean_dis = self.add_metric(pose_pred, pose_targets, model, diameter)
        # if class_type in symetry_ycb_cls:
        #     self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        # else:
        #     self.add_metric(pose_pred, pose_targets, model, diameter)

        # self.projection_2d(pose_pred, pose_targets, model, K)
        # self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred, mean_dis, np.dot(mdl_p3d, pose_targets[:, :3].T) + pose_targets[:, 3]

    def ycb_evaluate_2dkp_ctr(
            self, dt_p2d, pose_targets, class_type, intri_type='ycb_K1',
            vote_type=VotingType.Farthest, intri_matrix=None, use_ctr=True
    ):
        mdl_p3d = VotingType.ycb_get_pts_3d(
            vote_type, class_type, use_ctr=use_ctr
        )

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        # pose_pred = best_fit_transform(mdl_p3d, dt_p3d)
        pose_pred = pnp(mdl_p3d, dt_p2d, K)
        model = self.ycb_db.get_pointxyz(class_type)
        diameter = 0.02 * 10.0# self.ycb_db.get_diameter(class_type)
        symetry_ycb_cls = [
            '024_bowl','036_wood_block', '051_large_clamp',
            '052_extra_large_clamp', '061_foam_brick'
        ]

        mean_dis = self.add_metric_sym(pose_pred, pose_targets, model, diameter)

        # self.projection_2d(pose_pred, pose_targets, model, K)
        # self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred, mean_dis, np.dot(mdl_p3d, pose_targets[:, :3].T) + pose_targets[:, 3]

    def ycb_evaluate_add_2dkp_ctr(
            self, dt_p2d, pose_targets, class_type, intri_type='ycb_K1',
            vote_type=VotingType.Farthest, intri_matrix=None, use_ctr=True
    ):
        mdl_p3d = VotingType.ycb_get_pts_3d(vote_type, class_type, use_ctr=True)

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        # pose_pred = best_fit_transform(mdl_p3d, dt_p3d)
        pose_pred = pnp(mdl_p3d, dt_p2d, K)
        model = self.ycb_db.get_pointxyz(class_type)
        # model = self.ycb_db.get_ply_model(class_type)
        diameter = 0.02 * 10.0# self.ycb_db.get_diameter(class_type)

        mean_dis = self.add_metric(pose_pred, pose_targets, model, diameter)
        # self.projection_2d(pose_pred, pose_targets, model, K)
        # self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred, mean_dis, np.dot(mdl_p3d, pose_targets[:, :3].T) + pose_targets[:, 3]

    def ycb_evaluate_3dkp(
            self, dt_p3d, pose_targets, class_type, intri_type='ycb_K1',
            vote_type=VotingType.Farthest, intri_matrix=None
    ):
        mdl_p3d = VotingType.ycb_get_pts_3d(vote_type, class_type)

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        pose_pred = best_fit_transform(mdl_p3d[:8], dt_p3d)
        # pose_pred = pnp(points_3d, points_2d, K)
        model = self.ycb_db.get_pointxyz(class_type)
        # model = self.ycb_db.get_ply_model(class_type)
        diameter = 0.02 * 10.0# self.ycb_db.get_diameter(class_type)
        symetry_ycb_cls = [
            '024_bowl','036_wood_block', '051_large_clamp',
            '052_extra_large_clamp', '061_foam_brick'
        ]

        mean_dis = self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        # if class_type in symetry_ycb_cls:
        #     self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        # else:
        #     self.add_metric(pose_pred, pose_targets, model, diameter)

        # self.projection_2d(pose_pred, pose_targets, model, K)
        # self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred, mean_dis, np.dot(mdl_p3d, pose_targets[:, :3].T) + pose_targets[:, 3]

    def ycb_evaluate_3dkp_add(
            self, dt_p3d, pose_targets, class_type, intri_type='ycb_K1',
            vote_type=VotingType.Farthest, intri_matrix=None
    ):
        mdl_p3d = VotingType.ycb_get_pts_3d(vote_type, class_type)

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        pose_pred = best_fit_transform(mdl_p3d[:8], dt_p3d)
        # pose_pred = pnp(points_3d, points_2d, K)
        model = self.ycb_db.get_pointxyz(class_type)
        # model = self.ycb_db.get_ply_model(class_type)
        diameter = 0.02 * 10.0# self.ycb_db.get_diameter(class_type)
        symetry_ycb_cls = [
            '024_bowl','036_wood_block', '051_large_clamp',
            '052_extra_large_clamp', '061_foam_brick'
        ]

        mean_dis = self.add_metric(pose_pred, pose_targets, model, diameter)
        # if class_type in symetry_ycb_cls:
        #     self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        # else:
        #     self.add_metric(pose_pred, pose_targets, model, diameter)

        # self.projection_2d(pose_pred, pose_targets, model, K)
        # self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred, mean_dis, np.dot(mdl_p3d, pose_targets[:, :3].T) + pose_targets[:, 3]

    def ycb_evaluate_3dkp_3pt(
            self, dt_p3d, good_idx, pose_targets, class_type,
            intri_type='ycb_K1', vote_type=VotingType.Farthest, intri_matrix=None
    ):
        mdl_p3d = VotingType.ycb_get_pts_3d(vote_type, class_type)
        mdl_p3d = mdl_p3d[good_idx]

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        pose_pred = best_fit_transform(mdl_p3d, dt_p3d)
        # pose_pred = pnp(points_3d, points_2d, K)
        model = self.ycb_db.get_ply_model(class_type)
        diameter = 0.02 * 10.0# self.ycb_db.get_diameter(class_type)
        symetry_ycb_cls = [
            '024_bowl','036_wood_block', '051_large_clamp',
            '052_extra_large_clamp', '061_foam_brick'
        ]

        mean_dis = self.add_metric_sym_psl(pose_pred_lst, pose_targets, model, diameter)
        # if class_type in symetry_ycb_cls:
        #     self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        # else:
        #     self.add_metric(pose_pred, pose_targets, model, diameter)

        self.projection_2d(pose_pred, pose_targets, model, K)
        self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred, mean_dis, np.dot(mdl_p3d, pose_targets[:, :3].T) + pose_targets[:, 3]

    def ycb_evaluate_3dkp_cmbn(
            self, dt_p3d, pose_targets, class_type, intri_type='ycb_K1',
            vote_type=VotingType.Farthest, intri_matrix=None
    ):
        mdl_p3d = VotingType.ycb_get_pts_3d(vote_type, class_type)

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        cmbn_lst = list(
            itertools.combinations([i for i in range(dt_p3d.shape[0])], 3)
        )

        pose_pred_lst = []
        for i in range(len(cmbn_lst)):
            pose_pred = best_fit_transform(mdl_p3d[cmbn_lst[i]], dt_p3d[cmbn_lst[i]])
            pose_pred_lst.append(pose_pred)
        # pose_pred = pnp(points_3d, points_2d, K)
        model = self.ycb_db.get_ply_model(class_type)
        diameter = 0.02 * 10.0# self.ycb_db.get_diameter(class_type)
        symetry_ycb_cls = [
            '024_bowl','036_wood_block', '051_large_clamp',
            '052_extra_large_clamp', '061_foam_brick'
        ]

        self.add_metric_sym_psl(pose_pred_lst, pose_targets, model, diameter)
        # if class_type in symetry_ycb_cls:
        #     self.add_metric_sym_psl(pose_pred_lst, pose_targets, model, diameter)
        # else:
        #     self.add_metric_psl(pose_pred_lst, pose_targets, model, diameter)

        self.projection_2d(pose_pred, pose_targets, model, K)
        self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred

    def evaluate(self, points_2d, pose_targets, class_type, intri_type='blender', vote_type=VotingType.BB8, intri_matrix=None):
        points_3d = VotingType.get_pts_3d(vote_type, class_type)

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        pose_pred = best_fit_transform(mdl_p3d, dt_p3d)
        # pose_pred = pnp(points_3d, points_2d, K)
        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)

        if class_type in ['eggbox','glue']:
            self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        else:
            self.add_metric(pose_pred, pose_targets, model, diameter)

        self.projection_2d(pose_pred, pose_targets, model, K)
        self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred

    def evaluate_uncertainty(self, mean_pts2d, covar, pose_targets, class_type,
                             intri_type='blender', vote_type=VotingType.BB8,intri_matrix=None):
        points_3d=VotingType.get_pts_3d(vote_type,class_type)

        begin=time.time()
        # full
        cov_invs = []
        for vi in range(covar.shape[0]):
            if covar[vi,0,0]<1e-6 or np.sum(np.isnan(covar)[vi])>0:
                cov_invs.append(np.zeros([2,2]).astype(np.float32))
                continue

            cov_inv = np.linalg.inv(scipy.linalg.sqrtm(covar[vi]))
            cov_invs.append(cov_inv)
        cov_invs = np.asarray(cov_invs)  # pn,2,2
        weights = cov_invs.reshape([-1, 4])
        weights = weights[:, (0, 1, 3)]

        if intri_type=='use_intrinsic' and intri_matrix is not None:
            K=intri_matrix
        else:
            K=self.projector.intrinsic_matrix[intri_type]

        pose_pred = uncertainty_pnp(mean_pts2d, weights, points_3d, K)
        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)
        self.uncertainty_pnp_cost.put(time.time()-begin)

        if class_type in ['eggbox','glue']:
            self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        else:
            self.add_metric(pose_pred, pose_targets, model, diameter)

        self.projection_2d(pose_pred, pose_targets, model, K)
        self.cm_degree_5_metric(pose_pred, pose_targets)

        return pose_pred

    def evaluate_uncertainty_v2(self, mean_pts2d, covar, pose_targets, class_type,
                             intri_type='blender', vote_type=VotingType.BB8):
        points_3d = VotingType.get_pts_3d(vote_type, class_type)

        pose_pred = uncertainty_pnp_v2(mean_pts2d, covar, points_3d, self.projector.intrinsic_matrix[intri_type])
        model = self.linemod_db.get_ply_model(class_type)
        diameter = self.linemod_db.get_diameter(class_type)

        if class_type in ['eggbox','glue']:
            self.projection_2d_sym(pose_pred, pose_targets, model, self.projector.intrinsic_matrix[intri_type])
            self.add_metric_sym(pose_pred, pose_targets, model, diameter)
        else:
            self.projection_2d(pose_pred, pose_targets, model, self.projector.intrinsic_matrix[intri_type])
            self.add_metric(pose_pred, pose_targets, model, diameter)
        self.cm_degree_5_metric(pose_pred, pose_targets)

    def cal_auc(self, add_dis):
        max_dis = 0.1
        D = np.array(add_dis)
        D[np.where(D > max_dis)] = np.inf;
        D = np.sort(D)
        n = len(add_dis)
        acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
        aps = self.VOCap(D, acc)
        return aps * 100

    def VOCap(self, rec, prec):
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

    def average_precision(self,verbose=True, n_none=0):
        self.proj_mean_diffs = list(self.proj_mean_diffs.queue)
        np.save('tmp.npy',np.asarray(self.proj_mean_diffs))
        print("n_none: ", n_none)
        for i in range(n_none):
            self.projection_2d_recorder.put(0)
            self.add_recorder.put(0)
            self.add_dis.put(np.inf)
            self.cm_degree_5_recorder.put(0)
        self.projection_2d_recorder = list(self.projection_2d_recorder.queue)
        self.add_recorder = list(self.add_recorder.queue)
        self.add_dis = list(self.add_dis.queue)
        self.cm_degree_5_recorder = list(self.cm_degree_5_recorder.queue)

        if len(self.add_dis) > 2:
            auc = self.cal_auc(self.add_dis)
        else:
            auc = 0
        if verbose:
            print('2d projections metric: {}'.format(np.mean(self.projection_2d_recorder)))
            print('ADD metric: {}'.format(np.mean(self.add_recorder)))
            print('5 cm 5 degree metric: {}'.format(np.mean(self.cm_degree_5_recorder)))
            print('AUC: {}'.format(auc))

        return np.mean(self.projection_2d_recorder),np.mean(self.add_recorder),np.mean(self.cm_degree_5_recorder), auc

