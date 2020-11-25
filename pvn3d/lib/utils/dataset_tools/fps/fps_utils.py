import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from _ext import lib, ffi
import numpy as np


def farthest_point_sampling(pts, sn, init_center=False):
    pn, _ = pts.shape
    assert(pts.shape[1] == 3)

    pts = np.ascontiguousarray(pts, np.float32)
    idxs = np.ascontiguousarray(np.zeros([sn], np.int32))

    pts_ptr = ffi.cast('float*', pts.ctypes.data)
    idxs_ptr = ffi.cast('int*', idxs.ctypes.data)

    if init_center:
        lib.farthest_point_sampling_init_center(pts_ptr, idxs_ptr, pn, sn)
    else:
        lib.farthest_point_sampling(pts_ptr, idxs_ptr, pn, sn)

    return pts[idxs]
