import numpy as np 
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy import dot, sqrt
from numpy.linalg import norm, inv
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.linalg import rq
from src.sfm_pipeline.axis_angle import (rotation_mat_2_angle_axis, \
                                rotate_points_axis_angle)

import time



def unpack_func_result(result, ncameras, N):
    """[summary]

    Args:
        result ([type]): [description]
        ncameras ([type]): [description]
        N ([type]): [description]

    Returns:
        [type]: [description]
    """    

    # print('result.shape: ', result.shape)
    mot_slice = ncameras*3*2
    motion = np.reshape(result[0:mot_slice], (ncameras, 3, 2),order='F')
    structure = np.reshape(result[mot_slice:], (N, 3),order='F' )

    return motion, structure

def pack_func_args(motion, structure):
    """[summary]

    Args:
        motion ([type]): [description]
        structure ([type]): [description]

    Returns:
        [type]: [description]
    """    

    motion_vec = motion.flatten(order='F')
    structure_vec = structure.flatten(order='F')

    return np.concatenate((motion_vec, structure_vec))

def unpack_graph(graph):
    """[summary]

    Args:
        graph ([type]): [description]

    Returns:
        [type]: [description]
    """    
    
    ncameras = len(graph.frame_idxs)
    # R-> axis_angle (3x1), t (3x1)
    motion_aa = np.zeros((ncameras, 3,2))

    structure = graph.structure
    N = graph.obs_val.shape[1]

    for i in range(ncameras):
        Ri = graph.motion[i,:,:-1]
        ti = graph.motion[i,:,-1]
        motion_aa[i, :, 0] = rotation_mat_2_angle_axis(Ri)
        motion_aa[i, :, 1] = ti

    f = graph.f

    return motion_aa, structure, f, ncameras, N


def reprojection_residuals(obs_idx, obs_val, motion, structure, f, px, py):
    """[summary]

    Args:
        obs_idx ([type]): [description]
        obs_val ([type]): [description]
        motion ([type]): [description]
        structure ([type]): [description]
        f ([type]): [description]
        px ([type]): [description]
        py ([type]): [description]

    Returns:
        [type]: [description]
    """    
    ncameras = motion.shape[0]
    # residuals = np.zeros((ncameras, obs_val.shape[1]))
    residuals = np.zeros((0,2))
    for i in range(ncameras):
        axis_angle = motion[i, :, 0]
        translation = motion[i, :, 1]
        rot_points = rotate_points_axis_angle(axis_angle, structure.T)

        tr_x = rot_points[0,:] + translation[0]
        tr_y = rot_points[1,:] + translation[1]
        tr_z = rot_points[2,:] + translation[2]

        tr_x_z = tr_x/tr_z
        tr_y_z = tr_y/tr_z

        x_hat = f*tr_x_z + px
        y_hat = f*tr_y_z + py

        x = obs_val[i, :, 0]
        y = obs_val[i, :, 1]

        curr_residuals = np.zeros((len(x_hat),2))
        curr_residuals[:,0] = x_hat-x
        curr_residuals[:,1] = y_hat-y
        residuals = np.vstack((residuals, curr_residuals))

    return residuals.flatten()

def error_SSD(residuals):
    """[summary]

    Args:
        residuals ([type]): [description]

    Returns:
        [type]: [description]
    """    
    repj_error = 2*sqrt(np.sum(residuals**2, axis=0))/len(residuals)
    return repj_error

def objective_func(mot_str_vec, obs_idx, obs_val, ncameras, N, f, px, py):
    """[summary]

    Args:
        mot_str_vec ([type]): [description]
        obs_idx ([type]): [description]
        obs_val ([type]): [description]
        ncameras ([type]): [description]
        N ([type]): [description]
        f ([type]): [description]
        px ([type]): [description]
        py ([type]): [description]

    Returns:
        [type]: [description]
    """    
    # print('ncameras: ', ncameras)
    motion, structure = unpack_func_result(mot_str_vec, ncameras, N)

    residuals = reprojection_residuals(obs_idx, obs_val, motion, structure, f, px, py)
    # print('residuals: ', residuals.shape)
    # print('reprojection error before: ', error_SSD(residuals))

    return residuals


def bundle_adjustment(graph, px=0, py=0):
    """[summary]

    Args:
        graph ([type]): [description]
        px (int, optional): [description]. Defaults to 0.
        py (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """    

    start_time = time.time()

    motion, structure, f, ncameras, Npts = unpack_graph(graph)
    print('structure shape: ', structure.shape)
    residuals = reprojection_residuals(graph.obs_idx, graph.obs_val, motion, structure.T, f, px, py)
    print('residuals shape: ', residuals.shape)
    print('reprojection error before: ', error_SSD(residuals))
    mot_str_vec = pack_func_args(motion, structure)
    print('mot_str_vec: ', mot_str_vec.shape)
    fun = lambda x : objective_func(x, graph.obs_idx, graph.obs_val, ncameras, Npts, f, px, py)
    result = least_squares(fun=fun, x0=mot_str_vec,  method='lm', 
                                ftol=1e-08, xtol=1e-08, gtol=1e-08, 
                                max_nfev=1000, verbose=2)  #initial_guess jac='2-point', args=(graph.obs_idx, graph.obs_val, ncameras, Npts, f, px, py)
    
    motion, structure = unpack_func_result(result['x'], ncameras, Npts)
    print('structure shape: ', structure.shape)
    residuals = reprojection_residuals(graph.obs_idx, graph.obs_val, motion, structure, f, px, py)
    print('reprojection error final: ', error_SSD(residuals))

    print("Time since optimization start", time.time() - start_time)

    return graph


def adjust_focal_length(graph):





    return graph.f








