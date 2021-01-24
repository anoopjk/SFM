""" 

Author: Anoop Jakka
"""


import numpy as np
from numpy.linalg import svd
import cv2
from scipy.spatial.transform import Rotation as Rsci
from src.sfm_pipeline.min_reprojection_error import estimate_camera_matrix


def reprojection(P: np.array, p_3d: np.array) \
    -> np.array:
    """[summary]

    Returns:
        [type]: [description]
    """    
    # P and p_3d are homogenous
    p_2d_h =  P @ p_3d          #np.dot(P, p_3d)
    p_2d = p_2d_h[:-1,:]/p_2d_h[-1,:]
    
    return p_2d

def triangulation(P0: np.array, P1: np.array, p0_2d: np.array, p1_2d: np.array):
    # -> Tuple[np.array, np.array]:
    """[summary]

    Args:
        P0 (np.array): [description]
        P1 (np.array): [description]
        p0_2d (np.array): [description]
        p1_2d (np.array): [description]
    Returns:
        [type]: [description]
    """
    N = len(p0_2d)
    # print(P0.shape, P1.shape)
    # the 2d points should be of size 2xN 
    p1_3d_h = cv2.triangulatePoints(P0, P1, p0_2d.T, p1_2d.T) # P_3d11 is an 4xN  homogenious coordinates
    p1_3d = p1_3d_h[:-1,:]/p1_3d_h[-1,:]
#    print('P1_3d: ', P1_3d_h)
    p1_2d_hat = reprojection(P1, p1_3d_h)
    
    
    return p1_2d_hat, p1_3d

def cheirality_check(R, t, pts_3d):
    """[summary]

    Args:
        R ([type]): [description]
        t ([type]): [description]
        pts_3d ([type]): [description]

    Returns:
        [type]: [description]
    """    
    # for 3d points to be infront of camera
    # R[2,:]*(pts_3d - t) > 0
    # count the no.of such points
    # Rt config with max count is the valid one
    result = np.count_nonzero(R[2,:] @ (pts_3d[-1,:] - t) > 0)

    return result


def estimate_pose_5pt(p0_2d, p1_2d, K, F):
    """[summary]

    Args:
        p0_2d ([type]): [description]
        p1_2d ([type]): [description]
        K ([type]): [description]
        F ([type]): [description]

    Returns:
        [type]: [description]
    """    

    # calculate Essential matrix
    E = K.T @ F @ K
    ## calculate svd of E
    # U, S, Vt = svd(E)

    # E = U @ np.diag([1,1,0]) @ Vt

    # calculate svd again
    U, S, Vt = svd(E)
    W = np.array([[0, -1, 0], 
                [1, 0, 0],
                [0, 0, 1]])

    # there are two R's possible
    R1 = U @ W @ Vt  #np.dot(uW, vt) 
    # TODO remove
    r1 = Rsci.from_matrix(R1)
    R1 = r1.as_matrix()
    # print('R1: ', R1)

    R2 = U @ W.T @ Vt       #np.dot(uWt, vt)
    # TODO remove
    r2 = Rsci.from_matrix(R2)
    R2 = r2.as_matrix()
    # there are two t's possible
    t1 = U[:,-1].reshape(-1,1)
    t2 = -U[:,-1].reshape(-1,1)
    
    # need to identify right R and t
    P0 = np.eye(3,4)

    sol_array = np.empty((4,5), dtype=np.object)
    
    p1_3d_dict = {}

    # R1t1    
    R1t1 = np.hstack((R1, t1))
    P11 = K @ R1t1              #np.dot(K, R1t1)
    pose11 = np.eye(4)
    pose11[:3,:3] = R1
    pose11[:3,3] = t1.reshape(-1,)
    print('P11: ', P11)
    p1_2d11, p1_3d11 = triangulation(P0.copy(), P11.copy(), p0_2d.copy(), p1_2d.copy())
    p1_3d_dict['p1_2d11'] = [p1_2d11, p1_3d11, P11]
    sol_array[0,0] = 'p1_2d11'
    sol_array[0,1] = cheirality_check(R1, t1, p1_3d11)
    sol_array[0,2] = pose11
    sol_array[0,3] = p1_3d11
    sol_array[0,4] = p1_2d11

    # R1t2
    R1t2 = np.hstack((R1, t2))
    P12 =  K @ R1t2                 #np.dot(K, R1t2)
    pose12 = np.eye(4)
    pose12[:3,:3] = R1
    pose12[:3,3] = t2.reshape(-1,)
    p1_2d12, p1_3d12 = triangulation(P0, P12, p0_2d, p1_2d)
    p1_3d_dict['p1_2d12'] = [p1_2d12, p1_3d12, P12]
    sol_array[1,0] = 'p1_2d12'
    sol_array[1,1] = cheirality_check(R1, t2, p1_3d12)
    sol_array[1,2] = pose12
    sol_array[1,3] = p1_3d12
    sol_array[1,4] = p1_2d12

    # R2t1 
    R2t1 = np.hstack((R2, t1))
    P21 =  K @ R2t1                     #np.dot(K, R2t1)   
    pose21 = np.eye(4)
    pose21[:3,:3] = R2
    pose21[:3,3] = t1.reshape(-1,)
    p1_2d21, p1_3d21 = triangulation(P0, P21, p0_2d, p1_2d)    
    p1_3d_dict['p1_2d21'] = [p1_2d21, p1_3d21, P21]
    sol_array[2,0] = 'p1_2d21'
    sol_array[2,1] = cheirality_check(R2, t1, p1_3d21)
    sol_array[2,2] = pose21
    sol_array[2,3] = p1_3d21
    sol_array[2,4] = p1_2d21

    # R2t2  
    R2t2 = np.hstack((R2, t2))
    P22 =   K @ R2t2                        #np.dot(K, R2t2)  
    pose22 = np.eye(4)
    pose22[:3,:3] = R2
    pose22[:3,3] = t2.reshape(-1,)
    p1_2d22, p1_3d22 = triangulation(P0, P22, p0_2d, p1_2d)
    p1_3d_dict['p1_2d22'] = [p1_2d22, p1_3d22, P22]
    sol_array[3,0] = 'p1_2d22'
    sol_array[3,1] = cheirality_check(R2, t2, p1_3d22)
    sol_array[3,2] = pose22
    sol_array[3,3] = p1_3d22
    sol_array[3,4] = p1_2d22
    
    print(sol_array[:,1])
    idx = np.argmax(sol_array[:,1])
    print('idx: ', idx)
    if idx.size == 0:
        return [], [], np.eye(3)

    # print(sol_array[idx,0])
    pose = sol_array[idx,2]
    p_3d = sol_array[idx,3]
    p1_2d_reproj = sol_array[idx,4]
    
    return p1_2d_reproj, p_3d, pose   #, sol_array[idxs[0],2]


def estimate_pose_8pt(kpts0, kpts1, K0, K1, thresh=1, conf=0.99999):
    """
    Adapted from Superglue utils

    Args:
        kpts0 ([type]): [description]
        kpts1 ([type]): [description]
        K0 ([type]): [description]
        K1 ([type]): [description]
        thresh (int, optional): [description]. Defaults to 1.
        conf (float, optional): [description]. Defaults to 0.99999.

    Returns:
        [type]: [description]
    """
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    R_best, t_best, inliers_idxs = None, None, None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            R_best, t_best, inliers_idxs = (R, t, mask.ravel() > 0)

    # traingulate 3d points 
    # set the P0 to identity matrix
    P0 = np.eye(3,4)    
    Rt = np.hstack((R_best, t_best))
    P1 =  K1 @ Rt # K0 and K1 are same intrinsic matrices
    kpts1_hat, p_3d = triangulation(P0, P1, kpts0, kpts1)
    print('error before lm: ', reprojection_error_SSD(kpts1_hat, kpts1))
    if kpts1.shape[0] > 10:
        P1 = estimate_camera_matrix(kpts1, p_3d, P1)
        kpts1_hat, p_3d = triangulation(P0, P1, kpts0, kpts1)
        print('error after lm: ', reprojection_error_SSD(kpts1_hat, kpts1))

    pose = np.eye(4)
    pose[:3,:3] = R_best
    pose[:3,3] = t_best.reshape(-1,)
    return kpts1_hat, p_3d, pose
   

def reprojection_error_SSD(p_2d_hat, p_2d):
    """[summary]

    Args:
        p_2d_hat ([type]): [description]
        p_2d ([type]): [description]

    Returns:
        [type]: [description]
    """    
    diff = np.sum((p_2d_hat - p_2d.T)**2, axis=0)

    return np.sum(diff)

