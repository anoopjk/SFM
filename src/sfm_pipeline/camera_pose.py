""" 

Author: Anoop Jakka
"""


import numpy as np
from numpy.linalg import svd
import cv2
from scipy.spatial.transform import Rotation as Rsci


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
    # print(p0_2d.shape, p1_2d.shape)
    # the 2d points should be of size 2xN 
    p1_3d_h = cv2.triangulatePoints(P0, P1, p0_2d.T, p1_2d.T) # P_3d11 is an 4xN  homogenious coordinates
    p1_3d = p1_3d_h[:-1,:]/p1_3d_h[-1,:]
#    print('P1_3d: ', P1_3d_h)
    p1_2d_hat = reprojection(P1, p1_3d_h)
    
    
    return p1_2d_hat, p1_3d


def estimate_pose(p0_2d, p1_2d, K, F):

    # calculate Essential matrix
    E = K.T @ F @ K
    # calculate svd of E
    U, S, Vt = svd(E)

    E = U @ np.diag([1,1,0]) @ Vt

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

    return R1
    
    # # need to identify right R and t
    # P0 = np.eye(3,4)

    # sol_array = np.empty((4,3), dtype=np.object)
    
    # p1_3d_dict = {}
    
    # # R1t1    
    # R1t1 = np.hstack((R1, t1))
    # P11 = np.dot(K, R1t1)
    # p1_2d11, p1_3d11 = triangulation(P0, P11, p0_2d, p1_2d)
    # p1_3d11_neg = np.count_nonzero(p1_3d11[-1,:]<0)
    # # print('p1_2d11 black: ', p1_3d11_neg)
    # # print('R1, t1: ', R1.flatten(), t1.flatten())
    # p1_3d_dict['p1_2d11'] = [p1_2d11, p1_3d11, P11]
    # sol_array[0,0] = 'p1_2d11'
    # sol_array[0,1] = p1_3d11_neg
    # sol_array[0,2] = R1
    # # R1t2
    # R1t2 = np.hstack((R1, t2))
    # P12 = np.dot(K, R1t2)
    # p1_2d12, p1_3d12 = triangulation(P0, P12, p0_2d, p1_2d)
    # p1_3d12_neg = np.count_nonzero(p1_3d12[-1,:]<0)
    # # print('p1_2d12 cyan: ', p1_3d12_neg)
    # # print('R1, t2: ', R1.flatten(), t2.flatten())
    # p1_3d_dict['p1_2d12'] = [p1_2d12, p1_3d12, P12]
    # sol_array[1,0] = 'p1_2d12'
    # sol_array[1,1] = p1_3d12_neg
    # sol_array[1,2] = R1
    # # R2t1 
    # R2t1 = np.hstack((R2, t1))
    # P21 = np.dot(K, R2t1)   
    # p1_2d21, p1_3d21 = triangulation(P0, P21, p0_2d, p1_2d)    
    # p1_3d21_neg = np.count_nonzero(p1_3d21[-1,:]<0)
    # # print('p1_2d21 magenta: ', p1_3d21_neg)
    # # print('R2, t1: ', R2.flatten(), t1.flatten())
    # p1_3d_dict['p1_2d21'] = [p1_2d21, p1_3d21, P21]
    # sol_array[2,0] = 'p1_2d21'
    # sol_array[2,1] = p1_3d21_neg
    # sol_array[2,2] = R2

    # # R2t2  
    # R2t2 = np.hstack((R2, t2))
    # P22 = np.dot(K, R2t2)  
    # p1_2d22, p1_3d22 = triangulation(P0, P22, p0_2d, p1_2d)
    # p1_3d22_neg = np.count_nonzero(p1_3d22[-1,:]< 0)
    # # print('p1_2d22 red: ', p1_3d22_neg)
    # # print('R2, t2: ', R2.flatten(), t2.flatten())
    # p1_3d_dict['p1_2d22'] = [p1_2d22, p1_3d22, P22]
    # sol_array[3,0] = 'p1_2d22'
    # sol_array[3,1] = p1_3d22_neg
    # sol_array[3,2] = R2
    
    # # try:
    # percent_total = 0.05*p1_2d.shape[1]
    # if 1:
    #     idxs = np.where(sol_array[:,1] <= percent_total)[0]

    #     if idxs.size == 0:
    #         return [], [], np.eye(3)

    #     # print(sol_array[idxs[0],0])
    #     result = p1_3d_dict[sol_array[idxs[0],0]]
    #     R = sol_array[idxs[0],2]
    #     # print('result shape: ', result[0].shape)
    #     # print('error before lm: ', reprojection_error_SSD(result[0], p1_2d))
    #     # if p1_2d.shape[0] > 10:
    #     #     P1 = estimate_camera_matrix(p1_2d, result[1], result[2])
    #     #     result[0], result[1] = triangulation(P0, P1, p0_2d, p1_2d)
        
    #     # print('error after lm: ', reprojection_error_SSD(result[0], p1_2d))
        
    #     return result[0], result[1], R   #, sol_array[idxs[0],2]
    
    # # except:
        
    #     # return [], [], P0




