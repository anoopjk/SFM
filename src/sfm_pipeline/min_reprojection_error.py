import numpy as np 
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm, inv
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.linalg import rq

import time

def objective_func(P, p_3d, p_2d):
    """
        Calculates the difference in image (pixel coordinates) and returns 
        it as a 2*n_points vector

        Args: 
        -        P: numpy array of 11 parameters of P in vector form 
                    (remember you will have to fix P_34=1) to estimate the reprojection error
        - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                    retrieve these 2D and 3D points and then use them to compute 
                    the reprojection error.
        Returns:
        -     diff: A 2*N_points-d vector (1-D numpy array) of differences between 
                    projected and actual 2D points. (the difference between all the x
                    and all the y coordinates)

    """
    # print(P.shape, p_3d.shape, p_2d.shape)
    # P will be a vector, need to reshape it to matrix, after appending 1 to it
    P = np.append(P, [1.0])
    P = P.reshape(3,4)

    diff = np.sum((projection(P, p_3d) - p_2d.T)**2, axis=0)
    # print('P: ', P)
    # print(diff.shape)
      
    return diff.flatten()

def projection(P: np.ndarray, points_3d: np.ndarray) -> np.ndarray:
    """
        Computes projection from [X,Y,Z,1] in homogenous coordinates to
        (x,y) in non-homogenous image coordinates.

        Args:
        -  P: 3x4 projection matrix
        -  points_3d : n x 4 array of points [X_i,Y_i,Z_i,1] in homogenous coordinates
                       or n x 3 array of points [X_i,Y_i,Z_i]

        Returns:
        - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """
    # homogenize points_3d
    points_3d = np.vstack((points_3d, np.ones(points_3d.shape[1])))
    projected_points_2d = dot(P, points_3d)
    projected_points_2d = projected_points_2d/projected_points_2d[-1,:]
    projected_points_2d = projected_points_2d[:-1,:]
    # print('projected_points_2d: ', projected_points_2d.shape)
    
    return projected_points_2d

def estimate_camera_matrix(pts_2d: np.ndarray, 
                           pts_3d: np.ndarray, 
                           initial_guess: np.ndarray):
    '''
        Calls least_squres form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - pts2d: n x 2 array of known points (x_i, y_i) in image coordinates 
        - pts3d: n x 3 array of known points in 3D, (X_i, Y_i, Z_i, 1) 
        - initial_guess: 3x4 projection matrix initial guess

        Returns:
        - P: 3x4 estimated projection matrix 

        Note: Because of the requirements of scipy.optimize.least_squares
              you will have to pass the projection matrix P as a vector.
              Since we will fix P_34 to 1 you will not need to pass all 12
              matrix parameters. 
            
              You will also have to put pts2d and pts3d into a kwargs dictionary
              that you will add as an argument to least squares.
              
              We recommend that in your call to least_squares you use
              - method='lm' for Levenberg-Marquardt
              - verbose=2 (to show optimization output from 'lm')
              - max_nfev=50000 maximum number of function evaluations
              - ftol \
              - gtol  --> convergence criteria
              - xtol /
              - kwargs -- dictionary with additional variables 
                          for the objective function
    '''

    start_time = time.time()

    # print(pts_3d.shape)
    # print(pts_2d.shape)
    # initial guess is nothing but P0
    P0 = initial_guess.flatten()
    P0 = P0[:-1]
    # print('P0 shape: ', P0.shape)
    result = least_squares(fun=objective_func, x0=P0, jac='2-point', method='lm', 
                                ftol=1e-08, xtol=1e-08, gtol=1e-08, 
                                max_nfev=5000, verbose=2, args=(pts_3d, pts_2d))  #initial_guess
    
    # P will be a vector, need to reshape it to matrix, after appending 1 to it
    # print('result: ', result)
    P = result['x']
    # print('P: ', P)
    P = np.append(P, [1.0])
    P = P.reshape(3,4)

    print("Time since optimization start", time.time() - start_time)

    return P

def decompose_camera_matrix(P: np.ndarray) -> (np.ndarray, np.ndarray):
    '''
        Decomposes the camera matrix into the K intrinsic and R rotation matrix
        
        Args:
        -  P: 3x4 numpy array projection matrix
        
        Returns:

        - K: 3x3 intrinsic matrix (numpy array)
        - R: 3x3 orthonormal rotation matrix (numpy array)

        hint: use scipy.linalg.rq()
    '''
    # remove the last row from P to make 3x4
    # P= P[:-1,:]

    K, R = rq(P[:,:3])
    
    return K, R

def calculate_camera_center(P: np.ndarray,
                            K: np.ndarray, 
                            R_T: np.ndarray) -> np.ndarray:
    """
    Returns the camera center matrix for a given projection matrix.

    Args:
    -   P: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """
    # remove the last row from P to make 3x4
    # P = P[:-1,:]
    
    cc = dot(-R_T, P[:,-1])

    return cc





