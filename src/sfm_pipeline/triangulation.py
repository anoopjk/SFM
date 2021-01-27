"""
Adapted from Frank Dellaert's coursework project

Author: Anoop Jakka
"""
import numpy as np 
import time
import cv2
from scipy.optimize import least_squares

EPS = 1e-12


def triangulation_linear(P0: np.array, P1: np.array, p0_2d: np.array, p1_2d: np.array):
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
    p1_2d_hat = projection(P1, p1_3d)
    
    
    return p1_2d_hat, p1_3d

def objective_func(P_3d, P1, P2, p1_2d, p2_2d):
    """
        Calculates the difference in image (pixel coordinates) and returns 
        it as a 2*n_points vector

        Args: 
        - P_3d: 3d points vector 1x3N
        - P1: 3x4 projection matrix
        - P2: 3x4 projection matrix
        - p1_2d: 2xN points
        - p2_2d: 2xN points

        - **kwargs: dictionary that contains the 2D and the 3D points. You will have to
                    retrieve these 2D and 3D points and then use them to compute 
                    the reprojection error.
        Returns:
        -     diff: A 2*N_points-d vector (1-D numpy array) of differences between 
                    projected and actual 2D points. (the difference between all the x
                    and all the y coordinates)

    """
    # P_3d will be a vector, need to reshape it to matrix
    P_3d = P_3d.reshape(3,-1)
    # print('P_3d: ', P_3d.shape)
    residual1 = (projection(P1, P_3d) - p1_2d.T)**2
    residual2 = (projection(P2, P_3d) - p2_2d.T)**2
    residuals = np.concatenate((residual1.flatten(), residual2.flatten()))

    # print('residuals: ', residuals.shape)
      
    return residuals

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
    projected_points_2d = P @ points_3d
    projected_points_2d = projected_points_2d/projected_points_2d[-1,:]
    projected_points_2d = projected_points_2d[:-1,:]
    # print('projected_points_2d: ', projected_points_2d.shape)
    
    return projected_points_2d

def triangulation_nonlinear(P1, P2, pts1, pts2, P_3d0=None):
    '''
        Calls least_squres form scipy.least_squares.optimize and
        returns an estimate for the camera projection matrix

        Args:
        - P1: 3x4 Projection matrix
        - P2: 3x4 Projection matrix
        - pts1d: n x 2 array of known points (x_i, y_i) in image coordinates 
        - pts2d: n x 2 array of known points (x_i, y_i) in image coordinates 
        - P_3d0: n x 3 array of known points in 3D, (X_i, Y_i, Z_i, 1) 


        Returns:
        - X: 3xN estimated 3D points 

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
    # print('p_3d0: ', P_3d0.shape)
    # if there is no initial guess provided then estimate it using linear triangulation
    if P_3d0 is None:
        P_3d0 = triangulation_linear(P1, P2, pts1, pts2)

    # print(pts_3d.shape)
    # print(pts_2d.shape)
    # initial guess is nothing but P0
    P_3d0 = P_3d0.flatten()
    result = least_squares(fun=objective_func, x0=P_3d0, jac='2-point', method='lm', 
                                ftol=1e-08, xtol=1e-08, gtol=1e-08, 
                                max_nfev=5000, verbose=2, args=(P1, P2, pts1, pts2))  #initial_guess
    
    # P will be a vector, need to reshape it to matrix
    P_3d = result['x']
    # print('P: ', P)
    P_3d = P_3d.reshape(3,-1)

    print("Time since optimization start", time.time() - start_time)

    return P_3d


    

