import numpy as np
from numpy.linalg import norm, inv 
from numpy import dot, sqrt, log, mean
np.random.seed(1234)
eps = 1e-08

def homogenize_points(points: np.array) -> np.array:
    """homogenizes points by adding row of 1's

    Args:
        points (np.array): [2xN]

    Returns:
        np.array: [3xN]
    """    
    # points should be of size 2xN
    points = np.vstack((points, np.ones((1, points.shape[1]))))

    return points

def de_homogenize_points(points: np.array) -> np.array:
    """de_homogenizes points by scaling the first two
    rows with the last row and then discarding the last row


    Args:
        points (np.array): [3xN]

    Returns:
        np.array: [2xN]
    """    
    # points should be of size 2xN
    points[-1,:] = points[-1,:] + eps
    points = points/(points[-1,:])

    return points[:-1,:]

def normalize_points(points: np.array) -> np.array:
    """normalizes points by shifting the origin to (0,0) and 
    then scaling them

    Args:
        points (np.array): [2xN]

    Returns:
        np.array: [2xN]
    """    
    # points are already homogenized
    N = points.shape[1]
    mean = np.mean(points[:-1,:],axis=1).reshape(2,-1)
    # mean_dist_origin = np.sum(norm(points[:-1,:], axis=0))/(N*(2**0.5))
    std = np.std(points[:-1,:],axis=1)
    # std = np.mean(std)
    # print('std mean: ', std)

    # construct a similarity transformation matrix
    # translation
    # print('mean_dist_origin: ', mean_dist_origin)
    # print('std: ', std)
    # print('mean: ', mean)
    T = np.block([[np.eye(points.shape[0]-1),  -mean],
                 [np.array([0,0,1])]])

    # scaling 
    # scale = sqrt(2)/(np.max(std)+ 1e-08)
    # print('scale1: ', scale)
    scale = sqrt(2)/(sqrt(np.sum(norm(points[:-1,:], axis=0))/N) + eps)
    # print('scale2: ', scale)
    S = np.eye(T.shape[0])
    S[0,0] = scale
    S[1,1] = scale
    T = dot(S, T)   
    # print('T: ', T)

    points_norm = dot(T, points)
    # verification
    # print('verification: ')
    # point1 = points[:-1,0]
    # print('point1: ', point1)    # print('Df: ', Df)
    # print('VfT: ', VfT)
    # print('Uf: ', Uf)
    # point1_norm = points_norm[:,0]
    # print('point1_norm: ', point1_norm)
    # point1_hat = dot(inv(T), point1_norm)
    # print('distance from origin: ', np.sum(norm(points_norm[:-1,:], axis=0))/N )
    # print('point1_hat: ', point1_hat)
    return points_norm, T

def denormalize_matrix(M, T1, T2):

    M_T1 = dot(M, T1)
    M = dot(T2.T, M_T1)

    return M

def ransac_iterations(min_points, inlier_fraction, probability):

    niterations = log(1-probability)/(log(1-inlier_fraction**min_points) + eps)
    niterations = int(niterations)

    return niterations


def point_distance(pt1: np.array, pt2: np.array) -> float:
    """euclidean distance between two points

    Args:
        pt1 (np.array): [1x2]
        pt2 (np.array): [1x2]

    Returns:
        float: [description]
    """    

    return norm(pt1-pt2)

def point_distance_mean(pts1: np.array, pts2: np.array) -> np.array:
    """euclidean distance - mean

    Args:
        pts1 (np.array): [Nx2 or Nx3]
        pts2 (np.array): [Nx2 or Nx3]

    Returns:
        np.array: [1x2 or 1x3]
    """    

    print('pts1.shape: ', pts1.shape)
    # assumes pts shape to be Nx2 or Nx3

    return norm(mean(pts1, axis=0)- mean(pts2, axis=0))


def transform_points_Rt(p_3d: np.array, Rt: np.array, is_inverse=False) -> np.array:
    """Transforms the 3d points by given Rt matrix

    Args:
        p_3d (np.array): [3xN]
        Rt (np.array): [3x4]
        is_inverse (bool, optional): [description]. Defaults to False.

    Returns:
        np.array: [3xN]
    """    

    if is_inverse:
        p_3d_new = dot(Rt[:,:3].T, p_3d -  np.tile(Rt[:,-1], p_3d.shape[1]))

    else:
        p_3d_new = dot(Rt[:,:3], p_3d) + np.tile(Rt[:,-1].reshape(-1,1), p_3d.shape[1])

    return p_3d_new

def calculate_camera_center(Rt: np.array) -> np.array:
    """calculates the camera center

    Args:
        Rt (np.array): [3x3]

    Returns:
        np.array: [3x1]
    """    
    
    cc = dot(-Rt[:,:3].T, Rt[:,-1])

    return cc


def concatenateRts(Rt_out: np.array, Rt_in: np.array) -> np.array:
    """multiply two Rt matrices

    Args:
        Rt_out (np.array): [3x4]
        Rt_in (np.array): [3x4]

    Returns:
        np.array: [3x4]
    """    

    # Rt * X = Rt_out * Rt_in * X
    Rt = np.zeros((3,4))
    Rt[:,:3] = dot(Rt_out[:,:3], Rt_in[:,:3])
    Rt[:,-1] = dot(Rt_out[:,:3], Rt_in[:,-1]) + Rt_out[:,-1]

    return Rt

def inverseRt(Rt_in: np.array) -> np.array:
    """invert an Rt matrix

    Args:
        Rt_in (np.array): [3x4]

    Returns:
        np.array: [3x4]
    """    
    Rt_out = np.zeros((3,4))
    Rt_out[:,:3] = Rt_in[:,:3].T
    Rt_out[:,-1] = dot(-Rt_in[:,:3].T, Rt_in[:,-1])

    return Rt_out

def transform_points_camera_to_world(P: np.array, pts_3d: np.array) -> np.array:
    """[summary]

    Args:
        P (np.array): [camera pose in world coordinates 4x4]
        pts_3d (np.array): [3xN]

    Returns:
        np.array: [description]
    """    
    pts_3d_w = dot(P[:3,:3], pts_3d - P[:3,-1])

    return pts_3d_w

def transform_points_world_to_camera(P: np.array, pts_3d: np.array) -> np.array:
    """[summary]

    Args:
        P (np.array): [camera pose 4x4]
        pts_3d (np.array): [points in world frame, 3xN]

    Returns:
        np.array: [points in camera frame, 3xN]
    """    

    #homogenize points
    pts_3d_hom = homogenize_points(pts_3d)
    pts_3d_hom_w = dot(P, pts_3d_hom)
    pts_3d_w = de_homogenize_points(pts_3d_hom_w)

    # another way
    # pts_3d_w = dot(P[:3,:3].T, pts_3d) + P[:3,-1]

    return pts_3d_w




