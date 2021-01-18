import cv2
import numpy as np 
from numpy.linalg import svd, inv
from numpy import dot, log
from src.sfm_pipeline.geometric_functions import homogenize_points, \
                                normalize_points, \
                                denormalize_matrix, \
                                ransac_iterations
np.random.seed(1234)
eps = 1e-08



def estimate_fundamental_matrix(pts1, pts2, ransac=False):
    # Normalized 8 point algorithm with RANSAC
    # assuming pts1 & pts2 are correspondence pairs

    # pts2.T*F*pts1 = 0
    # construct the matrix A with point pairs and use
    # A*f = 0
    # convert pts1 and pts2 to homogenous
    if not ransac:
        pts1 = homogenize_points(pts1.T)
        pts2 = homogenize_points(pts2.T)

    N = pts1.shape[1]
    # print('N :', N)
    min_points = 8
    assert(pts1.shape[1]==pts2.shape[1] & N>=min_points)
    A = np.zeros((N, 9))

    # normalize the pts
    pts1, T1 = normalize_points(pts1)
    pts2, T2 = normalize_points(pts2)

    for idx in range(N):

        XidxT = pts1[:,idx].T
        # print('XidxT: ', XidxT.shape)
        wdash_XidxT = pts2[-1,idx]*XidxT
        # print('wdash_XidxT: ', wdash_XidxT.shape)
        ydash_XidxT = pts2[1,idx]*XidxT
        # print('ydash_XidxT: ', ydash_XidxT.shape)
        xdash_XidxT = pts2[0,idx]*XidxT
        # print('xdash_XidxT: ', xdash_XidxT.shape)
        A[idx, :] = np.block([[xdash_XidxT, ydash_XidxT, wdash_XidxT]])

    # minimizing A*h, such that norm(f)==1 , h is column vector 9x1
    U, D, VT = svd(A)#, full_matrices=True)
    # print('D: ', D)
    # print('VT: ', VT)
    # f is the last column of v
    F = VT.T[:,-1]
    # reshape to matrix form
    F = F.reshape(-1,3)

    # rank reduction of F
    Uf, Df, VfT = svd(F)
    # print('Df: ', Df)
    # print('VfT: ', VfT)
    # print('Uf: ', Uf)
    # setting last element (least singular value) of Df to 0
    Df[-1] = 0
    # construct Df diagonal matrix from Df array
    Df = np.diag(Df)
    # print('Df: ', Df)
    F = dot(dot(Uf, Df), VfT)

    # Denormalize
    F = denormalize_matrix(F, T1, T2)
    # and rescale
   
    F[-1,-1] += eps
    F = F/(F[-1,-1])
    return F




def estimate_fundamental_matrix_ransac(pts1, pts2):
    # using sequential ransac to fit planes using the points
    # using Homography

    N = pts1.shape[0]
    # print('no.of point matches: ', N)
    min_points = 8
    assert(pts1.shape[0] == pts2.shape[0] & N >= min_points)

    pts1 = homogenize_points(pts1.T)
    pts2 = homogenize_points(pts2.T)

    # assume we are fitting for n planes
    #  RANSAC vars9
    inlier_fraction = 0.6
    probability = 0.95

    niterations = ransac_iterations(min_points, inlier_fraction, probability)
    # niterations = 100
    thresh = 0.01 #1 - probability

    best_F = np.eye(3) #np.zeros((3,3))
    nbest_inliers = 0     

    for iter_idx in range(niterations):

        # print('curr ransac iter: ', iter_idx)


        # select four feature point pairs
        # 4 point pairs are required to estimate homography
        idxs = np.random.choice(N, min_points, replace=False)
        # print('idxs: ', idxs)
        # compute homography

        F = estimate_fundamental_matrix(pts1[:,idxs], 
                                        pts2[:,idxs], 
                                        ransac=True)

        # find inliers
        epipolar_line = dot(dot(pts2[:,idxs].T, F), pts1[:,idxs])

        # the following is a numpy array to scalar comparison
        inlier_idxs = np.where(epipolar_line.T <= thresh)[0]
        # print('inlier_idxs: ', inlier_idxs)
        # print('len of inliers: ', len(inlier_idxs))

        if len(inlier_idxs) >= min_points:
            # compute homography using all the inlier point pairs
            F_inliers = estimate_fundamental_matrix(pts1[:,inlier_idxs], 
                                                    pts2[:,inlier_idxs], 
                                                    ransac=True)
        else:
            # print('inliers are less than 8')
            F_inliers = F

        if len(inlier_idxs) > nbest_inliers:
            # print('updating best_F')
            best_F = F_inliers
            nbest_inliers = len(inlier_idxs)
  

    # print('nbest_inliers: ', nbest_inliers)
    return best_F


def estimate_fundamental_matrix_opencv(img, pts1, pts2, kp2, matches):  

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)  #cv2.FM_RANSAC)
                                                                         
#    print('F: ', np.array(F, np.int64))
    kp2_matched = []
    for match in matches:
        kp2_matched.append(kp2[match[0].trainIdx])


    # out_img = cv2.drawKeypoints(img, kp2_matched, None, color=(255,0,0))
    # cv2.imshow('keypoints-matched', out_img)
    
    return F
