"""[summary]

Author: Anoop Jakka
"""
import numpy as np
from numpy.linalg import norm 
from scipy.optimize import linear_sum_assignment



def find_common_points(pts1, pts2):
    pts1 = np.round(pts1)
    pts2 = np.round(pts2)
    print(pts1.view('d,d').reshape(-1))
    print(pts2.view('d,d').reshape(-1))
    p1_idxs = np.nonzero(np.in1d(pts1.view('d,d').reshape(-1), pts2.view('d,d').reshape(-1)))[0]
    # p2_idxs = np.nonzero(np.in1d(pts2.view('d,d').reshape(-1), pts1.view('d,d').reshape(-1)))[0]
    p2_idxs = []
    for p1_idx in p1_idxs:
        pt1 = pts1[p1_idx,:]
        p2_idxs.append(np.where(np.all(pts2 == pt1, axis=1))[0][0])

    p2_idxs = np.int64(p2_idxs)

    return p1_idxs, p2_idxs

def find_common_points_dist_min(pts1, pts2):
    tol = 2
    p1_idxs = []
    p2_idxs = []
    for p1_idx, p1 in enumerate(pts1):
        min_tol = tol
        min_tol_idx = 0
        match = False
        for p2_idx, p2 in enumerate(pts2):
            euclid_distance = norm(p1-p2)
            # print(p1, p2, euclid_distance)
            if euclid_distance <= min_tol:
                min_tol = euclid_distance
                min_tol_idx = p2_idx
                match = True

        if match:
            p1_idxs.append(p1_idx)
            p2_idxs.append(min_tol_idx)

    return p1_idxs, p2_idxs
    

def find_common_points_dist_assign(pts1, pts2):
    tol = 2
    p1_idxs = []
    p2_idxs = []
    similarity_matrix = np.zeros((len(pts1), len(pts2)))
    for p1_idx, p1 in enumerate(pts1):

        for p2_idx, p2 in enumerate(pts2):
            euclid_distance = norm(p1-p2)
            # print(p1, p2, euclid_distance)
            similarity_matrix[p1_idx, p2_idx] = euclid_distance

    # print('similarity matrix: ', similarity_matrix)
    matches = linear_sum_assignment(similarity_matrix)
    # print(matches)
    p1_idxs = np.where(similarity_matrix[matches[0], matches[1]] <= tol)[0]
    p2_idxs = matches[1][p1_idxs]
    
    return p1_idxs, p2_idxs


# if __name__ == "__main__":

#     pts1 = np.array([[1,2], 
#                     [3,4], 
#                     [2,3],
#                     [4,5],
#                     [1,1],
#                     [0,1],
#                     [2,1]])
#     pts2 = np.array([[2,1], 
#                     [2,3], 
#                     [3,4],
#                     [5,6],
#                     [1,1]])

#     p1_idxs, p2_idxs = find_common_points_dist_assign(pts1, pts2)
#     print(p1_idxs, p2_idxs)

