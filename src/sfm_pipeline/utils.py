"""
Geometric utils for SFM

Author: Anoop Jakka
"""

import numpy as np
import cv2


def quat_2_rot():
    pass


def rot_2_quat():
    pass

def get_point_colors(p2, p_3d, img):
    
    clr_inds = np.asarray(p2.T, np.int64)
    #filter the outliers
    clr_inds[clr_inds[:,0] < 0, 0] = 0 
    clr_inds[clr_inds[:,0] > img.shape[1]-1, 0] = img.shape[1]-1
    clr_inds[clr_inds[:,1] < 0, 1] = 0
    clr_inds[clr_inds[:,1] > img.shape[0]-1, 1] = img.shape[0]-1
    # x-> cols and y-> rows
    p_3d_colors = img[clr_inds[:,1], clr_inds[:,0], :]

    return p_3d_colors

