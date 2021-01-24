# following function is ported over from SFMedu

import numpy as np 
from numpy.linalg import norm
from numpy import arctan2, sqrt, dot, cos, sin


def rotation_mat_2_angle_axis(R):

    """
    The conversion of a rotation matrix to the angle-axis form is
    numerically problematic when the rotation angle is close to zero
    or to Pi. The following implementation detects when these two cases
    occurs and deals with them by taking code paths that are guaranteed
    to not perform division by a small number.
    """

    # x = k * 2 * sin(theta), where k is the axis of rotation.
    axis_angle = np.zeros((3,))
    axis_angle[0] = R[2, 1] - R[1, 2]
    axis_angle[1] = R[0, 2] - R[2, 0]
    axis_angle[2] = R[1, 0] - R[0, 1]

    """
    Since the right hand side may give numbers just above 1.0 or
    below -1.0 leading to atan misbehaving, we threshold them
    """
    cos_theta = min(max((R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0,  -1.0), 1.0)

    """
    sqrt is guaranteed to give non-negative results, so we only
    threshold above.
    """
    sin_theta = min(norm(axis_angle)/ 2.0, 1.0)

    # Use the arctan2 to get the right sign on theta
    theta = arctan2(sin_theta, cos_theta)

    """
    Case 1: sin(theta) is large enough, so dividing by it is not a
    problem. We do not use abs here, because while jets.h imports
    std::abs into the namespace, here in this file, abs resolves to
    the int version of the function, which returns zero always.

    We use a threshold much larger then the machine epsilon, because
    if sin(theta) is small, not only do we risk overflow but even if
    that does not occur, just dividing by a small number will result
    in numerical garbage. So we play it safe.
    """
    k_thresh = 1e-12
    if sin_theta > k_thresh or sin_theta < -k_thresh:
        r = theta / (2.0 * sin_theta)
        axis_angle = axis_angle * r
        return axis_angle

    """
    Case 2: theta ~ 0, means sin(theta) ~ theta to a good
    approximation.
    """
    if cos_theta > 0.0:
        axis_angle = axis_angle * 0.5
        return axis_angle

    """
     Case 3: theta ~ pi, this is the hard case. Since theta is large,
     and sin(theta) is small. Dividing by theta by sin(theta) will
     either give an overflow or worse still numerically meaningless
     results. Thus we use an alternate more complicated formula
     here.

     Since cos(theta) is negative, division by (1-cos(theta)) cannot
     overflow.
    """
    inv_one_minus_costheta = 1.0 / (1.0 - cos_theta)

    """
     We now compute the absolute value of coordinates of the axis
     vector using the diagonal entries of R. To resolve the sign of
     these entries, we compare the sign of axis_angle[i]*sin(theta)
     with the sign of sin(theta). If they are the same, then
     axis_angle[i] should be positive, otherwise negative.
    """

    for i in range(3):
        axis_angle[i] = theta * sqrt((R[i, i] - cos_theta) * inv_one_minus_costheta)
        if (sin_theta < 0.0 and axis_angle[i]) > 0.0 or (sin_theta > 0.0 and axis_angle[i] < 0.0):
            axis_angle[i] = -axis_angle[i]
        
    
    return axis_angle



def axis_angle_2_rotation_mat(axis_angle):

    theta2 = dot(axis_angle,axis_angle)
    if theta2 > 0.0:
        """
         We want to be careful to only evaluate the square root if the
         norm of the axis_angle vector is greater than zero. Otherwise
         we get a division by zero.
        """
        R = np.zeros((3,3))

        theta = sqrt(theta2)
        wx = axis_angle[0] / theta
        wy = axis_angle[1] / theta
        wz = axis_angle[2] / theta
        
        cos_theta = cos(theta)
        sin_theta = sin(theta)
        
        R[0, 0] =     cos_theta   + wx*wx*(1 -    cos_theta)
        R[1, 0] =  wz*sin_theta   + wx*wy*(1 -    cos_theta)
        R[2, 0] = -wy*sin_theta   + wx*wz*(1 -    cos_theta)
        R[0, 1] =  wx*wy*(1 - cos_theta)     - wz*sin_theta
        R[1, 1] =     cos_theta   + wy*wy*(1 -    cos_theta)
        R[2, 1] =  wx*sin_theta   + wy*wz*(1 -    cos_theta)
        R[0, 2] =  wy*sin_theta   + wx*wz*(1 -    cos_theta)
        R[1, 2] = -wx*sin_theta   + wy*wz*(1 -    cos_theta)
        R[2, 2] =     cos_theta   + wz*wz*(1 -    cos_theta)
    else:
        # cos At zero, we switch to using the first order Taylor expansion.
        R[0, 0] =  1
        R[1, 0] = -axis_angle[2]
        R[2, 0] =  axis_angle[1]
        R[0, 1] =  axis_angle[2]
        R[1, 1] =  1
        R[2, 1] = -axis_angle[0]
        R[0, 2] = -axis_angle[1]
        R[1, 2] =  axis_angle[0]
        R[2, 2] = 1

    return R


def rotate_points_axis_angle(axis_angle, pts):

    # pts are (Nx3), 3d points
    # pts = pts.T
    # print('pts.shape: ', pts.shape)
    theta_sqr = dot(axis_angle, axis_angle)

    if theta_sqr > 0.0:
        """
         Away from zero, use the rodriguez formula
        
           result = pt costheta + (w x pt) * sintheta + w (w . pt) (1 - costheta)
        
         We want to be careful to only evaluate the square root if the
         norm of the angle_axis vector is greater than zero. Otherwise
         we get a division by zero.
        """

        theta = sqrt(theta_sqr)
        w = axis_angle / theta
        w = w.reshape(1,3)
        cos_theta = cos(theta)
        sin_theta = sin(theta)

        w_cross_pts = dot(vec_2_skew_symmetric_mat(w), pts) 
        w_dot_pts = dot(w, pts)

        p1 = pts*cos_theta 
        p2 = w_cross_pts*sin_theta
        p3 = (1-cos_theta)*(dot(w.T, w_dot_pts))

        new_pts = p1 + p2 + p3


    else:
        """
         Near zero, the first order Taylor approximation of the rotation
         matrix R corresponding to a vector w and angle w is
        
           R = I + hat(w) * sin(theta)
        
         But sintheta ~ theta and theta * w = angle_axis, which gives us
        
          R = I + hat(w)
        
         and actually performing multiplication with the point pt, gives us
         R * pt = pt + w x pt.
        
         Switching to the Taylor expansion at zero helps avoid all sorts
         of numerical nastiness.
        """
        #w_cross_pt = np.cross(axis_angle, pts)
        w_cross_pts = dot(vec_2_skew_symmetric_mat(axis_angle),  pts)
        
        new_pts = pts + w_cross_pts
    

    return new_pts


def vec_2_skew_symmetric_mat(a_vec):
    a_vec = a_vec.reshape(-1,)
    ax = a_vec[0]
    ay = a_vec[1]
    az = a_vec[2]

    A_mat = np.array([[0, -az, ay], 
                    [az, 0, -ax], 
                    [-ay, ax, 0]
                     ])

    return A_mat