""" 

Author: Anoop Jakka
"""
import numpy as np
import open3d as o3d
from src.sfm_pipeline.utils import quat_2_rot



def read_poses_rgbdscenes(pose_path):
    pose_matrices = np.zeros((0,4,4))
    with open(pose_path, 'r') as pose_file:
        for line in pose_file.readlines():
            pose_vector = np.array(map(int, line))
            R = quat_2_rot(pose[:4])
            t = pose[4:]
            pose_matrix = np.stack([[R,      t],
                                   [[0,0,0], 1]])
            pose_matrices = np.stack((pose_matrices, pose_matrix), axis=0)

    return pose_matrices


def map_visualization(args):

    poses = read_poses(pose_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(my_pointcloud)


    # K = 3x3 intrinsics
    # P = 4x4 pose
    ctr = vis.get_view_control()
    init_param = ctr.convert_to_pinhole_camera_parameters()
    w, h = 640, 480
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    init_param.intrinsic.width = w
    init_param.intrinsic.height = h
    init_param.intrinsic.set_intrinsics(init_param.intrinsic.width, init_param.intrinsic.height, fx, fy, cx, cy)
    init_param.extrinsic = P
    ctr.convert_from_pinhole_camera_parameters(init_param)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 168/255., 1.0])
    vis.run()
    vis.destroy_window()



if __name__ == '__main__':

    