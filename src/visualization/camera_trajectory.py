# examples/Python/Basic/visualization.py

import numpy as np
import open3d as o3d
import time
from quaternions import quat2mat
from copy import deepcopy
# import ipdb

# ipdb.set_trace()

EPS = 1e-6
def read_poses_rgbdscenes(pose_path):
    pose_matrices = np.zeros((4,4))
    with open(pose_path, 'r') as pose_file:
        for line in pose_file.readlines():
            pose_list = line.rstrip().split(' ')
            # print(pose_list)
            pose_vector = np.array(list(map(float, pose_list)))
            # print('pose vector: ', pose_list, pose_vector)
            # R = quat_2_rot(pose_vector[:4])
            # R = o3d.geometry.get_rotation_matrix_from_quaternion(pose_vector[:4])
            print(pose_vector[:4])
            R = quat2mat(pose_vector[:4])
            print(np.linalg.det(R))
            assert(abs(np.linalg.det(R) - 1) <= EPS )
            t = pose_vector[4:]
            pose_matrix = np.eye(4)
            pose_matrix[:3,:3] = R
            pose_matrix[:3,3] = t
            # stacking along 3rd axis
            pose_matrices = np.dstack((pose_matrices, pose_matrix))
    print(pose_matrices.shape)
    return pose_matrices[:,:,1:]

def plot_camera_trajectory(pose_path):

    # read the camera poses
    poses = read_poses_rgbdscenes(pose_path=pose_path)
    N = poses.shape[-1]
    print('number of poses: ', N)

    # The following creates a camera pyramid
    # camera = o3d.geometry.TriangleMesh.create_cone(radius=1.0, height=1.0, resolution=4, split=1)
    camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera.compute_vertex_normals()
    # camera.paint_uniform_color([0.9, 0.1, 0.1])
    ## open it in window
    # o3d.visualization.draw_geometries([camera])
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=0.6, origin=[-2, -2, -2])
    # o3d.visualization.draw_geometries(
    #     [camera, mesh_frame])

    # new_origin = [10,10,10]
    # transformation = np.identity(4)
    # transformation[:3,3] = new_origin
    # camera.transform(transformation)
    # o3d.visualization.draw_geometries([camera])

    # plot the trajectory of a camera
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(camera)
    camera_list = [camera]
    for i in range(N//10):
        print('processing frame: ', i)
        # quick check for translation freezing rotation
        poses[:3,:3,i] = np.eye(3) 
        poses[:3,3,i] =-(1/5)*poses[:3,3,i]
        print('translation: ', poses[:3,3,i])
        camera.transform(poses[:,:,i])
        camera_new = deepcopy(camera).transform(poses[:,:,i])
        camera_list.append(camera_new)
        vis.update_geometry(camera)
        vis.poll_events()
        vis.update_renderer()
        # time.sleep(0.5)

    print('end of frames')
    vis.destroy_window()

    o3d.visualization.draw_geometries(camera_list)

if __name__ == "__main__":
    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pose_path = '/media/seagate4TB/deeplearning/datasets/rgbd-scenes/rgbd-scenes-v2_pc/rgbd-scenes-v2/pc/07.pose'
    plot_camera_trajectory(pose_path)