# examples/Python/Basic/visualization.py

import numpy as np
import open3d as o3d
import time
import os
from copy import deepcopy
# import ipdb

# ipdb.set_trace()

EPS = 1e-6

class SFMViz(object):

    def __init__(self):
        ## camera initalization
        ## The following creates a camera pyramid
        self.camera = o3d.geometry.TriangleMesh.create_cone(radius=0.5, height=0.5, resolution=4, split=1)
        self.camera.paint_uniform_color([0.9, 0.1, 0.1])
        # self.camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.camera.compute_vertex_normals()

        # plot the trajectory of a camera
        self.vis_camera = o3d.visualization.Visualizer()
        self.vis_camera.create_window()
        self.vis_camera.add_geometry(self.camera)
        self.camera_list = [self.camera]
        self.pcd = o3d.geometry.PointCloud()
        self.vis_pcd = o3d.visualization.Visualizer()
        self.vis_pcd.create_window()
        

    def plot_camera_trajectory(self, pose):

        self.camera.transform(pose)
        camera_new = deepcopy(self.camera).transform(pose)
        self.camera_list.append(camera_new)
        self.vis_camera.update_geometry(self.camera)
        self.vis_camera.poll_events()
        self.vis_camera.update_renderer()
        # time.sleep(0.5)


    def close_window(self):

        self.vis_camera.destroy_window()
        self.vis_pcd.destroy_window()

    def plot_cameras(self):

        o3d.visualization.draw_geometries(self.camera_list)

    def plot_pointcloud(self, p_3d, p_3d_color):

        self.pcd.points = o3d.utility.Vector3dVector(p_3d)
        self.pcd.colors = o3d.utility.Vector3dVector(p_3d_color)
        # o3d.visualization.draw_geometries([self.pcd])
        self.vis_pcd.add_geometry(self.pcd)
        # self.vis_pcd.update_geometry(self.pcd)
        self.vis_pcd.poll_events()
        self.vis_pcd.update_renderer()

    def plot_pointcloud_offline(self, p_3d, p_3d_color):

        self.pcd.points = o3d.utility.Vector3dVector(p_3d)
        self.pcd.colors = o3d.utility.Vector3dVector(p_3d_color)
        o3d.visualization.draw_geometries([self.pcd])

    def write_pointcloud(self, p_3d, p_3d_color, output_path, name=''):
        self.pcd.points = o3d.utility.Vector3dVector(p_3d)
        self.pcd.colors = o3d.utility.Vector3dVector(p_3d_color)
        
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        o3d.io.write_point_cloud(os.path.join(output_path, 'p3d' + str(name) + '.ply'), self.pcd)

        return
        
