# examples/Python/Basic/visualization.py

import numpy as np
import open3d as o3d
import time
from copy import deepcopy
# import ipdb

# ipdb.set_trace()

EPS = 1e-6

class CameraViz(object):

    def __init__(self):
        ## camera initalization
        ## The following creates a camera pyramid
        self.camera = o3d.geometry.TriangleMesh.create_cone(radius=0.5, height=0.5, resolution=4, split=1)
        self.camera.paint_uniform_color([0.9, 0.1, 0.1])
        # self.camera = o3d.geometry.TriangleMesh.create_coordinate_frame()
        self.camera.compute_vertex_normals()

        # plot the trajectory of a camera
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.camera)
        self.camera_list = [self.camera]
        

    def plot_camera_trajectory(self, pose):

        self.camera.transform(pose)
        camera_new = deepcopy(self.camera).transform(pose)
        self.camera_list.append(camera_new)
        self.vis.update_geometry(self.camera)
        self.vis.poll_events()
        self.vis.update_renderer()
        # time.sleep(0.5)


    def close_window(self):

        self.vis.destroy_window()

    def plot_cameras(self):

        o3d.visualization.draw_geometries(self.camera_list)
