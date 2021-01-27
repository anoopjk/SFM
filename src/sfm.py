"""
sfm.py

Author: Anoop Jakka
"""

import os
import sys
sys.path.append('/home/anoop/projects/SFM/')
print(sys.path)

import cv2
import argparse
import numpy as np
from numpy import dot
import open3d as o3d
import matplotlib.pyplot as plt
from copy import deepcopy
from src.sfm_pipeline.feature_matching import Features
from src.sfm_pipeline.fundamental_matrix import estimate_fundamental_matrix_ransac
from src.visualization.sfm_visualizer import SFMViz
from src.sfm_pipeline.camera_pose import (estimate_pose_8pt, estimate_pose_5pt) 
from src.sfm_pipeline.utils import get_point_colors
from src.sfm_graph.pair_graph import PairGraph
from src.sfm_pipeline.bundle_adjustment import bundle_adjustment
from src.sfm_graph.merge_graphs import merge_graphs


#colors BGR format
yellow = (25, 225, 255)
cyan = (240, 240, 70)
magenta = (230, 50, 240)
black = (0,0,0)
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)

class SFM(object):
    def __init__(self, input_path, skip_factor):
        self.input = input_path
        self.skip_factor = skip_factor
        self.pair_graphs = []
        self.K = None
        self.img1 = None
        self.img2 = None
        self.pts1 = None
        self.pts2 = None
        self.pts_3d = None
        self.pts_3d_acc = np.empty((0,3))
        self.pts_3d_colors = None
        self.pts_3d_colors_acc = np.empty((0,3))
        self.min_lm_points = 10
        self.f = 100
        self.pose_i = np.eye(4)
        self.online_plot = False

    def camera_intrinsics(self,img_shape):
        # camera intrinsics 
        fx = self.f
        fy = fx
        cx = img_shape[1]/2  # midpoint on image x-axis (which is columns)
        cy = img_shape[0]/2  # midpoint on image y-axis (rows)
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1.0]], np.float64)
        
        return K

    def sfm_init(self,frames_path, frame_names):
        # detect feature points in the frames              
        self.img1 = cv2.imread(os.path.join(frames_path, frame_names[0]))
        if self.img1.shape[0] > 640:
            self.img1 = cv2.resize(self.img1,(480,640))
        ## TODO merge opencv and superglue configuration
        # kp1, des1 = feature_detection(img1)
        self.K = self.camera_intrinsics(self.img1.shape)
        # return img1, kp1, des1, K
        self.mapper = SFMViz()
        if self.online_plot:
            self.mapper.create_windows()

    def run_sfm(self):

        cv2.destroyAllWindows()    

        # convert the video to images
        frames_path = self.input
        frame_names = os.listdir(frames_path)
        print('img skip factor: ', self.skip_factor)
        print ('len of frame_names before skip: ', len(frame_names))
        frame_names.sort()
        frame_names = frame_names[:]
        frame_names = frame_names[::self.skip_factor] 
        print ('final len of frame_names: ', len(frame_names))


        # img1, kp1, des1, K = sfm_init(frames_path, frame_names)
        self.sfm_init(frames_path, frame_names)

        for ctr, frame_name in enumerate(frame_names[1:]):
            print ("reading image: ", frame_name)
            self.img2 = cv2.imread(os.path.join(frames_path, frame_name))
            if self.img2.shape[0] > 640:
                self.img2 = cv2.resize(self.img2,(480,640))
            ## detect features
            ## TODO merge opencv and superglue configuration
            # kp2, des2 = feature_detection(img2)      

            # feature matching
            # if min(len(kp1), len(kp2)) >= 1:
                ## TODO merge opencv and superglue configuration
                # matches = feature_matching2(img1, img2, kp1, kp2, des1, des2)
            features = Features()
            self.pts1, self.pts2 = features.feature_extraction_matching(self.img1, self.img2)
            # print(pts1, pts2)
            
            if len(self.pts1) >= self.min_lm_points:                 
                
                # p1, p2, matches = matches_2_point_pairs(kp1, kp2, matches)
                # estimate the fundamental matrix given the  2d image point correspondences
                # F_opencv = estimate_fundamental_matrix_opencv(img2, p1, p2, kp2, matches)
                # F_anoop = estimate_fundamental_matrix_ransac(pts1, pts2)

                # print('F_opencv: ', F_opencv)
                # print('F_anoop: ', F_anoop)

                # if F_anoop is None:
                #     print('F is None')
                #     continue
                
                # estimate camera pose
                # _, pts_3d, pose = estimate_pose(pts1.copy(), pts2.copy(), K, F_anoop)
                # print('pose: ', pose)
                # print('p_3d: ', pts_3d.shape)
                _, pts_3d, pose = estimate_pose_5pt(self.pts1.copy(), self.pts2.copy(), self.K, self.K)
                print('pose: ', pose)
                # self.mapper.plot_camera_trajectory(pose)
                # self.mapper.plot_pointcloud(pts_3d.T, get_point_colors(pts2, pts_3d, img2))

                # create pair graph
                pair_graph = PairGraph(pair_img_idxs=[ctr, ctr+1], 
                                    pose= pose,
                                    pts_3d=pts_3d, 
                                    pts1_2d=self.pts1, 
                                    pts2_2d=self.pts2, 
                                    f=self.K[0,0])

                # do pairwise bundle adjustment
                if len(self.pts1) >= self.min_lm_points:
                    pair_graph = bundle_adjustment(pair_graph, self.K[0,-1], self.K[1,-1])
                    pose = np.eye(4)
                    # pose[:3,:3] = pair_graph.motion[1,:,:-1]
                    pose[:3,:] = pair_graph.motion[1,:,:]
                    if self.online_plot:
                        self.mapper.plot_camera_trajectory(pose)
                    # save all the camera poses
                    self.mapper.stack_camera_poses(pose)


                # update the pose
                self.pose_i = self.pose_i @ pose

                self.pts_3d = pair_graph.structure
                self.pts_3d_acc = np.vstack((self.pts_3d_acc, self.pts_3d.T))
                self.pts_3d_colors = get_point_colors(self.pts2, self.pts_3d, self.img2)
                self.pts_3d_colors_acc = np.vstack((self.pts_3d_colors_acc, self.pts_3d_colors))
                # store the pair graphs
                self.pair_graphs.append(pair_graph)


            else:
                print('skipping F no.of matches < '+ str(self.min_lm_points))
            
            
            # update the keypoints and descriptors
            self.img1 = self.img2

        # end of image for loop

        # # merge pair_graphs
        # for idx in range(len(self.pair_graphs)-1):
        #     pg1, pg2 = self.pair_graphs[idx], self.pair_graphs[idx+1]
        #     merge_graphs(pg1, pg2)

        # plot the structure
        # self.mapper.plot_cameras_offline()
        # self.mapper.plot_pointcloud_offline(self.pts_3d_acc, self.pts_3d_colors_acc)
        self.mapper.plot_cameras_pointcloud_offline(self.pts_3d_acc, self.pts_3d_colors_acc)

        # self.mapper.close_windows_online()
        cv2.destroyAllWindows()   
        plt.show() 


        return

# if __name__ == '__main__' :
#     main()
      







