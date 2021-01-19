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
from src.sfm_pipeline.feature_matching import Features
from src.sfm_pipeline.fundamental_matrix import estimate_fundamental_matrix_ransac
from src.visualization.camera_motion_visualizer import CameraViz
from src.sfm_pipeline.camera_pose import estimate_pose


#colors BGR format
yellow = (25, 225, 255)
cyan = (240, 240, 70)
magenta = (230, 50, 240)
black = (0,0,0)
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)

class SFM(object):
    def __init__(self, input_path):
        self.input = input_path
        pass

    def camera_intrinsics(self,img_shape):
        # camera intrinsics 
        fx = 520
        fy = fx
        cx = img_shape[1]/2  # midpoint on image x-axis (which is columns)
        cy = img_shape[0]/2  # midpoint on image y-axis (rows)
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1.0]], np.float64)
        
        return K

    def sfm_init(self,frames_path, frame_names):
        # detect feature points in the frames              
        img1 = cv2.imread(os.path.join(frames_path, frame_names[0]))
        if img1.shape[0] > 640:
            img1 = cv2.resize(img1,(480,640))
        ## TODO merge opencv and superglue configuration
        # kp1, des1 = feature_detection(img1)
        K = self.camera_intrinsics(img1.shape)
        # return img1, kp1, des1, K
        self.camera = CameraViz()
        return img1, K
    ################################################################################################################################

    def run_sfm(self):

        cv2.destroyAllWindows()    

        # convert the video to images
        frames_path = self.input
        frame_names = os.listdir(frames_path)
        skip_factor = 2
        print ('len of frame_names before skip: ', len(frame_names))
        frame_names.sort()
        frame_names = frame_names[:]
        frame_names = frame_names[::skip_factor] 
        print ('final len of frame_names: ', len(frame_names))


        # img1, kp1, des1, K = sfm_init(frames_path, frame_names)
        img1, K = self.sfm_init(frames_path, frame_names)

        for ctr, frame_name in enumerate(frame_names[1:]):
            print ("reading image: ", frame_name)
            img2 = cv2.imread(os.path.join(frames_path, frame_name))
            if img2.shape[0] > 640:
                img2 = cv2.resize(img2,(480,640))
            ## detect features
            ## TODO merge opencv and superglue configuration
            # kp2, des2 = feature_detection(img2)      
            # print ('len of kp1', len(kp1))
            # print ('len of kp2', len(kp2))


            # feature matching
            # if min(len(kp1), len(kp2)) >= 1:
                ## TODO merge opencv and superglue configuration
                # matches = feature_matching2(img1, img2, kp1, kp2, des1, des2)
            features = Features()
            pts1, pts2 = features.feature_extraction_matching(img1, img2)
            # print(pts1, pts2)
            
            if len(pts1) >= 8:                 
                
                # p1, p2, matches = matches_2_point_pairs(kp1, kp2, matches)
                # estimate the fundamental matrix given the  2d image point correspondences
                # F_opencv = estimate_fundamental_matrix_opencv(img2, p1, p2, kp2, matches)
                F_anoop = estimate_fundamental_matrix_ransac(pts1, pts2)

                # print('F_opencv: ', F_opencv)
                print('F_anoop: ', F_anoop)

                if F_anoop is None:
                    print('F is None')
                    continue

                # estimate camera pose
                R = estimate_pose(pts1, pts2, K, F_anoop)
                pose = np.eye(4)
                pose[:3,:3] = R
                print(R)
                self.camera.plot_camera_trajectory(pose)

            else:
                print('skipping F no.of matches < 5')
            
            
            # update the keypoints and descriptors
            img1 = img2

        self.camera.close_window()
        cv2.destroyAllWindows()   
        plt.show() 


        return

# if __name__ == '__main__' :
#     main()
      







