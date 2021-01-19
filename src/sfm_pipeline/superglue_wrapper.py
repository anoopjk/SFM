from pathlib import Path
import argparse
import cv2
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch

from src.superglue.models.matching import Matching
from src.superglue.models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)


class SuperGlueFeature(object):
    def __init__(self):
        self.resize = [640, 480]
        self.superglue = 'indoor'
        self.max_keypoints = -1
        self.keypoint_threshold = 0.005
        self.nms_radius = 4
        self.sinkhorn_iterations = 20
        self.match_threshold = 0.5
        self.show_keypoints = True
        self.no_display = False
        self.force_cpu = False
        self.device = 'cuda' if torch.cuda.is_available() and not self.force_cpu else 'cpu'
        

    def prepare_config(self):
        if len(self.resize) == 2 and self.resize[1] == -1:
            self.resize = self.resize[0:1]
        if len(self.resize) == 2:
            print('Will resize to {}x{} (WxH)'.format(
                self.resize[0], self.resize[1]))
        elif len(self.resize) == 1 and self.resize[0] > 0:
            print('Will resize max dimension to {}'.format(self.resize[0]))
        elif len(self.resize) == 1:
            print('Will not resize images')
        else:
            raise ValueError('Cannot specify more than two integers for --resize')

        
        print('Running inference on device \"{}\"'.format(self.device))
        self.config = {
            'superpoint': {
                'nms_radius': self.nms_radius,
                'keypoint_threshold': self.keypoint_threshold,
                'max_keypoints': self.max_keypoints
            },
            'superglue': {
                'weights': self.superglue,
                'sinkhorn_iterations': self.sinkhorn_iterations,
                'match_threshold': self.match_threshold,
            }
        }

    def initialize_matching(self):
        self.matching = Matching(self.config).eval().to(self.device)

    def preprocess_input(self, frame):
        # convert color frame to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame


    def feature_extraction_matching(self, frame0, frame1):

        frame0 = self.preprocess_input(frame0)
        frame1 = self.preprocess_input(frame1)

        frame0_tensor = frame2tensor(frame0, self.device)
        keys = ['keypoints', 'scores', 'descriptors']
        frame0_data = self.matching.superpoint({'image': frame0_tensor})
        frame0_data = {k+'0': frame0_data[k] for k in keys}
        frame0_data['image0'] = frame0_tensor


        # Create a window to display the demo.
        if not self.no_display:
            cv2.namedWindow('SuperGlue matches', cv2.WINDOW_AUTOSIZE) #cv2.WINDOW_NORMAL
            # cv2.resizeWindow('SuperGlue matches', (640*2, 480))
        else:
            print('Skipping visualization, will not show a GUI.')

        frame1_tensor = frame2tensor(frame1, self.device)
        self.pred = self.matching({**frame0_data, 'image1': frame1_tensor})
        self.kpts0 = frame0_data['keypoints0'][0].cpu().numpy()
        self.kpts1 = self.pred['keypoints1'][0].cpu().numpy()
        self.matches = self.pred['matches0'][0].cpu().numpy()
        self.confidence = self.pred['matching_scores0'][0].cpu().numpy()

        self.valid = self.matches > -1
        self.mkpts0 = self.kpts0[self.valid]
        self.mkpts1 = self.kpts1[self.matches[self.valid]]

        mkpts0, mkpts1 = np.round(self.mkpts0).astype(int), np.round(self.mkpts1).astype(int)

        return mkpts0, mkpts1

    def plot_matches(self, frame0, frame1):

        frame0 = self.preprocess_input(frame0)
        frame1 = self.preprocess_input(frame1)

        color = cm.jet(self.confidence[self.valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(self.kpts0), len(self.kpts1)),
            'Matches: {}'.format(len(self.mkpts0))
        ]
        k_thresh = self.matching.superpoint.config['keypoint_threshold']
        m_thresh = self.matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(0, 1),
        ]
        out = make_matching_plot_fast(
            frame0, frame1, self.kpts0, self.kpts1, self.mkpts0, self.mkpts1, color, text,
            path=None, show_keypoints=self.show_keypoints, small_text=small_text,
            opencv_display=True)

        # if not self.no_display:
        # cv2.imshow('SuperGlue matches', out)
        # plt.imshow(out)


        # cv2.destroyAllWindows()
