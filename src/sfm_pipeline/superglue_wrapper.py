from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
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
        self.match_threshold = 0.2
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

    def extract_features(self, frame, idx=0):
        
        frame_tensor = frame2tensor(frame, self.device)
        keys = ['keypoints', 'scores', 'descriptors']
        frame_data = self.matching.superpoint({'image': frame_tensor})
        frame_data = {k+str(idx): frame_data[k] for k in keys}
        frame_data['image'+str(idx)] = frame

        return frame_data


    def  match_features(self, frame0, frame1):


        # Create a window to display the demo.
        if not self.no_display:
            cv2.namedWindow('SuperGlue matches', cv2.WINDOW_AUTOSIZE) #cv2.WINDOW_NORMAL
            # cv2.resizeWindow('SuperGlue matches', (640*2, 480))
        else:
            print('Skipping visualization, will not show a GUI.')



        frame_tensor = frame2tensor(frame, device)
        pred = matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]

    def plot_matches(self):
        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=self.show_keypoints, small_text=small_text)

        if not self.no_display:
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data = {k+'0': pred[k+'1'] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)



        cv2.destroyAllWindows()
