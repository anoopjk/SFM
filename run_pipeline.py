""" 
Structure From Motion
Author: Anoop Jakka
"""

import argparse
from src.sfm import SFM


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run feature matching')
    parser.add_argument('--input', help='images path', 
    default='/media/seagate4TB/deeplearning/datasets/rgbd-scenes/rgbd-scenes-v2_imgs/rgbd-scenes-v2/imgs/scene_01/color')
    parser.add_argument('--skip-factor', help='skip factor image', type=int,
    default=5)
    args = parser.parse_args()

    # run structure from motion
    # initialize the SFM object
    sfm_obj = SFM(input_path=args.input, skip_factor=args.skip_factor)
    sfm_obj.run_sfm()








