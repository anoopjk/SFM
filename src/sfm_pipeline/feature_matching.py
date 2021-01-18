"""
feature matching an extraction  

Author: Anoop Jakka
"""

import numpy as np
from src.sfm_pipeline.superglue_wrapper import SuperGlueFeature  as SGF

class Features(object):
    def __init__(self):
        # Initialize SuperGlueFeature object
        self.sgf = SGF()
        self.sgf.prepare_config()
        self.sgf.initialize_matching()


    def feature_extraction_matching(self, img1, img2):
        pts1, pts2 = self.sgf.feature_extraction_matching(img1, img2)
        # plot the matches
        self.sgf.plot_matches(img1, img2)

        return pts1, pts2


    def filter_matches(self):
        pass





