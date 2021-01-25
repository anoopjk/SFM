""" 

Author: Anoop Jakka
"""

import numpy as np
from src.sfm_graph.pair_graph import PairGraph
from src.sfm_graph.point_visibility import find_common_points_dist_assign


def merge_graphs(pg1, pg2):

    pair1 = pg1.frame_idxs
    pair2 = pg2.frame_idxs

    # merged_graph = PairGraph()

    kpts1 = pg1.kpts[1,:,:]
    kpts2 = pg2.kpts[0,:,:]

    p1_idxs, p2_idxs = find_common_points_dist_assign(kpts1, kpts2)
    # print(p1_idxs, p2_idxs)
    print('# common points: ', len(p1_idxs))

    print(pg1.kpts_idxs[p1_idxs, 0])
    print(pg2.kpts_idxs[p2_idxs, 1])





    return 0