""" 

Author: Anoop Jakka
"""

import numpy as np
from src.sfm_graph.pair_graph import PairGraph
from src.sfm_graph.point_visibility import find_common_points_dist_assign
from copy import deepcopy


def merge_graphs(graph1, graph2):

    pair1 = graph1.frame_idxs
    pair2 = graph2.frame_idxs
    new_frame_idxs_graph2 = np.setdiff1d(graph2.frame_idxs, graph1.frame_idxs)

    graph_merged = deepcopy(graph1)
    graph_merged.frame_idxs = graph1.frame_idxs + new_frame_idxs_graph2

    kpts1 = graph1.kpts[1,:,:]
    kpts2 = graph2.kpts[0,:,:]

    p1_idxs, p2_idxs = find_common_points_dist_assign(kpts1, kpts2)
    # print(p1_idxs, p2_idxs)
    print('# common points: ', len(p1_idxs))

    # print(graph1.kpts_idxs[p1_idxs, 0])
    # print(graph2.kpts_idxs[p2_idxs, 1])
    if len(kpts2) > len(kpts1):
        diff = len(kpts2) - len(kpts1)
        graph_merged.kpts = np.vstack((graph_merged.kpts, -1*np.ones((2, diff, 2))))
        graph_merged.kpts = np.stack((graph_merged.kpts, graph2.kpts[1,:,:]))
    else:
        diff = len(kpts1) - len(kpts2)
        graph2.kpts = np.vstack((graph2.kpts, -1*np.ones((2, diff, 2))))
        graph_merged.kpts = np.stack((graph_merged.kpts, graph2.kpts[1,:,:]))

    



    return 0