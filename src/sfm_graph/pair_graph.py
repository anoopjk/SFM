import numpy as np 

class PairGraph(object):

    def __init__(self, pair_img_idxs=[], pose=np.eye(4),
                pts_3d=np.empty((0,0)), pts1_2d=np.empty((0,0)), pts2_2d=np.empty((0,0)), f=1.0):
        
        ncameras = 2
        self.f = f
        self.motion = np.zeros((ncameras,3,4))
        self.motion[0,:,:] = np.eye(3,4)
        self.motion[1,:,:] = pose[:3,:]
        self.camera_center = np.zeros((3,1))
        self.structure = pts_3d
        self.frame_idxs = pair_img_idxs
        
        # self.matches = np.hstack((pts1_2d, pts2_2d))
        N = pts1_2d.shape[0]
        self.obs_val =  np.zeros((ncameras,N,2))            #np.hstack((pts1_2d, pts2_2d))
        self.obs_val[0,:,:] = pts1_2d
        self.obs_val[1,:,:] = pts2_2d
        self.obs_idx = np.zeros((N,ncameras), dtype=np.int32)
        self.obs_idx[:,0] = np.arange(0,N)
        self.obs_idx[:,1] = np.arange(0,N)








        


