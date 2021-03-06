import os
import h5py
import torch
import numpy as np
import scipy
import json

class CorresPondenceNet(torch.utils.data.Dataset):
    def __init__(self, cfg, flag='train'):
        super().__init__()
        with open(os.path.join(cfg['data_path'], 'name2id.json'), 'r') as f: 
            self.name2id = json.load(f)

        try:
            self.catg = self.name2id[cfg['class_name'].capitalize()]
        except:
            raise ValueError

        self.task = cfg['task_type']

        with h5py.File(os.path.join(cfg['data_path'], 'corr_mean_dist_geo', '{}_mean_distance.h5'.format(self.catg)), 'r') as f: 
            self.mean_distance = f['mean_distance'][:]
            
        if self.task == 'embedding':
            self.users = {}
            self.pcds = []
            self.keypoints = []
            self.num_annos = 0
            filename = os.path.join(
                cfg['data_path'], '{}.h5'.format(self.catg))
            with h5py.File(filename, 'r') as f:
                self.pcds = f['point_clouds'][:]
                self.keypoints = f['keypoints'][:]
                self.mesh_names = f['mesh_names'][:]
           
            num_train = int(self.pcds.shape[0] * 0.7)
            num_divide = int(self.pcds.shape[0] * 0.85)


            if flag == 'train':
                self.pcds = self.pcds[:num_train]
                self.keypoints = self.keypoints[:num_train]
                self.mesh_names = self.mesh_names[:num_train]
            elif flag == 'val':
                self.pcds = self.pcds[num_train:num_divide]
                self.keypoints = self.keypoints[num_train:num_divide]
                self.mesh_names = self.mesh_names[num_train:num_divide]
            elif flag == 'test':
                self.pcds = self.pcds[num_divide:]
                self.keypoints = self.keypoints[num_divide:]
                self.mesh_names = self.mesh_names[num_divide:]
            else:
                raise ValueError

            self.num_annos = self.pcds.shape[0]

        else:
            raise ValueError

    def __getitem__(self, item):
        if self.task == 'embedding':
            pcd = self.pcds[item]
            keypoint_index = np.array(self.keypoints[item], dtype=np.int32)
            return torch.tensor(pcd).float(), torch.tensor(keypoint_index).int(), torch.tensor(self.mean_distance).float(), 0
        else:
            raise ValueError

    def __len__(self):
        return self.num_annos