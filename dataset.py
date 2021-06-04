import os
import h5py
import torch
import numpy as np
import scipy
import json

class CorresPondenceNet(torch.utils.data.Dataset):
    def __init__(self, cfg, flag='train'):
        super().__init__()
        with open(os.path.join(cfg['data_path'], 'synsetoffset2category.json'), 'r') as f: 
            self.name2id = json.load(f)

        try:
            self.catg = self.name2id[cfg['class_name'].capitalize()]
        except:
            raise ValueError

        self.task = cfg['task_type']
        self.laplacian_reg = cfg['laplacian_reg']

        with h5py.File(os.path.join(cfg['data_path'], 'kp_mean_distance_geodesic', '{}_mean_distance.h5'.format(self.catg)), 'r') as f: 
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
            self.pids = np.load(os.path.join(cfg['data_path'], '{}_pids.npy'.format(self.catg)))
           
            num_train = int(self.pcds.shape[0] * 0.7)
            num_divide = int(self.pcds.shape[0] * 0.85)

            # if self.laplacian_reg:
            #     # self.evecs = [-1 for i in range(self.pcds.shape[0])]
            #     self.evecs = []
            #     # read evecs
            #     for i, mesh in enumerate(self.mesh_names):
            #         # print(mesh)
            #         if os.path.exists(os.path.join(cfg['root_path'], 'evecs', mesh + '.evecs.npy')):
            #             evecs_curr = np.load(os.path.join(
            #                 cfg['root_path'], 'evecs', mesh + '.evecs.npy'))
            #         else:
            #             laplacian = pc2lap(self.pcds[i])
            #             _, evecs_curr = scipy.linalg.eigh(laplacian)
            #             np.save(os.path.join(
            #                 cfg['root_path'], 'evecs', mesh + '.evecs'), evecs_curr)
            #         self.evecs.append(evecs_curr)
            #     self.evecs = np.stack(self.evecs)
            # else:
            #     self.evecs = [-1 for i in range(self.pcds.shape[0])]

            if flag == 'train':
                self.pcds = self.pcds[:num_train]
                self.keypoints = self.keypoints[:num_train]
                # self.evecs = self.evecs[:num_train]
                self.mesh_names = self.mesh_names[:num_train]
            elif flag == 'val':
                self.pcds = self.pcds[num_train:num_divide]
                self.keypoints = self.keypoints[num_train:num_divide]
                # self.evecs = self.evecs[num_train:num_divide]
                self.mesh_names = self.mesh_names[num_train:num_divide]
            elif flag == 'test':
                self.pcds = self.pcds[num_divide:]
                self.keypoints = self.keypoints[num_divide:]
                # self.evecs = self.evecs[num_divide:]
                self.mesh_names = self.mesh_names[num_divide:]
            else:
                raise ValueError

            self.num_annos = self.pcds.shape[0]

        else:
            raise ValueError

    def __getitem__(self, item):
        if self.task == 'embedding':
            pcd = self.pcds[item]
            # evecs = self.evecs[item]
            keypoint_index = np.array(self.keypoints[item], dtype=np.int32)
            # return torch.tensor(pcd).float(), torch.tensor(keypoint_index).int(), torch.tensor(self.mean_distance).float(), torch.tensor(evecs).float()
            return torch.tensor(pcd).float(), torch.tensor(keypoint_index).int(), torch.tensor(self.mean_distance).float(), 0
        else:
            raise ValueError

    def __len__(self):
        return self.num_annos