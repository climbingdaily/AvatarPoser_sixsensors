import torch
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
from trdParties.human_body_prior.body_model.body_model import BodyModel
from trdParties.human_body_prior.tools.omni_tools import copy2cpu as c2c
from trdParties.human_body_prior.tools.rotation_tools import aa2matrot,matrot2aa,local2global_pose
import random
from utils import utils_transform

from scipy import signal

import glob
from IPython import embed
import time
import copy
import pickle


class AMASS_Dataset(Dataset):
    """Motion Capture dataset"""

    def __init__(self, opt):
        self.opt        = opt
        self.ws         = opt['window_size']
        self.num_input  = opt['num_input']
        self.input_dim  = opt['input_dim']  # 108, six joints
        self.batch_size = opt['dataloader_batch_size']
        self.file_path  = os.path.join(opt['dataroot'], opt['data_name'])
        self.index_list = []

        self.load_data()

    def generate_sublists(self, start, length):
        window_size = self.ws if self.opt['phase'] == 'train' else length
        indices = list(range(start, start + length))

        if length >= window_size:
            num_full_windows = length // window_size
            sublists = [indices[j * window_size:(j + 1) * window_size] for j in range(num_full_windows)]

            if length % window_size >= window_size / 2:
                sublists.append(indices[-window_size:])
        else:
            # 对于长度小于window_size的情况，重复最后一个索引直到子列表长度为window_size
            last_index = indices[-1] if indices else start  # 防止索引列表为空
            padding    = [last_index] * (window_size - length)  # 生成填充用的索引列表
            sublists   = [indices + padding]

        all_indices = [index for sublist in sublists for index in sublist]
    
        return sublists, list(set(all_indices))

    def load_data(self):
        with open(self.file_path, 'rb') as f:
            merged_data = pickle.load(f)
        
        data_list   = merged_data['data']
        length_list = merged_data['lenghts']

        self.data = {key: [] for key in ['rotation_local_full_gt_list', 
                                         'input_joints_params', 
                                         'root_orient', 'pose_body', 'trans', 'trans_vel', 
                                         'head_global_trans_list']}

        start = 0

        for _, (data, lenght) in enumerate(zip(data_list, length_list)):

            new_indexes, unique_indices = self.generate_sublists(start, lenght)
            for ni in new_indexes:
                if len(ni) > 0:
                    self.index_list.append(ni)
            start += len(unique_indices)

            for key in ['rotation_local_full_gt_list', 'input_joints_params', 'body_parms_list', 'head_global_trans_list']:
                indices = [i - start for i in unique_indices]
                if key == 'body_parms_list':
                    real_index = [i + 1 - start for i in unique_indices]
                    self.data['root_orient'].append(data[key]['root_orient'][real_index])
                    self.data['pose_body'].append(data[key]['pose_body'][real_index])
                    self.data['trans'].append(data[key]['trans'][real_index])
                    self.data['trans_vel'].append(data[key]['trans'][real_index] - data[key]['trans'][indices])
                else:
                    self.data[key].append(data[key][indices])
        
        for data in self.data:
            self.data[data] = torch.cat(self.data[data], axis=0)

        print(f'[Data loaded]: {start} frames loaded from {len(data_list)} files')
        print(f'[Data loaded]: {len(self.index_list)} samples loaded, each with {self.ws} frames')

    def __len__(self):
        return max(len(self.index_list), self.batch_size)

    def __getitem__(self, idx):
        
        indices = self.index_list[idx]

        input_data = self.data['input_joints_params'][indices].clone()
        output_gt  = self.data['rotation_local_full_gt_list'][indices].clone()

        head_trans = self.data['head_global_trans_list'][indices].clone()
        pelvis_vel = self.data['trans_vel'][indices].clone()
        pelvis_pos = self.data['trans'][indices] - self.data['trans'][indices][:1]

        head_trans[:, :3, -1] -= self.data['trans'][indices][:1]
        input_data[:, 72: 90] = (input_data[:, 72:90].reshape(-1, 6, 3) - input_data[:1, 72:75].unsqueeze(1)).reshape(-1, 18) # subtract the global translation  
        if input_data.shape[1] == 108 and self.input_dim == 54:    #input_data.shape[1] = 108 but self.input_dim = 54
            input_data = input_data[:, np.r_[18:36, 54:72, 81:90, 99:108]]
        elif input_data.shape[1] == 108 and self.input_dim == 90:
            input_data = input_data[:, np.r_[0:18, 24:36, 36:54, 60:72, 72:81, 84:90, 90:99, 102:108]]

        if self.opt['phase'] == 'train':
            return {'in': input_data.float(),
                    'gt': output_gt[-1:].float(),
                    'P': 1,
                    'Head_trans_global': head_trans[-1:].float(),
                    'pos_pelvis_gt'    : pelvis_pos[-1:].float(),
                    'vel_pelvis_gt'    : pelvis_vel[-1:].float()}

        else:
            body_parms_list = {'root_orient': self.data['root_orient'][indices].clone(), 'pose_body': self.data['pose_body'][indices].clone(), 'trans': pelvis_pos.clone()}

            return {'in': input_data.float(),
                    'gt': output_gt.float(),
                    'P': body_parms_list,
                    'Head_trans_global': head_trans.float(),
                    'pos_pelvis_gt'    : pelvis_pos.float(),
                    'vel_pelvis_gt'    : pelvis_vel.float()
                    }
