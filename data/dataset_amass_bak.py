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
        self.opt = opt
        self.window_size = opt['window_size']
        self.num_input = opt['num_input']
        self.input_dim = opt['input_dim']  # 108, six joints

        self.batch_size = opt['dataloader_batch_size']
        dataroot = opt['dataroot']
        filenames_train = os.path.join(dataroot, '*/train/*.pkl')
        filenames_test = os.path.join(dataroot, '*/test/*.pkl')

        # CMU,BioMotionLab_NTroje,MPI_HDM05
        if self.opt['phase'] == 'train':
#            self.filename_list = glob.glob('data_fps60/*/train/*.pkl')
            self.filename_list = glob.glob(filenames_train)
        else:
#            self.filename_list = glob.glob('data_fps60/*/test/*.pkl')
            self.filename_list = glob.glob(filenames_test)

            print('-------------------------------number of test data is {}'.format(len(self.filename_list)))

    def __len__(self):

        return max(len(self.filename_list), self.batch_size)


    def __getitem__(self, idx):
        ws = self.window_size

        filename = self.filename_list[idx]
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        if self.opt['phase'] == 'train':
            while data['rotation_local_full_gt_list'].shape[0] <= ws:
                idx = random.randint(0,idx)
                filename = self.filename_list[idx]
                with open(filename, 'rb') as f:
                    data = pickle.load(f)

        rotation_local_full_gt_list = data['rotation_local_full_gt_list']
        input_joints_params         = data['input_joints_params' if 'input_joints_params' in data else 'hmd_position_global_full_gt_list']
        body_parms_list             = data['body_parms_list']
        head_global_trans_list      = data['head_global_trans_list']

        if self.opt['phase'] == 'train':
            if input_joints_params.shape[0] > ws:
                start = np.random.randint(input_joints_params.shape[0] - ws)
            else:
                raise ValueError("Error: The window size 'ws' is too large or 'input_joints_params' is too small.")
            input_data =         input_joints_params[start:start + ws+1,...].reshape(ws+1, -1).float()  # shape = (ws+1, 108)
            output_gt  = rotation_local_full_gt_list[start + ws:start + ws + 1,...].float()
            head_trans =      head_global_trans_list[start + ws:start + ws + 1,...]       # shape = ( 1, 4, 4)

            # body_parms_list has one more frame than input_joints_params
            pelvis_pos = body_parms_list['trans'][start + ws + 1: start + ws + 2,...]
            pelvis_vel = body_parms_list['trans'][start + ws + 1: start + ws + 2,...] - body_parms_list['trans'][start+ws:start+ws+1,...]

            # subtract the global translation from the first frame
            if input_data.shape[1] > self.input_dim:    #input_data.shape[1] = 108 but self.input_dim = 54
                input_data = input_data[:, np.r_[18:36, 54:72, 81:90, 99:108]]
                input_data[:, 36:45] = (input_data[:, 36:45].reshape(-1, 3, 3) - input_data[:, 36:39].unsqueeze(1)).reshape(-1, 9) # subtract the global translation
                head_trans[:, :3, -1] -= input_data[-2:-1, 36:39]
            else:
                input_data[:, 72: 90] = (input_data[:, 72:90].reshape(-1, 6, 3) - input_data[:, 72:75].unsqueeze(1)).reshape(-1, 18) # subtract the global translation  
                head_trans[:, :3, -1] -= input_data[-2:-1, 72:75]

            return {'in': input_data,
                    'gt': output_gt,
                    'P': 1,
                    'Head_trans_global': head_trans,
                    'pos_pelvis_gt'    : pelvis_pos,
                    'vel_pelvis_gt'    : pelvis_vel}

        else:
            input_data = input_joints_params.reshape(input_joints_params.shape[0], -1)[1:]
            output_gt  = rotation_local_full_gt_list[1:]

            if input_data.shape[1] > self.input_dim:    # input_data.shape[1] = 108 but self.input_dim = 54
                input_data = input_data[:, np.r_[18:36, 54:72, 81:90, 99:108]]

            return {'in': input_data.float(),
                    'gt': output_gt.float(),
                    'P': body_parms_list,
                    'Head_trans_global':head_global_trans_list[1:],
                    'pos_pelvis_gt':body_parms_list['trans'][2:],
                    'vel_pelvis_gt':body_parms_list['trans'][2:]-body_parms_list['trans'][1:-1]
                    }
