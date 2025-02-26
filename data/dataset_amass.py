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

        filename = self.filename_list[idx]
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        if self.opt['phase'] == 'train':
            while data['rotation_local_full_gt_list'].shape[0] <self.window_size:
                idx = random.randint(0,idx)
                filename = self.filename_list[idx]
                with open(filename, 'rb') as f:
                    data = pickle.load(f)

        rotation_local_full_gt_list = data['rotation_local_full_gt_list']
        input_joints_params         = data['input_joints_params' if 'input_joints_params' in data else 'hmd_position_global_full_gt_list']
        body_parms_list             = data['body_parms_list']
        head_global_trans_list      = data['head_global_trans_list']

        if self.opt['phase'] == 'train':
            ws = self.window_size
            frame      = np.random.randint(input_joints_params.shape[0] - ws)
            input_data = input_joints_params[frame:frame + ws+1,...].reshape(ws+1, -1).float()
            output_gt  = rotation_local_full_gt_list[frame + ws - 1 : frame + ws,...].float()

            return {'in': input_data,
                    'gt': output_gt,
                    'P': 1,
                    'Head_trans_global': head_global_trans_list[frame + ws - 1:frame + ws,...],
                    'pos_pelvis_gt'    : body_parms_list['trans'][frame + ws: frame + ws - 1,...],
                    'vel_pelvis_gt'    : body_parms_list['trans'][frame + ws: frame + ws - 1,...] - body_parms_list['trans'][frame + ws - 1:frame + ws,...]
                    }

        else:
            input_data = input_joints_params.reshape(input_joints_params.shape[0], -1)[1:]
            output_gt  = rotation_local_full_gt_list[1:]

            return {'in': input_data.float(),
                    'gt': output_gt.float(),
                    'P': body_parms_list,
                    'Head_trans_global':head_global_trans_list[1:],
                    'pos_pelvis_gt':body_parms_list['trans'][2:],
                    'vel_pelvis_gt':body_parms_list['trans'][2:]-body_parms_list['trans'][1:-1]
                    }

        pass
    