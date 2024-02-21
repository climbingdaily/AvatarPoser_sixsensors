import os
import argparse
import json
import torch
import pickle
import numpy as np
import pickle as pkl
from scipy.spatial.transform import Rotation as R

from trdParties.human_body_prior.body_model.body_model import BodyModel
from trdParties.human_body_prior.tools.rotation_tools import aa2matrot,matrot2aa,local2global_pose

def merge_data(folder_path, file_type='train'):
    # input: folder path
    # output: find all the pkls subfolders and merge them into one pkl
    # type: 'train' or 'test'
    pkl_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pkl") and file_type in root:
                pkl_files.append(os.path.join(root, file))
    data_list = []
    length_list = []
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pkl.load(f)
            length_list.append(data['input_joints_params'].shape[0])
            data_list.append(data)
    totoal_length = sum(length_list)
    totoal_files = len(pkl_files)
    print(f"Total length: {totoal_length}, Total files: {totoal_files}, min length: {min(length_list)}, max length: {max(length_list)}")

    # save the merged data
    with open(f'{folder_path}/{file_type}_merged.pkl', 'wb') as f:
        pkl.dump({'data': data_list, 'lenghts': length_list}, f)
    

# merge_data('./data_fps60_J6', file_type='train')
# merge_data('./data_fps60_J6', file_type='test')
        

SMPL_JOINTS = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand']
BODDY_JOINTS = [0,7,8,15,20,21 ]

selecte_joints = [SMPL_JOINTS[i] for i in BODDY_JOINTS]

def load_body_models(support_dir='support_data/', num_betas=16, num_dmpls=8):
    # Load SMPL body models (here we load
    # @support_dir, path to the body model directory
    # @num_betas, body shape parameters
    # @num_dmpls, DMPL parameters
    bm_fname_male   = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format('male'))
    dmpl_fname_male = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format('male'))

    bm_fname_female   = os.path.join(support_dir, 'body_models/smplh/{}/model.npz'.format('female'))
    dmpl_fname_female = os.path.join(support_dir, 'body_models/dmpls/{}/model.npz'.format('female'))

    bm_male   = BodyModel(bm_fname=bm_fname_male, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_male)#.to(comp_device)
    bm_female = BodyModel(bm_fname=bm_fname_female, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname_female)

    return bm_male, bm_female

def save_joint_data(new_file_path, body_aa, Body_mdoel, joints_params):
    joints_data   = {}
    if body_aa.shape[-1] == 3 or body_aa.shape[-1] == 66:
        rotation_local_matrot  = aa2matrot(torch.tensor(body_aa).reshape(-1,3)).reshape(body_aa.shape[0],-1,9)
        rotation_global_matrot = local2global_pose(rotation_local_matrot, Body_mdoel.kintree_table[0][:22].long())
        body_quat              = R.from_matrix(rotation_global_matrot.reshape(-1, 3, 3)).as_quat().reshape(-1,22, 4)[:, BODDY_JOINTS]
    else:
        body_quat = body_aa
    for i, joint in enumerate(selecte_joints):
        joints_data[joint] = {'pos': joints_params[:, i],
                              'rot'  : body_quat[:, i],
                              'frame': (np.arange(len(joints_params))).tolist(),
                              }

    # save to new pkl
    with open(new_file_path, 'wb') as f:
        pickle.dump(joints_data, f) 
    print(f'[Data saved]: {new_file_path}')
