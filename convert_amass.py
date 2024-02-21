import os 
import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from trdParties.human_body_prior.body_model.body_model import BodyModel
from trdParties.human_body_prior.tools.rotation_tools import aa2matrot,matrot2aa,local2global_pose

# load_data from pkl
# and save to a new pkl

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

def save_data(new_file_path, body_aa, Body_mdoel, joints_params):
    joints_data   = {}
    rotation_local_matrot  = aa2matrot(torch.tensor(body_aa).reshape(-1,3)).reshape(body_aa.shape[0],-1,9)
    rotation_global_matrot = local2global_pose(rotation_local_matrot, Body_mdoel.kintree_table[0][:22].long())
    body_quat              = R.from_matrix(rotation_global_matrot.reshape(-1, 3, 3)).as_quat().reshape(-1,22, 4)[1:, BODDY_JOINTS]

    for i, joint in enumerate(selecte_joints):
        joints_data[joint] = {'pos': joints_params[:, i],
                              'rot'  : body_quat[:, i],
                              'frame': (np.arange(len(joints_params))).tolist(),
                              }

    # save to new pkl
    with open(new_file_path, 'wb') as f:
        pickle.dump(joints_data, f) 
    print(f'[Data saved]: {new_file_path}')

def load_data(file_path):
    with open(file_path, 'rb') as f:
        merged_data = pickle.load(f)

    bm = load_body_models()[0]

    human_data    = merged_data['body_parms_list']
    joints_params = merged_data['input_joints_params'][:, 72:90].reshape(-1, 6, 3).float().numpy()
    joints_data   = {}
    body_aa       = torch.cat([human_data['root_orient'], human_data['pose_body']], axis=1).numpy()
    new_file_path = file_path.replace('.pkl', '_joints.pkl')
    
    save_data(new_file_path, body_aa, bm, joints_params)

load_data('/home/guest/github/AvatarPoser_sixsensors/data_fps60_J6/MPI_HDM05/train/4.pkl')