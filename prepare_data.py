import os

import pickle
import torch
import numpy as np
from utils import utils_transform

from torch.utils.data import Dataset, DataLoader
from trdParties.human_body_prior.body_model.body_model import BodyModel
from trdParties.human_body_prior.tools.omni_tools import copy2cpu as c2c
from trdParties.human_body_prior.tools.rotation_tools import aa2matrot,matrot2aa,local2global_pose

dataroot_amass ="amass" # root of amass dataset

HEAD_POS = 15

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

def load_filepaths(dataroot, subset, phase):
    print(f"Preparing {phase} data")
    savedir = os.path.join("./data_fps60", dataroot_subset, phase)
    
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    split_file = os.path.join(dataroot, subset, phase + "_split.txt")
    try:
        with open(split_file, 'r') as f:
            filepaths = [line.rstrip('\n') for line in f]
            return filepaths, savedir
    except:
        raise ValueError("File not found: {}".format(split_file))
        

for dataroot_subset in ["MPI_HDM05", "BioMotionLab_NTroje", "CMU"]:
    print(dataroot_subset)
    for phase in ["train", "test"]:
        print(f"Preparing {phase} data")
        
        filepaths, savedir = load_filepaths("./data_split", dataroot_subset, phase)
        bm_male, bm_female = load_body_models()

        rotation_local_full_gt_list = []
        input_joints_params         = []
        body_parms_list             = []
        head_global_trans_list      = []

        idx = 0
        for filepath in filepaths:
            data  = dict()
            bdata = np.load(os.path.join('data_split', filepath), allow_pickle=True)
            # print(list(bdata.keys())) ### check keys of body data: ['trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses']
            try:
                framerate = bdata["mocap_framerate"]
                print("framerate is {}".format(framerate))
            except:
                print(filepath)
                print(list(bdata.keys()))       
                continue # skip shape.npz

            idx += 1

            print(idx)

            if framerate == 120:
                stride = 2
            elif framerate == 60:
                stride = 1

            bdata_poses = bdata["poses"][::stride,...]
            bdata_trans = bdata["trans"][::stride,...]
            subject_gender = bdata["gender"]

            bm = bm_male # if subject_gender == 'male' else bm_female

            # embed()
            body_parms = {
                'root_orient': torch.Tensor(bdata_poses[:, :3]),   #.to(comp_device), # controls the global root orientation
                'pose_body'  : torch.Tensor(bdata_poses[:, 3:66]), #.to(comp_device), # controls the body
                'trans'      : torch.Tensor(bdata_trans),          #.to(comp_device), # controls the global body position
            }

            body_parms_list = body_parms

            # embed()
            body_pose_world = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body','root_orient','trans']})

            # embed()
            # self.rotation_local_full_gt_list.append(body_parms['pose_body'])
            # self.rotation_local_full_gt_list.append(torch.Tensor(bdata['poses'][:, :66]))
            
            # === rotation_local_full_gt_list ===
            output_aa = torch.Tensor(bdata_poses[:, :66]).reshape(-1,3)
            output_6d = utils_transform.aa2sixd(output_aa).reshape(bdata_poses.shape[0],-1)
            rotation_local_full_gt_list = output_6d[1:] # ! shape: (num_frames, 132)

            # === position_global_full_gt_list ===
            # ! shape: (num_frames, 22, 3) position of joints relative to the world origin
            position_full_gt_world  = body_pose_world.Jtr[:,:22,:] 
            input_position_global   = position_full_gt_world[1:, [15,20,21], :]
            input_position_relative = position_full_gt_world[1:, [15,20,21], :] - position_full_gt_world[1:, [15,20,21], :]

            # ==== head global transformation ====
            position_head_world    = position_full_gt_world[:,15,:] # world position of head
            rotation_local_matrot  = aa2matrot(torch.tensor(bdata_poses).reshape(-1,3)).reshape(bdata_poses.shape[0],-1,9)
            rotation_global_matrot = local2global_pose(rotation_local_matrot, bm.kintree_table[0].long())
            head_rot_global_matrot = rotation_global_matrot[:,[15],:,:]
            head_global_trans      = torch.eye(4).repeat(position_head_world.shape[0],1,1)
            head_global_trans[:,:3,:3] = head_rot_global_matrot.squeeze()
            head_global_trans[:,:3,3]  = position_full_gt_world[:,15,:]
            head_global_trans_list     = head_global_trans[1:] # ! shape: (num_frames, 4, 4)

            # === head, left hand, right hand's global 6d rotation representations ===
            rotation_global_6d  = utils_transform.matrot2sixd(rotation_global_matrot.reshape(-1,3,3)).reshape(rotation_global_matrot.shape[0],-1,6)
            input_rot_global_6d = rotation_global_6d[1:,[15,20,21],:] # ! shape: (num_frames, 3, 6) head, left hand, right hand

            # === head, left hand, right hand's global 6d rotation velocity representations ===
            rotation_velocity_global_matrot   = torch.matmul(torch.inverse(rotation_global_matrot[:-1]),rotation_global_matrot[1:])
            rotation_velocity_global_6d       = utils_transform.matrot2sixd(rotation_velocity_global_matrot.reshape(-1,3,3)).reshape(rotation_velocity_global_matrot.shape[0],-1,6)
            input_rot_velocity_global_6d = rotation_velocity_global_6d[:,[15,20,21],:] # ! shape: (num_frames, 3, 6) head, left hand, right hand

            assert input_rot_global_6d.shape[0] == input_rot_velocity_global_6d.shape[0] == input_position_global.shape[0] == input_position_relative.shape[0], "shape mismatch"

            num_frames = input_rot_global_6d.shape[0]

            input_joints_params = torch.cat([input_rot_global_6d.reshape(num_frames,-1),              # ! shape: (num_frames, 3*6) 
                                             input_rot_velocity_global_6d.reshape(num_frames,-1),     # ! shape: (num_frames, 3*6)
                                             input_position_global.reshape(num_frames,-1),            # ! shape: (num_frames, 3*3) 
                                             input_position_relative.reshape(num_frames,-1)], dim=-1) # ! shape: (num_frames, 3*3)

            # training data
            data['rotation_local_full_gt_list'] = rotation_local_full_gt_list # local rotation,  (num_frames, 22*6)
            data['input_joints_params']         = input_joints_params         # global position, (num_frames, 22*3)
            data['body_parms_list']             = body_parms_list             # input data,  including global orientation, rotation velocity, global translation, relative translation
            data['head_global_trans_list']      = head_global_trans_list      # head global transformation (num_frames, 4, 4)

            # data properties
            data['framerate'] = 60
            data['gender']    = subject_gender
            data['filepath']  = filepath

            print(str(idx)+'/'+str(len(filepaths)))
            #embed()
            with open(os.path.join(savedir,'{}.pkl'.format(idx)), 'wb') as f:
                pickle.dump(data, f)
