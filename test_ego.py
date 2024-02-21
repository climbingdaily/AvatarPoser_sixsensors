import os.path
import argparse
import logging
import time

import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R

from my_func import save_joint_data
from utils import utils_logger
from utils import utils_option as option
from utils import utils_visualize as vis
from utils import utils_transform
from models.select_model import define_Trainer, define_G
from trdParties.human_body_prior.tools.rotation_tools import aa2matrot,matrot2aa,local2global_pose
from trdParties.human_body_prior.body_model.body_model import BodyModel

# MOCAP_INIT = np.array([[-1, 0, 0], 
#                        [0, 0, 1], 
#                        [0, 1, 0]])
MOCAP_INIT = np.array([[1, 0, 0], 
                       [0, -1, 0], 
                       [0, 0, -1]])
MOCAP_INIT = torch.from_numpy(MOCAP_INIT).float()

class Ego_test(Dataset):
    """Motion Capture dataset"""

    def __init__(self, file_path, opt, data_type=0, input_dim=108):
        self.opt = opt
        self.batch_size  = opt['dataloader_batch_size']
        self.window_size = opt['window_size']
        self.num_input   = opt['num_input']
        self.phase       = opt['phase']
        self.file_path   = file_path
        self.data_type   = data_type
        self.input_dim   = input_dim
        self.is_load     = False

        print('Loading data from: ', self.file_path)

    def __len__(self):
        return 1

    def load_ego(self, data):
        # root, left ankle, right ankle, head, left hand, right hand
        frames = []
        for key, value in data.items():
            frames.append(value['frame'])    # key's frame
        
        # find the index of the overlapping frames
        overlapping_frames       = set(frames[0]).intersection(*frames[1:])
        overlapping_frames_index = [[frame.index(ov) for ov in overlapping_frames] for frame in frames]

        position_full_world  = torch.zeros(len(overlapping_frames), 6, 3).float()
        rotation_global_quat = torch.zeros(len(overlapping_frames), 6, 4).float()

        for i, (key, value) in enumerate(data.items()): 
            rotation_global_quat[:, i, :] = torch.tensor(value['rot'][overlapping_frames_index[i]]).float()
            position_full_world[:, i, :]  = torch.tensor(value['pos'][overlapping_frames_index[i]]).float()

        input_position_global   = position_full_world[1:, ...]
        input_position_relative = position_full_world[1:, ...] - position_full_world[:-1, ...]

        # === head, left hand, right hand's global 6d rotation representations ===
        rotation_global_matrot = aa2matrot(utils_transform.quat2aa(rotation_global_quat.reshape(-1, 4))).reshape(rotation_global_quat.shape[0],-1,3,3)
        input_rot_global_6d    = utils_transform.matrot2sixd(rotation_global_matrot.reshape(-1, 3, 3)).reshape(rotation_global_quat.shape[0],-1,6)[1:]

        rotation_velocity_global_matrot = torch.matmul(torch.inverse(rotation_global_matrot[:-1]),rotation_global_matrot[1:])
        input_rot_velocity_global_6d    = utils_transform.matrot2sixd(rotation_velocity_global_matrot.reshape(-1,3,3)).reshape(rotation_velocity_global_matrot.shape[0],-1,6)

        assert input_rot_global_6d.shape[0] == input_rot_velocity_global_6d.shape[0] == input_position_global.shape[0] == input_position_relative.shape[0], "shape mismatch"

        num_frames = input_rot_global_6d.shape[0]
        input_joints_params = torch.cat([input_rot_global_6d.reshape(num_frames,-1),              # ! shape: (num_frames, J*6) 
                                        input_rot_velocity_global_6d.reshape(num_frames,-1),     # ! shape: (num_frames, J*6)
                                        input_position_global.reshape(num_frames,-1),            # ! shape: (num_frames, J*3) 
                                        input_position_relative.reshape(num_frames,-1)], dim=-1) # ! shape: (num_frames, J*3)
        
        self.input_joints_params = input_joints_params
        self.global_trans_pelvis = position_full_world[1:, 0, :]
        self.global_trans_head   = position_full_world[1:, 3, :]
        self.gt_global_rot       = None
        self.gt_joints_rot       = None

    def load_amass(self, data):
        
        self.input_joints_params = data['input_joints_params']
        self.global_trans_pelvis = data['body_parms_list']['trans'][1:]
        self.global_trans_head   = data['head_global_trans_list'][:,:3, 3]
        self.gt_global_rot       = data['body_parms_list']['root_orient'][1:]
        self.gt_joints_rot       = data['body_parms_list']['pose_body'][1:]

    def load_data(self):
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
        if self.data_type == 0:
            self.load_ego(data)
        else:
            self.load_amass(data)


    def __getitem__(self, idx):
        if self.is_load is False:
            self.load_data()
            self.is_load = True
        input_data = self.input_joints_params.float()

        if input_data.shape[1] > self.input_dim:    #input_data.shape[1] = 108 but self.input_dim = 54
            input_data = input_data[:, np.r_[18:36, 54:72, 81:90, 99:108]]
            input_data[:, 36:45] = (input_data[:, 36:45].reshape(-1, 3, 3) - input_data[:1, 36:39].unsqueeze(1)).reshape(-1, 9) # subtract the global translation
        else:
            input_data[:, 72: 90] = (input_data[:, 72:90].reshape(-1, 6, 3) - input_data[:1, 72:75].unsqueeze(1)).reshape(-1, 18) # subtract the global translation  

        return {'in'                 : input_data,
                'Pelvis_trans_global': self.global_trans_pelvis.float(),
                'Head_trans_global'  : self.global_trans_head.float(),
                'gt_global_rot'      : self.gt_global_rot.float() if self.gt_global_rot is not None else 0,
                'gt_joints_rot'      : self.gt_joints_rot.float() if self.gt_joints_rot is not None else 0,
                }

def model_test(input_params, input_model, window_size):

    input_model.eval()
    E_global_orientation_list = []
    E_joint_rotation_list     = []
    E_SMPL_joints_list        = []
    E_SMPL_verts_list         = []

    with torch.no_grad():
        if input_params.shape[0] < window_size:

            for frame_idx in range(0,input_params.shape[0]):
                rot, pose, (joints, verts, _) = input_model(input_params[0:frame_idx+1].unsqueeze(0), 
                                                                    do_fk=True, 
                                                                    return_verts=True, 
                                                                    select_last=True)
                E_global_orientation_list.append(rot)
                E_joint_rotation_list.append(pose)
                E_SMPL_joints_list.append(joints)
                E_SMPL_verts_list.append(verts)

            E_global_orientation_tensor = torch.cat(E_global_orientation_list, dim=0)
            E_joint_rotation_tensor     = torch.cat(E_joint_rotation_list, dim=0)
            E_SMPL_joints_tensor        = torch.cat(E_SMPL_joints_list, dim=0)
            E_SMPL_verts_tensor         = torch.cat(E_SMPL_verts_list, dim=0)

        else:  

            input_list = []

            # first window
            for frame_idx in range(0,window_size):
                rot, pose, (joints, verts, faces) = input_model(input_params[0:frame_idx+1].unsqueeze(0), 
                                                            do_fk        = True,
                                                            return_verts = True,
                                                            select_last  = True)

                E_global_orientation_list.append(rot)
                E_joint_rotation_list.append(pose)
                E_SMPL_joints_list.append(joints)
                E_SMPL_verts_list.append(verts)

            E_global_orientation_list = torch.cat(E_global_orientation_list, dim=0)
            E_joint_rotation_list     = torch.cat(E_joint_rotation_list, dim=0)
            E_SMPL_joints_tensor      = torch.cat(E_SMPL_joints_list, dim=0)
            E_SMPL_verts_tensor       = torch.cat(E_SMPL_verts_list, dim=0)

            # window to the end
            for frame_idx in range(window_size,input_params.shape[0]):
                input_list.append(input_params[frame_idx-window_size+1:frame_idx+1,...].unsqueeze(0))
                
            input_tensor_2 = torch.cat(input_list, dim = 0)

            rot, pose, (joints, verts, _) = input_model(input_tensor_2,
                                                        do_fk        = True,
                                                        return_verts = True,
                                                        select_last  = True)

            E_global_orientation_tensor = torch.cat([E_global_orientation_list, rot], dim=0)
            E_joint_rotation_tensor     = torch.cat([E_joint_rotation_list, pose], dim=0)
            E_SMPL_joints_tensor        = torch.cat([E_SMPL_joints_tensor, joints], dim=0)
            E_SMPL_verts_tensor         = torch.cat([E_SMPL_verts_tensor, verts], dim=0)

    return E_global_orientation_tensor, E_joint_rotation_tensor, E_SMPL_joints_tensor, E_SMPL_verts_tensor, faces


def test(json_path, model_path, pkl_path, save_animation=False, resolution=(800,800), data_type=0):

    opt = option.parse(json_path, is_train=True)

    opt = option.dict_to_nonedict(opt)

    print('model_path:', model_path)
    if os.path.exists(model_path):
        opt['path']['pretrained_netG'] = model_path
        opt['path']['pretrained'] = model_path
    else:
        raise ValueError(f"Model path {model_path} does not exist")
    
    if model_path is not None:
        
        opt['path']['pretrained_netG'] = model_path
        opt['path']['pretrained'] = model_path

    model_name = os.path.basename(opt['path']['pretrained_netG'])[:-4]
    data_name = os.path.basename(pkl_path)[:-4]

    # configure logger
    logger_name = 'Ego_' + time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], f"Test_{model_name}_{data_name}_{logger_name}.log"))
    logger = logging.getLogger(logger_name)

    Trainer = define_Trainer(opt)
    Trainer.init_test()
    input_dim = Trainer.netG.module.linear_embedding.in_features

    test_set = Ego_test(pkl_path, opt, data_type=data_type, input_dim=input_dim)
    test_loader = DataLoader(test_set, 
                             batch_size=1,
                             shuffle=False, 
                             num_workers=1, 
                             drop_last=False, 
                             pin_memory=True)
    
    save_vertices = {'first_person': {}, 'second_person': {}}

    for index, test_data in enumerate(test_loader):
        logger.info(f"testing the sample {index+1}/{len(test_loader)}")

        rotations, poses, joints, verts, faces = model_test(test_data['in'].squeeze(), 
                                                            Trainer.netG, 
                                                            opt['datasets']['train']['window_size'])
        global_trans = test_data['Pelvis_trans_global'].permute(1,0,2).cpu().numpy()

        save_vertices['first_person']['opt_pose']  = (verts - joints[:, :1, :]).cpu().numpy() + global_trans
        
        save_vertices['frame_rate']             = 30
        save_vertices['faces']                  = faces.cpu().numpy()
        save_vertices['lidar_extrinsics'] = np.array([np.eye(4)] * len(joints))

        global_rot = utils_transform.sixd2quat(test_data['in'][0][:, :36].reshape(-1, 6))
        global_rot = global_rot[:, [1, 2, 3, 0]]
        print(global_rot.shape)
        save_joint_data(os.path.join("/home/guest/Documents/EgoMotionProject/2024-02-11-20h27m50s/results/blender_file", 
                                     os.path.basename(pkl_path)[:-4] + '_joints.pkl'), 
                        global_rot.reshape(-1, 6, 4).float().numpy(),
                        Trainer.bm, 
                        # joints.cpu().numpy())
                        test_data['in'][0][:, 72:90].reshape(-1, 6, 3).float().numpy() + global_trans[:1])

        if data_type == 1:
            global_rots = utils_transform.aa2sixd(test_data['gt_global_rot'][0]).to(Trainer.device)
            body_rots = utils_transform.aa2sixd(test_data['gt_joints_rot'][0].reshape(-1,3)).to(Trainer.device)
            joints, verts, faces = Trainer.netG.module.fk_module(global_rots, 
                                                                 body_rots.reshape(-1, 21, 6), 
                                                                 Trainer.bm,
                                                                 return_verts=True)
            save_joint_data(os.path.join("/home/guest/Documents/EgoMotionProject/2024-02-11-20h27m50s/results/blender_file", 
                                     os.path.basename(pkl_path)[:-4] + '_joints_gt.pkl'), 
                            torch.cat([test_data['gt_global_rot'][0], test_data['gt_joints_rot'][0]], axis=1).numpy(), 
                            Trainer.bm, 
                            test_data['in'][0][:, 72:90].reshape(-1, 6, 3).float().numpy() + global_trans[:1])
            save_vertices['second_person']['opt_pose'] = ((verts - joints[:, :1, :]).cpu().numpy() + global_trans)
            
        # predicted_angle = utils_transform.sixd2aa(self.E[:,:132].reshape(-1,6).detach()).reshape(self.E[:,:132].shape[0],-1).float()

        # Determine the filename for the output file
        pkl_file = os.path.join("/home/guest/Documents/EgoMotionProject/2024-02-11-20h27m50s/results/blender_file", 
                                     os.path.basename(pkl_path)[:-4] + f'_vertices.pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump(save_vertices, f)

        print(f"File is stored in {pkl_file}, {len(save_vertices['lidar_extrinsics'])} frames")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='options/test_sixposer.json')
    parser.add_argument('--save_animation', type=bool, default=False)
    parser.add_argument('--resolution', type=tuple, default=(800,800))
    parser.add_argument('--pkl_path', type=str, 
                        # default='/home/guest/Documents/EgoMotionProject/2024-02-11-20h27m50s/results/blender_file/joints.pkl',
                        default='/home/guest/github/AvatarPoser_sixsensors/data_fps60_J6/MPI_HDM05/train/4.pkl',
                        )
    parser.add_argument('--model_path', type=str, 
                        # default='/home/guest/github/AvatarPoser_sixsensors/results/AvatarPoseEstimation/models_6j/130000_data_fps60_J6_G.pth',
                        default='/home/guest/github/AvatarPoser_sixsensors/results/AvatarPoseEstimation/models/best_101_data_fps60_J6_G.pth',
                        # default='/home/guest/github/AvatarPoser_sixsensors/model_zoo/avatarposer.pth'
                        )
    parser.add_argument('--data_type', type=int, default=1, help='Ego or AMASS')
    args = parser.parse_args()

    test(args.json_path, args.model_path, args.pkl_path, args.save_animation, args.resolution, args.data_type)