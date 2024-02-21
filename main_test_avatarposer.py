import os.path
import logging
import pickle

import argparse
import numpy as np
import torch

from scipy.spatial.transform import Rotation as R

from my_func import save_joint_data
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Trainer
from utils import utils_visualize as vis
from trdParties.human_body_prior.body_model.body_model import BodyModel
from trdParties.human_body_prior.tools.rotation_tools import aa2matrot,matrot2aa,local2global_pose


save_animation = False
resolution = (800,800)

def test(json_path='options/test_avatarposer.json', model_path=None):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    opt = option.parse(json_path, is_train=True)
    tag =os.path.basename(opt['datasets']['test']['dataroot'])

    print('model_path:', model_path)
    if os.path.exists(model_path):
        opt['path']['pretrained_netG'] = model_path
    else:
        raise ValueError(f"Model path {model_path} does not exist")
    
    if model_path is not None:
        opt['path']['pretrained_netG'] = model_path

    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        if not os.path.exists(paths):
            os.makedirs(paths)
    else:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type=f'{tag}_G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter

    # save opt to  a '../option.json' file
    option.save(opt)

    # return None for missing key
    opt = option.dict_to_nonedict(opt)

    # configure logger
    logger_name = 'test'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # 1) create_dataset
    dataset_opt = opt['datasets']['test']
    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                shuffle=False, num_workers=1,
                                drop_last=False, pin_memory=True)

    # Step--3 (initialize model)
    model = define_Trainer(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    model.init_test()

    pos_error = []
    vel_error = []
    pos_error_hands = []

    for index, test_data in enumerate(test_loader):
        logger.info("testing the sample {}/{}".format(index, len(test_loader)))
        model.feed_data(test_data, test=True)
        model.test()

        body_parms_pred    = model.current_prediction()
        body_parms_gt      = model.current_gt()

        predicted_angle    = body_parms_pred['pose_body']
        predicted_position = body_parms_pred['position']
        predicted_body     = body_parms_pred['body']

        gt_angle    = body_parms_gt['pose_body']
        gt_position = body_parms_gt['position']
        gt_body     = body_parms_gt['body']

        # save_joint_data(os.path.join("/home/guest/Documents/EgoMotionProject/2024-02-11-20h27m50s/results/blender_file", f'{index}_joints.pkl'),
        #                 torch.cat([body_parms_pred['root_orient'], body_parms_pred['pose_body']], axis=1).reshape(-1, 22, 3).detach().cpu().numpy(),
        #                 model.bm, 
        #                 test_data['in'][0][:, 72:90].reshape(-1, 6, 3).float().numpy())
        # save_joint_data(os.path.join("/home/guest/Documents/EgoMotionProject/2024-02-11-20h27m50s/results/blender_file", f'{index}_gt_joints.pkl'),
        #                 torch.cat([body_parms_gt['root_orient'], body_parms_gt['pose_body']], axis=1).reshape(-1, 22, 3).detach().cpu().numpy(),
        #                 model.bm, 
        #                 test_data['in'][0][:, 72:90].reshape(-1, 6, 3).float().numpy())
        

        if index in [0, 10, 20] and save_animation:
            video_dir = os.path.join(opt['path']['images'], str(index))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            save_video_path_gt = os.path.join(video_dir, 'gt.avi')
            if not os.path.exists(save_video_path_gt):
                vis.save_animation(body_pose=gt_body, savepath=save_video_path_gt, bm = model.bm, fps=60, resolution = resolution)

            save_video_path = os.path.join(video_dir, '{:d}.avi'.format(current_step))
            vis.save_animation(body_pose=predicted_body, savepath=save_video_path, bm = model.bm, fps=60, resolution = resolution)

        predicted_position = predicted_position#.cpu().numpy()
        gt_position = gt_position#.cpu().numpy()

        predicted_angle = predicted_angle.reshape(body_parms_pred['pose_body'].shape[0],-1,3)                    
        gt_angle = gt_angle.reshape(body_parms_gt['pose_body'].shape[0],-1,3)


        pos_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1)))
        pos_error_hands_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))[...,[20,21]])

        gt_velocity = (gt_position[1:,...] - gt_position[:-1,...])*60
        predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*60
        vel_error_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))

        pos_error.append(pos_error_)    
        vel_error.append(vel_error_)

        pos_error_hands.append(pos_error_hands_)



    pos_error = sum(pos_error)/len(pos_error)
    vel_error = sum(vel_error)/len(vel_error)
    pos_error_hands = sum(pos_error_hands)/len(pos_error_hands)


    # testing log
    logger.info('Average positional error [cm]: {:<.5f}, Average velocity error [cm/s]: {:<.5f}, Average positional error at hand [cm]: {:<.5f}\n'.format(pos_error*100, vel_error*100, pos_error_hands*100))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='options/test_sixposer.json')
    args = parser.parse_args()
    test(args.json_path)