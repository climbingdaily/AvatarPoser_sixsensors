import os.path
import math
import argparse
import random
import logging
import time

import torch
import wandb
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

from my_func import save_joint_data

from utils import utils_logger
from utils import utils_option as option
from utils import utils_visualize as vis
from utils.compare_vibe_hmr import mqh_output_metric

from data.select_dataset import define_Dataset
from models.select_model import define_Trainer

save_animation = False
resolution = (800,800)

def train(json_path='options/train_avatarposer.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    opt = option.parse(json_path, is_train=True)
    tag =os.path.basename(opt['datasets']['train']['dataroot'])
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
    if opt['train']['resume']:
        init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type=f'{tag}_G')
        opt['path']['pretrained_netG'] = init_path_G
    else:
        init_iter = 0
    current_step = init_iter

    # save opt to  a '../option.json' file
    option.save(opt)

    # return None for missing key
    opt = option.dict_to_nonedict(opt)

    # configure logger
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Step--2 (creat dataloader)
    dataset_opt = opt['datasets']['train']
    train_set = define_Dataset(dataset_opt)
    train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))  
    logger.info(f'Number of train images: {len(train_set)}, iters: {train_size}')
    train_loader = DataLoader(train_set,
                                batch_size  = dataset_opt['dataloader_batch_size'],
                                shuffle     = dataset_opt['dataloader_shuffle'],
                                num_workers = dataset_opt['dataloader_num_workers'],
                                drop_last   = True,
                                pin_memory  = True)
    
    dataset_opt = opt['datasets']['test']
    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                shuffle=False, num_workers=1, drop_last=False, pin_memory=True)
        
    # Step--3 (initialize model)
    model = define_Trainer(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    # Step--4 (main training)
    
    pre_loss = -1
    total_wandb_log_step = 0
    total_test_step = 0
    start_time = time.time()
    best_mpjpe = 10000000000000000

    for epoch in range(1000):  # keep running
        

        # -------------------------------
        # training
        # -------------------------------
        stats = defaultdict(list)
        bar = tqdm(train_loader)
        bar.set_description(f'Train {epoch:02d}')
        for _, train_data in enumerate(bar):

            current_step += 1
            total_wandb_log_step += 1
            
            model.feed_data(train_data)
            model.optimize_parameters(current_step)     # loss.backward() and optimizer.step()
            model.update_learning_rate(current_step)    # update learning rate

            # ===== update loss =====
            loss_dict = model.current_log()
            pre_loss = loss_dict['total_loss']

            for k, v in loss_dict.items():
                stats[k].append(v)

            loss_dict['step'] = current_step
            wandb.log({'train': loss_dict}, step=total_wandb_log_step)
            desc = f"step: {current_step} | time {time.time() - start_time:.1f} | pre loss {pre_loss:.6f}"

            # -------------------------------
            # merge bnorm
            # -------------------------------
            if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                logger.info('^_^ -----merging bnorm----- ^_^')
                model.merge_bnorm_train()
                model.print_network()

            bar.set_description(desc)

        loss_summary = {'mean_' + k: np.array(v).mean() for k, v in stats.items()}
        loss_summary['epoch'] = epoch
        wandb.log({'train': loss_summary}, step=total_wandb_log_step)
        desc = 'Train: step {:d}, loss {:.2f}'.format(current_step, pre_loss)

        # -------------------------------
        # testing
        # -------------------------------
        stats = defaultdict(list)

        bar = tqdm(test_loader)
        bar.set_description(f'Test {epoch:02d}')
        
        all_pred_joints = []
        all_gt_joints = []

        for index, test_data in enumerate(bar):

            model.feed_data(test_data, test=True)
            model.test()
            total_test_step += 1

            body_parms_pred    = model.current_prediction()
            body_parms_gt      = model.current_gt()
            predicted_angle    = body_parms_pred['pose_body']
            predicted_position = body_parms_pred['position']
            predicted_body     = body_parms_pred['body']

            gt_angle    = body_parms_gt['pose_body']
            gt_position = body_parms_gt['position']
            gt_body     = body_parms_gt['body']

            if index in [0, 10, 20] and save_animation:
                video_dir = os.path.join(opt['path']['images'], str(index))
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir)

                save_video_path_gt = os.path.join(video_dir, 'gt.avi')
                if not os.path.exists(save_video_path_gt):
                    vis.save_animation(body_pose=gt_body, savepath=save_video_path_gt, bm = model.bm, fps=60, resolution = resolution)

                save_video_path = os.path.join(video_dir, '{:d}.avi'.format(current_step))
                vis.save_animation(body_pose=predicted_body, savepath=save_video_path, bm = model.bm, fps=60, resolution = resolution)

            predicted_angle = predicted_angle.reshape(body_parms_pred['pose_body'].shape[0],-1,3)
            gt_angle        = gt_angle.reshape(body_parms_gt['pose_body'].shape[0],-1,3)

            # ==== calculate error ====
            pos_error_       = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1)))
            pos_error_hands_ = torch.mean(torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))[...,[20,21]])

            gt_velocity        = (gt_position[1:,...] - gt_position[:-1,...])*60
            predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*60
            vel_error_         = torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))

            all_pred_joints.append(predicted_position.cpu().numpy())
            all_gt_joints.append(gt_position.cpu().numpy())

            # ==== set wandb logs ====
            loss_dict = {'pos_error': pos_error_, 'vel_error': vel_error_, 'pos_error_hands': pos_error_hands_}
            for k, v in loss_dict.items():
                stats[k].append(v.detach().cpu())
            loss_dict['step'] = total_test_step
            wandb.log({'eval': loss_dict}, step=total_wandb_log_step)

        loss_summary = {'mean_' + k: torch.tensor(v).mean() for k, v in stats.items()}
        matric = mqh_output_metric(np.concatenate(all_pred_joints, axis=0), np.concatenate(all_gt_joints, axis=0))
        loss_summary.update(matric)
        loss_summary['epoch'] = epoch
        desc = 'Evaluate: mpjpe {:.2f} pa_mpjpe {:.2f} pck30 {:.2f} pck50 {:.2f} acc {:.2f} hand_pos [cm] {:.2f}'.format( 
        loss_summary['mpjpe'], 
        loss_summary['pa_mpjpe'], 
        loss_summary['pck_30'] * 100, 
        loss_summary['pck_50'] * 100,
        loss_summary['accel_error'], 
        loss_summary['mean_pos_error_hands'] * 100)
        logger.info(desc)
        wandb.log({'eval': loss_summary}, step=total_wandb_log_step)

        # model.update_learning_rate(current_step)
        # -------------------------------
        # saving model
        # -------------------------------
        if epoch > 10:
            if epoch % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(f'{current_step}_{tag}')

            if loss_summary['mpjpe'] < best_mpjpe:
                best_mpjpe = loss_summary['mpjpe']
                logger.info('Saving the best model.')
                model.save(f'best_{epoch}_{tag}') 
        
    logger.info('Saving the final model.')
    model.save(f'latest_{tag}')
    logger.info('End of training.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='options/train_sixposer.json')
    args = parser.parse_args()
    
    train(args.json_path)
