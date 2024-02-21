import os
import time
from os.path import join, basename, dirname

import argparse
import shutil
import glob
import wandb


from main_train_avatarposer import train
from main_test_avatarposer import test

def set_wandb(args):
    if args.debug:
        os.environ['WANDB_MODE'] = 'disabled'
        args.name =  args.name + '-test'
    if args.offline:
        os.environ['WANDB_MODE'] = 'offline'

    file_name = basename(os.path.splitext(args.data_path)[0])

    wandb.init(project='EgoMocap', entity='climbingdaily', resume='allow', group=file_name)
    
    cur_data = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))

    wandb.run.name = f'{file_name}-{args.name}-{cur_data}-{wandb.run.id}'

    wandb.config.update(args, allow_val_change=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(e) for e in args.gpu])

    model_dir = join(dirname(__file__), 'output', wandb.run.id)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=False)

    for f in glob.glob(join(dirname(__file__), "*.py")) + glob.glob(
            join(dirname(__file__), "*.yaml")):
        shutil.copy(f, model_dir)

    return wandb.config, model_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', help='train or test or both', )
    parser.add_argument('--train_path', type=str, default='options/train_sixposer.json')
    parser.add_argument('--test_path', type=str, default='options/test_sixposer.json')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--offline', action='store_true', help='offline mode')
    parser.add_argument('--name', type=str, default='', help='name of the run')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='gpu ids')
    parser.add_argument('--data_path', type=str, default='./data_fps60_J6', help='path to data')
    parser.add_argument('--model_path', type=str, 
                        default='results/AvatarPoseEstimation/models/best_63_data_fps60_J6_G.pth', 
                        help='path to models')

    args = parser.parse_args()

    if args.mode == 'train': 
        config, model_dir = set_wandb(args)
        train(args.train_path)
    elif args.mode == 'test':
        test(args.test_path, args.model_path)
    else:
        print('Invalid mode')
