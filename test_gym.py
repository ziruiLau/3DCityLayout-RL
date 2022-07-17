import argparse
import copy

import numpy as np
import torch
import os
from tensorboardX import SummaryWriter
# from environment import Environment
from network import Net, Agent
from memory import Memory
import config as p
from utils import utils
import gym
from grid_world2 import GridWorldEnv


shape_ref_type=''
shape_category=''
shape_ref_path=''
shape_vox_path=''
load_net_path=''
save_result_path=''

def parse_args():

    """parse input arguments"""
    parser = argparse.ArgumentParser(description='Prim-Agent')
    
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('--reference', type=str, default='rgb', help='type of shape reference, rgb or depth image')
    # parser.add_argument('--category', type=str, default='airplane-02691156', help='category name, should be consistent with the name used in file path')
    parser.add_argument('--data_root', type=str, default='./data/', help='root directory of all the data')
    
    parser.add_argument('--load_net', type=str, default='./save_para/', help='directory to load the pre-trained network parameters')
    parser.add_argument('--save_result', type=str, default='./tmp/', help='directory to save the modeling results')

    args = parser.parse_args()
    
    return args

def test(agent, env):
    
    all_reward=[]
    
    for shape_count in range(4):

        env.reset_for_rl()
        s, target = env.load_data(shape_count, 'test')
        box = copy.copy(env.obs)
        step = env.step_vec
        acm_r = 0

        print('Shape:', shape_count)
        while True:
            
            valid_mask = env.valid_mask()
            a = agent.choose_action(s, box, step, valid_mask, 1.0)
            s_ = s
            box_, step_, r, done = env.step(a)
                        
            acm_r+=r
            
            if done:
                all_reward.append(acm_r)
                # log_info=shape_name+'_r_'+str(format(acm_r, '.4f'))
                
                env.output_result('il-test', './tmp/')
                # env.save_edgeloop(save_result_path)
                break
            
            s = s_
            box = box_    
            step = step_
            
    return np.mean(all_reward)
        

if __name__ == "__main__":
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    args = parse_args()
    
    shape_ref_type=args.reference
    shape_ref_path=args.data_root + 'ref/test/'
    shape_vox_path=args.data_root + 'corr/test/'
    
    load_eval_net_path=args.load_net + 'eval_IL_rgb_layout.pth'
    load_target_net_path = args.load_net + 'target_IL_rgb_layout.pth'
    
    save_result_path = args.save_result
    utils.check_dirs([save_result_path])

    #GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    #initialize
    env = GridWorldEnv()
    agent = Agent([load_eval_net_path, load_target_net_path])
    mean_reward = test(agent, env)
    
    print(mean_reward)