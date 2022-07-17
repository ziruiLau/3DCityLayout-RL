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


def parse_args():
    """parse input arguments"""
    parser = argparse.ArgumentParser(description='Prim-Agent')

    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('--reference', type=str, default='rgb', help='type of shape reference, rgb or depth image')
    # parser.add_argument('--category', type=str, default='airplane-02691156', help='category name, should be consistent with the name used in file path')
    parser.add_argument('--data_root', type=str, default='./data/', help='root directory of all the data')

    parser.add_argument('--save_net', type=str, default='./save_para/',
                        help='directory to save the network parameters')
    parser.add_argument('--save_net_f', type=int, default=300,
                        help='frequency to save the network parameters (number of episode)')
    parser.add_argument('--save_log', type=str, default='./log/',
                        help='directory to save the training log (tensorboard)')
    parser.add_argument('--save_tmp_result', type=str, default='./tmp/',
                        help='directory to save the temporary results during training')
    parser.add_argument('--load_net', type=str, default='./save_para/',
                        help='directory to load the pre-trained network parameters')
    parser.add_argument('--save_gt_img', type=str, default='./gt_img/')

    args = parser.parse_args()

    return args


def imitation_learning(agent, env, writer):
    episode_count = 0
    for epoch in range(p.DAGGER_EPOCH):

        for name in range(p.DEMO_NUM):
            # corr = 'data/corr/demo/' + str(name) + '.npy'
            # ref = 'data/ref/demo/' + str(name) + '.png'
            for episode in range(p.DAGGER_ITER):

                print('Shape', name, 'Dagger episode', episode)

                env.reset_for_rl()
                s, target = env.load_data(name, 'demo')
                box = copy.copy(env.obs)
                step = env.step_vec
                episode_count += 1
                acm_r = 0

                while True:

                    valid_mask = env.valid_mask()

                    # poll the expert
                    a = env.get_virtual_expert_action_gym(valid_mask)
                    s_ = s
                    box_, step_, r, done = env.step_no_update(a)
                    # expert_action = env._action_to_direction[a]
                    # print('expert:', a,expert_action)

                    agent.memory_long.store(s, box, step, a, r, s_, box_, step_)
                    agent.memory_self.store(s, box, step, a, r, s_, box_, step_)

                    # update the state
                    if episode != 0:
                        a = agent.choose_action(s, box, step, valid_mask, 1.0)
                    # real_action = env._action_to_direction[a]
                    # print('real:', a,  real_action)

                    box_, step_, r, done = env.step(a)

                    acm_r += r

                    if done:
                        # uncomment the following lines to output the intermediate results
                        # log_info = 'RL_' + str(epoch) + '_shape_' + str(1) + '_r_' + str(
                        #     format(acm_r, '.4f')) + '_' + 'shape'
                        # env.output_result('il', save_tmp_result_path)
                        writer.add_scalar('Prim_RL/' + 'grid', acm_r, episode_count)
                        break

                    s = s_
                    box = copy.copy(box_)
                    step = step_

                print('reward:', acm_r)

                for learn in range(p.DAGGER_LEARN):
                    agent.learn(learning_mode=2, is_ddqn=True)

def reinforcement_learning_gym(agent, env, writer):
    episode_count = 0
    for epoch in range(p.RL_EPOCH):
        for name in range(p.TRAIN_NUM):
        # for name in range(2):
            print('Shape:', name, 'RL epoch:', epoch)
            env.reset_for_rl()
            s, target = env.load_data(name, 'train')
            box = copy.copy(env.obs)
            step = env.step_vec
            episode_count += 1
            acm_r = 0

            while True:

                valid_mask = env.valid_mask()
                a = agent.choose_action(s, box, step, valid_mask, p.EPSILON)
                box_, step_, r, done = env.step(a)
                s_ = s
                agent.memory_self.store(s, box, step, a, r, s_, box_, step_)

                acm_r += r

                if agent.memory_self.memory_counter >= p.MEMORY_SELF_CAPACITY:
                    agent.learn(learning_mode=3, is_ddqn=True)

                if done:
                    # log_info='RL_'+str(epoch)+'_shape_'+str(1)+'_r_'+str(format(acm_r, '.4f'))+'_'+'shape'
                    # env.output_result('rl', save_tmp_result_path)
                    writer.add_scalar('Prim_RL/' + 'grid', acm_r, episode_count)
                    break

                s = s_
                box = copy.copy(box_)
                step = step_
            print('reward:', acm_r)
            # if episode_count % save_net_f == 0:
            #     torch.save(agent.eval_net.state_dict(),
            #                save_net_path + 'eval_RL_' + shape_ref_type + '_' + shape_category + '.pth')
            #     torch.save(agent.target_net.state_dict(),
            #                save_net_path + 'target_RL_' + shape_ref_type + '_' + shape_category + '.pth')

if __name__ == '__main__':
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    args = parse_args()

    shape_ref_type = args.reference
    save_net_path = args.save_net
    save_net_f = args.save_net_f
    save_log_path = args.save_log
    save_tmp_result_path = args.save_tmp_result

    utils.check_dirs([save_net_path, save_log_path, save_tmp_result_path, args.save_gt_img])

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # initialize
    gymenv = GridWorldEnv()
    gymenv.reset()
    agent = Agent()
    writer = SummaryWriter(save_log_path)


    print('train il')
    imitation_learning(agent, gymenv, writer)
    torch.save(agent.eval_net.state_dict(), save_net_path+'eval_IL_'+ shape_ref_type + '_layout' + '.pth')
    torch.save(agent.target_net.state_dict(),  save_net_path+'target_IL_'+ shape_ref_type + '_layout' + '.pth')
    agent.memory_self.clear()

    # load_eval_net_path = args.load_net + 'eval_IL_rgb_layout.pth'
    # load_target_net_path = args.load_net + 'target_IL_rgb_layout.pth'
    # agent = Agent([load_eval_net_path, load_target_net_path])

    print('train rl')
    reinforcement_learning_gym(agent, gymenv, writer)
    torch.save(agent.eval_net.state_dict(), save_net_path+'eval_RL_'+ shape_ref_type + '_layout' + '.pth')
    torch.save(agent.target_net.state_dict(),  save_net_path+'target_RL_'+ shape_ref_type + '_layout' + '.pth')