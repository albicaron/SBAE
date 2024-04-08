from algs import *
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

from config import envs_config as conf
from config import dyn_config as dyn_conf
from config.envs_config import start_timer, set_seeds, config_log, start_print

from dynamics.utils import DynamicsBuffer, plot_smoothed, MC_se, setup_plot

import argparse


def process_arguments():
    parser = argparse.ArgumentParser(description='Example Python Script')

    # Define the command-line arguments
    parser.add_argument('--seed', type=int, required=True, help='Input seed')

    # Parse the arguments
    args = parser.parse_args()

    return args


def train():

    # set seeds
    args = process_arguments()

    random_seed = args.seed
    rnd_seed = (random_seed + 1) ** 2

    # Configure environment settings and hyperparameters
    env_name = 'MountainCar-v0'

    # Measuring just the position of the car
    state_range = [(-1.2, 0.6)]

    # Setup agent and directory for logging results
    my_agents = ['Random', 'PPO',  # Random Expl and π-Entropy
                 'PPO_ICM', 'PPO_VIME',  # Reactive ICM and VIME
                 'PPO_MAX_1', 'PPO_GP_1', 'PPO_DKL_1',  # 1-step
                 'PPO_MAX', 'PPO_GP', 'PPO_DKL']  # H-step

    # Setup plot and colors
    fig, axs, colors = setup_plot(my_agents)

    # Agent loop
    for my_agent, color in zip(my_agents, colors):

        print('\nAgent: ', my_agent)

        # one step ahead or H-step ahead
        if my_agent in ['PPO_MAX_1', 'PPO_GP_1', 'PPO_DKL_1']:
            im_h = 5
        else:
            im_h = 50

        # list to keep track of visited states and rewards per seed
        states_perc_list, reward_list, time_list = [], [], []

        print('\nSeed: ', rnd_seed)

        # Set seeds
        set_seeds(rnd_seed)

        # Restart environment
        conf_env = conf.ConfigRun(env_name, max_ep_len=400, update_timestep=20)

        # Configure logging of percentage of states visited
        percentage_states_visited = torch.zeros(conf_env.max_ep_len)

        reward_tot = 0
        reward_per_step = torch.zeros(conf_env.max_ep_len)

        # Create agent and dynamics model
        agent = conf_env.create_agent(my_agent)
        # Models I want here are: PPO_GP, PPO_DKL, PPO_BALE, PPO_MAX
        if my_agent in ['Random', 'PPO', 'PPO_ICM']:
            dyn_model, dyn_data = None, None
        else:
            dyn_model = dyn_conf.create_dyn_model(conf_env, my_agent, dyn_hidden=32, num_batches=2, dyn_layers=2,
                                                  ensemble_size=10, update_every=conf_env.update_every, num_ind_pts=20)
            dyn_data = None

        # Create buffer for active exploration RL:
        if my_agent not in ['Random', 'PPO', 'PPO_ICM']:
            dyn_data = DynamicsBuffer()
            dyn_model.setup_optimizer() if 'PPO_MAX' in my_agent else print('')

        # training loop
        while conf_env.time_step < conf_env.max_ep_len:

            state = conf_env.env.reset()
            conf_env.current_ep_reward = 0

            # Start max and min position state
            state_max, state_min = state[0], state[0]

            for t in range(1, conf_env.max_ep_len + 1):

                # Update max and min position state
                state_max = max(state[0], state_max)
                state_min = min(state[0], state_min)

                # Update percentage of states visited
                perc_sts = (state_max - state_min) / (state_range[0][1] - state_range[0][0])
                percentage_states_visited[conf_env.time_step] = perc_sts

                # Update reward
                reward_tot += conf_env.current_ep_reward if state[0] <= 0.5 else 100
                reward_per_step[conf_env.time_step] = conf_env.current_ep_reward if state[0] <= 0.5 else 100

                # Episode update
                if my_agent in ['PPO', 'PPO_ICM', 'PPO_VIME']:
                    state, done = conf_env.update_reactive(agent, dyn_model, state, expl_mode=my_agent)
                elif my_agent in ['Random']:
                    # Select random action
                    action = conf_env.env.action_space.sample()
                    state, reward, done, _ = conf_env.env.step(action)
                    conf_env.current_ep_reward += reward
                    print('Reward: ', conf_env.current_ep_reward)
                    conf_env.time_step += 1
                else:
                    state, done = conf_env.update_active(agent, dyn_model, state, expl_mode=my_agent,
                                                         dynam_data=dyn_data,
                                                         update_every=10, t=t, warm_start=50, trajs=10, im_h=im_h)

                if state[0] == 0.6:
                    # Substitute all the remaining states with 0.6 and break loop
                    percentage_states_visited[conf_env.time_step:] = perc_sts
                    reward_per_step[conf_env.time_step:] = 100

                    break

            if state[0] == 0.6:
                break

        # Save percentage of states visited
        states_perc_list.append(percentage_states_visited)
        reward_list.append(reward_per_step)

        # Plot average percentage of states visited and 95% confidence interval
        states_tsr = torch.stack(states_perc_list)
        reward_tsr = torch.stack(reward_list)

        # Save state percentages and rewards as numpy array
        np.save('logs/State/' + my_agent + '/PercStates_' + str(rnd_seed) + '.npy', states_tsr.numpy())
        np.save('logs/Reward/' + my_agent + '/Rewards_' + str(rnd_seed) + '.npy', reward_tsr.numpy())


if __name__ == '__main__':
    train()
