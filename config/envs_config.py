import gym
import os
from datetime import datetime

import torch

from algs.ppo import *
from algs.ppo_icm import *
from algs.ppo_vime import *

from envs.unichain import CustomUniChainEnv

from dynamics.deep_ensemble import *
from dynamics.utils import JR_Div, eig_gauss
from config.dyn_config import compute_intrinsic_reward
from dynamics.gps import predict_model, train_model


class ConfigRun:
    """
    Class to configure env settings and hyperparameters
    """
    def __init__(self, env_name, max_ep_len=400, update_timestep=None, num_states=50):

        # Void function to configure the environment and related parameters
        if env_name == 'MountainCar-v0':

            self.has_continuous_action_space = False

            self.max_ep_len = max_ep_len
            self.max_training_timesteps = int(5e4)

            self.print_freq = self.max_ep_len * 4  # print avg reward in the interval (in num timesteps)
            self.log_freq = self.max_ep_len * 2  # log avg reward in the interval (in num timesteps)
            self.save_model_freq = int(2e4)  # save model frequency (in num timesteps)

            self.action_std = None
            self.action_std_decay_rate = None
            self.action_std_decay_freq = None
            self.min_action_std = None

            # Update every, for active exploration agents
            self.update_every = 25

        elif env_name == 'Unichain':

            self.has_continuous_action_space = False

            self.action_std = None
            self.action_std_decay_rate = None
            self.action_std_decay_freq = None
            self.min_action_std = None

            self.max_ep_len = 100
            self.max_training_timesteps = int(1e4) // 20

            # Update every, for active exploration agents
            self.update_every = 10

        # Else correspond to 'CartPole-v0'
        else:

            self.has_continuous_action_space = False

            self.max_ep_len = max_ep_len
            self.max_training_timesteps = int(2e4)

            self.print_freq = self.max_ep_len * 4  # print avg reward in the interval (in num timesteps)
            self.log_freq = self.max_ep_len * 2  # log avg reward in the interval (in num timesteps)
            self.save_model_freq = int(2e4)  # save model frequency (in num timesteps)

            self.action_std = None
            self.action_std_decay_rate = None
            self.action_std_decay_freq = None
            self.min_action_std = None

            # Update every, for active exploration agents
            self.update_every = 25

        # Configure PPO parameters
        if update_timestep is None:
            self.update_timestep = self.max_ep_len * 2  # update policy every n timesteps
        else:
            self.update_timestep = update_timestep

        self.K_epochs = 80  # update policy for K epochs
        self.eps_clip = 0.2  # clip parameter for PPO
        self.gamma = 0.99  # discount factor

        self.lr_actor = 0.001  # learning rate for actor
        self.lr_critic = 0.001  # learning rate for critic

        # Create Environment
        print("Training environment : {}".format(env_name))

        if env_name == 'Unichain':
            self.env = CustomUniChainEnv(num_states=num_states, max_steps=self.max_ep_len)
        else:
            self.env = gym.make(env_name)
            self.env._max_episode_steps = self.max_ep_len

        # Check if environment is continuous or discrete
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self.state_dim = self.env.observation_space.n
        else:
            self.state_dim = self.env.observation_space.shape[0]

        # action space dimension
        if self.has_continuous_action_space:
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.action_dim = self.env.action_space.n

        # Logging variables
        self.time_step = 0
        self.current_ep_reward = 0

    def create_agent(self, my_agent):

        # Initialize agent
        if my_agent == 'PPO_ICM':

            icm_lr = 0.001  # learning rate for ICM
            icm_eta = 0.2  # intrinsic reward weight
            agent = PPO_ICM(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, self.gamma,
                            self.K_epochs, self.eps_clip, self.has_continuous_action_space, icm_lr, icm_eta)

        # VIME
        elif my_agent == 'PPO_VIME':

            expl_rate = 0.01  # exploration rate
            agent = PPO_VIME(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, self.gamma, self.K_epochs,
                             self.eps_clip, self.has_continuous_action_space, expl_rate=expl_rate, decay_ex_rate=1.,
                             action_std_init=0.6)

        # No intrinsic reward
        else:

            if my_agent in ['PPO_GP', 'PPO_DKL']:
                entropy_bonus = 0.00
            else:
                entropy_bonus = 0.01

            agent = PPO(self.state_dim, self.action_dim, self.lr_actor, self.lr_critic, self.gamma, self.K_epochs,
                        self.eps_clip, self.has_continuous_action_space, self.action_std, entropy_bonus=entropy_bonus)

        return agent

    def update_reactive(self, agent, dyn_model, state, expl_mode):
        """
        Function to update agent + dynamics model
        """

        # select action with policy
        action = agent.select_action(state)
        new_state, reward, done, _ = self.env.step(action)

        # saving reward and is_terminals
        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(done)

        self.time_step += 1
        self.current_ep_reward += reward

        # update PPO agent
        if self.time_step % self.update_timestep == 0:

            if expl_mode == 'PPO_VIME':

                # Get data in the right shape
                sts = torch.stack(agent.buffer.states[:-1]).clone().detach()
                acts = torch.unsqueeze(torch.tensor(agent.buffer.actions[:-1]), dim=1)
                sts_acts = torch.cat([sts, acts], dim=1)

                next_sts = torch.stack(agent.buffer.states[1:]).clone().detach()

                # Compute intrinsic reward
                kl = compute_intrinsic_reward(dynamics=dyn_model, p=0, _inputs=sts_acts, _targets=next_sts,
                                              dim_like=agent.buffer.rewards,
                                              num_steps=sts_acts.shape[0], kl_batch_size=1,
                                              second_order_update=True, n_itr_update=10, use_replay_pool=True)

                # Perform normalization of the intrinsic rewards.
                agent.kl_previous.append(np.median(np.hstack(kl)))
                previous_mean_kl = np.mean(np.asarray(agent.kl_previous))
                kl = kl / previous_mean_kl

                kl[-1] = kl[-2] = kl[-3]

                agent.buffer.instr_rewards = kl.reshape(-1).tolist()

                # # # Update agent
                agent.update()

                # Train BNN
                for _ in range(100):
                    dyn_model.train_fn(sts_acts, next_sts)

            elif expl_mode == 'PPO_ICM' or expl_mode == 'PPO':
                agent.update()

        return new_state, done

    def update_active(self, agent, dyn_model, state, expl_mode, dynam_data, t, update_every=25, warm_start=100,
                      trajs=10, im_h=200):
        """
        Function to actively update agent + dynamics model
        """

        # Allow for warm start of the dynamics model before computing the utility rewards
        if t > warm_start:

            for traj in range(trajs):

                im_state = torch.tensor(state).unsqueeze(0).float().clone().detach()

                for h in range(im_h):

                    # Sample action
                    im_action = agent.select_action(im_state.squeeze(0).float())
                    im_action = torch.tensor(im_action).clone().detach()

                    if 'PPO_MAX' in expl_mode:
                        # get deep ensemble mu and var for the current imagined state, action
                        with torch.no_grad():
                            mu, var = dyn_model.forward_all(im_state, im_action)

                        # Sample only one mu and var, and one state only, for transition
                        next_state = dyn_model.sample(mu, var)

                        # compute the reward r_t = JSD(s_t, a_t, s_t+1)
                        act_reward = JR_Div().compute_utility(im_state, im_action, next_state,
                                                              mu.clone().detach(), var.clone().detach(),
                                                              dyn_model)

                        next_state_smpl = next_state.mean(dim=1).clone().detach()

                    elif expl_mode in ['PPO_GP', 'PPO_DKL', 'PPO_GP_1', 'PPO_DKL_1']:

                        # Normalize state
                        sts_ = dyn_model['model'].normalizer.normalize_states(im_state)
                        acts_ = im_action.unsqueeze(0).unsqueeze(1)
                        sa_test = torch.cat([sts_, acts_], dim=-1).float()

                        if torch.cuda.is_available():
                            sa_test = sa_test.cuda()

                        # get GP mu and var for the current imagined state, action
                        mean, covar, y_covar, f_lower, f_upper = predict_model(dyn_model['model'],
                                                                               dyn_model['likelihood'],
                                                                               sa_test)

                        if torch.cuda.is_available():
                            mean, covar, y_covar, f_lower, f_upper = mean.cpu(), covar.cpu(), y_covar.cpu(), f_lower.cpu(), f_upper.cpu()

                        # Set next state as the mean of the GP
                        mean = dyn_model['model'].normalizer.denormalize_state_delta_means(mean)

                        # Set next state as the mean of the GP
                        next_state_mean = mean + im_state
                        next_state_smpl = next_state_mean.clone().detach()

                    # store the sample in the imaginary ppo buffer
                    if expl_mode in ['PPO', 'PPO_ICM', 'PPO_VIME', 'PPO_MAX', 'PPO_MAX_1']:
                        agent.buffer.rewards.append(act_reward)

                    agent.buffer.is_terminals.append(False)  # Agent cannot imagine terminal states

                    im_state = next_state_smpl.clone().detach()


            # Compute EIG as BALD
            if expl_mode in ['PPO_GP', 'PPO_DKL', 'PPO_GP_1', 'PPO_DKL_1']:
                # Shape the action as a tensor
                acts_ = torch.tensor(agent.buffer.actions).unsqueeze(1)
                sts_ = torch.stack(agent.buffer.states)
                sts_ = dyn_model['model'].normalizer.normalize_states(sts_)

                sa_test = torch.cat([sts_, acts_], dim=-1).float()

                if torch.cuda.is_available():
                    sa_test = sa_test.cuda()

                # Compute act reward as EIG
                act_reward = eig_gauss(dyn_model['model'], dyn_model['likelihood'], sa_test)

                # Store the reward in the imaginary ppo buffer
                agent.buffer.rewards = act_reward.reshape(-1).tolist()



            # print the next state and action corresponding to the highest reward
            print('\n\n\n\n1) IMAGINARY MDP')
            print('Highest act reward: ' + str(max(agent.buffer.rewards)))
            print('Highest Reward State: ' + str(agent.buffer.states[agent.buffer.rewards.index(max(agent.buffer.rewards))]))
            print('Highest Reward Action: ' + str(
                agent.buffer.actions[agent.buffer.rewards.index(max(agent.buffer.rewards))].item()))

            # Update agent
            agent.update()

        # Select action a_t according to the updated agent
        action = agent.select_action(state)
        agent.buffer.clear()

        # Actual Real Transition
        next_state, reward, done, _ = self.env.step(action)

        self.time_step += 1
        self.current_ep_reward += reward

        print('\n2) REAL MDP')
        print('Current state: ' + str(state) + '   |   Current Action: ' + str(action))
        print('Current reward: ' + str(self.current_ep_reward))

        if 'PPO_MAX' in expl_mode:

            # Store the final transition in the dynamics buffer to train model
            dynam_data.add(state, action, next_state - state, done)

            # Update the deep ensemble normalizer with the new samples
            dyn_model.normalizer.update(torch.tensor(state).unsqueeze(0),
                                        torch.tensor(action).unsqueeze(0).unsqueeze(1),
                                        torch.tensor(next_state - state).unsqueeze(0))

            # Train the dynamics model every k steps
            if t % update_every == 0:

                dyn_model.train_model(dynam_data, batch_size=len(dynam_data.states), num_epochs=100)

                # Clear the dynamics buffer after training
                dynam_data.clear()



        elif expl_mode in ['PPO_GP', 'PPO_DKL', 'PPO_GP_1', 'PPO_DKL_1']:

            # Store the final transition in the dynamics buffer to train model
            dynam_data.add(state, action, next_state - state, done)

            # Update the deep ensemble normalizer with the new samples
            dyn_model['model'].normalizer.update(torch.tensor(state).unsqueeze(0),
                                                 torch.tensor(action).unsqueeze(0).unsqueeze(1),
                                                 torch.tensor(next_state - state).unsqueeze(0))

            # Train the dynamics model every k steps
            if t % update_every == 0:

                # Get data in the right shape (normalize state and next state)
                sts = torch.tensor(np.array(dynam_data.states))
                sts = dyn_model['model'].normalizer.normalize_states(sts)
                acts = torch.tensor(np.array(dynam_data.actions)).unsqueeze(1)

                sts_acts = torch.cat([sts, acts], dim=1)

                # normalize next state delta
                next_sts = torch.tensor(np.array(dynam_data.next_states))
                next_sts = dyn_model['model'].normalizer.normalize_state_deltas(next_sts)

                # change type to float
                sts_acts = sts_acts.float()
                next_sts = next_sts.float()

                if torch.cuda.is_available():
                    sts_acts = sts_acts.cuda()
                    next_sts = next_sts.cuda()

                # Train GP
                train_model(dyn_model, sts_acts, next_sts, training_iterations=100)

                # Clear the dynamics buffer after training
                # dynam_data.clear()

        else:
            pass

        return next_state, done


def log_rewards(log_run_reward, log_run_episodes, time_step, i_episode, log_f):
    # log average reward till last episode
    log_avg_reward = log_run_reward / log_run_episodes
    log_avg_reward = round(log_avg_reward, 4)

    log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
    log_f.flush()

    log_run_reward = 0
    log_run_episodes = 0

    return log_run_reward, log_run_episodes


def print_rewards(print_run_reward, print_run_episodes, time_step, i_episode):

    print_avg_reward = print_run_reward / print_run_episodes
    print_avg_reward = round(print_avg_reward, 2)

    print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                            print_avg_reward))

    print_run_reward = 0
    print_run_episodes = 0

    return print_run_reward, print_run_episodes


def save_model(agent, start_time, checkpoint_path):

    print("--------------------------------------------------------------------------------------------")
    print("saving model at : " + checkpoint_path)
    agent.save(checkpoint_path)
    print("model saved")
    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
    print("--------------------------------------------------------------------------------------------")


def print_total_time(start_time):

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


def config_log(my_agent, env_name, run_num):

    # log files for multiple runs are NOT overwritten
    log_dir = my_agent + "_logs"
    log_dir = './logs/' + log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get number of log files in log directory
    current_num_files = next(os.walk(log_dir))[2]
    print("Starting training log number : ", len(current_num_files))

    # Create new log file for each run
    log_f_name = log_dir + my_agent + "_" + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)

    # Check if run_num log file already exists
    directory = my_agent + "_preTrained"
    directory = './preTrained/' + directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    return log_f_name, directory


def set_seeds(rnd_seed):

    torch.manual_seed(rnd_seed)
    np.random.seed(rnd_seed)
    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to", rnd_seed)


def start_timer():

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    return start_time


def start_print():

    # Start printing variables
    print_running_reward = 0
    print_running_episodes = 0

    # Start logging variables
    log_running_reward = 0
    log_running_episodes = 0

    # Start time
    i_episode = 0

    return print_running_reward, print_running_episodes, log_running_reward, log_running_episodes, i_episode
