import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np

from dynamics.bnn import *

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


# Functions to implement KL divergence intrinsic reward

# Compute log probability of normal distribution
def _log_prob_normal(input, mu=0., sigma=5.):

    sigma = torch.ones_like(input) * sigma
    # log_normal = - torch.log(sigma) - np.log(np.sqrt(2 * torch.pi)) - torch.square(input - mu) / (2 * torch.square(sigma))
    log_normal = -torch.nn.functional.gaussian_nll_loss(mu, input, sigma)

    return torch.sum(log_normal)


# Compute log marginal likelihood p(D|theta)
def get_log_D_given_theta(x, y, model, lik_sigma=5., n_samples=10):

    loglike_samples = []

    for _ in range(n_samples):
        mu = model.forward(x)[0]

        log_like = _log_prob_normal(y, mu, sigma=lik_sigma)
        loglike_samples.append(log_like.clone().detach().item())

    log_p_D_given_theta = np.mean(loglike_samples)

    return log_p_D_given_theta


# Buffer class for storing single trajectories
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []

        # Rewards
        self.rewards = []
        self.instr_rewards = []

        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]

        # Rewards
        del self.rewards[:]
        del self.instr_rewards[:]

        del self.state_values[:]
        del self.is_terminals[:]


# Actor Critic neural network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor network
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    # method for setting new std dev
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    # method for selecting action
    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


# PPO_VIME class
class PPO_VIME:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, expl_rate=0.1, decay_ex_rate=0.9999, action_std_init=0.6, use_gae=False):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.use_gae = use_gae

        self.buffer = RolloutBuffer()
        self.kl_previous = []

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss(reduction='none')

        # BNN dynamics
        self.expl_rate = expl_rate
        self.decay_ex_rate = decay_ex_rate

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    # Update
    def update(self):

        # # # Monte Carlo estimate of extrinsic state rewards:
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Intrinsic reward
        instr_reward = torch.tensor(self.buffer.instr_rewards, dtype=torch.float32).to(device)

        # Normalizing the intrinsic rewards:
        # instr_reward = (instr_reward - instr_reward.mean()) / (instr_reward.std() + 1e-7)

        # print('shape rewards:' + str(rewards.shape))
        # print('shape intrinsic rewards:' + str(instr_reward.shape))
        #
        # print('mean rewards:' + str(rewards.mean()))
        # print('mean intrinsic rewards:' + str(instr_reward.mean()))

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_states_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # Normalizing state:
        # old_states = (old_states - old_states.mean()) / (old_states.std() + 1e-7)

        # Calculate total rewards
        self.expl_rate = self.expl_rate * self.decay_ex_rate
        combined_rewards = rewards + self.expl_rate * instr_reward

        # Normalizing the combined rewards:
        # combined_rewards = (combined_rewards - combined_rewards.mean()) / (combined_rewards.std() + 1e-7)

        # print('mean rewards: ', rewards.mean().item())
        # print('mean intrinsic rewards: ', instr_reward.mean().item())
        # print('mean combined rewards: ', combined_rewards.mean().item())

        # Advantages
        if self.use_gae:
            # Calculate Generalized Advantage Estimation:
            advantages = torch.zeros_like(combined_rewards)
            for t in range(int(combined_rewards.size(0))):
                for l in range(int(10)):
                    if t + l < len(combined_rewards):
                        advantages[t] += (0.99 * 0.95) ** l * (combined_rewards[t + l] - old_states_values[t])

            # Convert advantages to tensor
            advantages = advantages.detach().to(device)

        else:
            # Advantages = reward to go - state values
            advantages = combined_rewards.detach() - old_states_values.detach()

        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # # Print Advantages associated with states[0] values sorted
        # sort_indx = old_states[:, 0].sort()[1]
        # next_s_intr = torch.cat([old_states[sort_indx],
        #                          instr_reward.clone().detach()[sort_indx].unsqueeze(dim=1)], dim=1)
        # print("next state & advantages: ", next_s_intr)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Match state_values dimensions with rewards dimensions:
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Extrinsic reward loss:
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.float(),
                                                                 combined_rewards.float()) - 0.01 * dist_entropy

            # Intrinsic reward loss:
            loss = loss.mean().float()

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            # # Variational Inference step
            # loss_svi = self.svi.step(state=old_states[:-1], action=old_actions[:-1], next_state=old_states[1:])

            # if _ % 100 == 0:
            #     print("SVI loss : ", loss_svi)

        # # Compute new posterior
        # self.theta_loc.append(self.svi.guide.get_posterior().mean.clone().detach())
        # self.theta_scale.append(self.svi.guide.get_posterior().stddev.clone().detach())
        #
        # # print("theta_loc_new: ", self.theta_loc)
        # # print("theta_scale_new: ", self.theta_scale)

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    # save and load model's parameters
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path + "ppo_vime.pth", )

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path + "ppo_vime.pth", map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
