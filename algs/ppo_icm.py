import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

# set device to cpu or cuda
device = torch.device('cpu')

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


# Function to calculate Generalized Advantage Estimation (GAE)
# See : https://arxiv.org/pdf/1506.02438.pdf
def calc_gae(rewards, values, next_value, gamma=0.99, lamda=0.95):
    advantages = torch.zeros_like(rewards)
    gae = 0
    next_value = next_value.detach()

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lamda * gae
        advantages[t] = gae
        next_value = values[t]

    return advantages


# Buffer class for storing single trajectories
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
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


# implement the ICMModule class referenced in the above class 'PPO_ICM'?
class ICMModule(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, latent_dim=32):
        super(ICMModule, self).__init__()

        self.state_dim = state_dim

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        else:
            self.action_dim = 1

        self.has_continuous_action_space = has_continuous_action_space
        self.action_std_init = action_std_init

        self.forward_model = ForwardModel(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.inverse_model = InverseModel(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)

        # State encoding network
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, state, next_state, action):

        pred_next_state = self.forward_model(state, action)
        pred_action = self.inverse_model(state, next_state)

        state_latent = self.state_encoder(state).float()
        next_state_latent = self.state_encoder(next_state).float()

        return pred_next_state, pred_action, state_latent, next_state_latent


# Forward Model in ICM predicts next state given current state and action
class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, latent_dim=32):
        super(ForwardModel, self).__init__()
        self.state_dim = state_dim

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        else:
            self.action_dim = 1

        self.model = nn.Sequential(
            nn.Linear(latent_dim + self.action_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, latent_dim)
        )

        # State encoding network
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, state, action):
        state = self.state_encoder(state).float()

        act_unsq = torch.unsqueeze(action, dim=1)
        state_action = torch.cat([state, act_unsq], dim=1)

        next_state = self.model(state_action).float()
        return next_state


# Inverse Model in ICM predicts action given current state and next state
class InverseModel(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, latent_dim=32):
        super(InverseModel, self).__init__()
        self.state_dim = state_dim

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        else:
            self.action_dim = 1

        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.action_dim),
            nn.Softmax(dim=1)
        )

        # State encoding network
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, state, next_state):
        state = self.state_encoder(state).float()
        next_state = self.state_encoder(next_state).float()

        state_next_state = torch.cat([state, next_state], dim=1)
        action = self.model(state_next_state).float()
        return action


# Intrinsic Curiosity Module

# can you implement PPO_ICM.py in a similar way? You can use the above code snippet as a reference.
# You can also refer to the original PPO code in the link below:
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py
class PPO_ICM:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, icm_lr, icm_eta, reward_scale=0.01, policy_weight=1.,
                 intrinsic_reward_integration=0.1, action_std_init=0.6, use_gae=False):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.use_gae = use_gae

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.icm_eta = icm_eta
        self.icm = ICMModule(state_dim, action_dim, has_continuous_action_space, action_std_init,
                             latent_dim=32).to(device)
        # self.icm_optimizer = torch.optim.Adam(self.icm.parameters(), lr=icm_lr)

        # Set up optimizers for policy and icm
        self.optimizer_pol = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
        ])

        self.optimizer_forw = torch.optim.Adam(self.icm.forward_model.parameters(), lr=icm_lr)
        self.optimizer_inv = torch.optim.Adam(self.icm.inverse_model.parameters(), lr=icm_lr)

        # Set scaling options
        self.policy_weight = policy_weight

        self.reward_scale = reward_scale
        self.intrinsic_reward_integration = intrinsic_reward_integration

        # Set loss functions
        self.MseLoss_pol = nn.MSELoss()

        self.MseLoss_forw = nn.MSELoss()
        self.MseLoss_inv = nn.MSELoss()  # This is equivalent to Brier Score for binary classification

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

    def icm_loss(self, state, next_state, action):
        # get predicted next state and action
        pred_next_state, pred_action, state_latent, next_state_latent = self.icm(state, next_state, action)

        action = action.unsqueeze(dim=1).float()

        forward_loss = 0.5 * (next_state_latent - pred_next_state).norm(2, dim=-1).pow(2).mean()
        inverse_loss = self.MseLoss_inv(pred_action, action)

        curiosity_loss = self.icm_eta * forward_loss + (1 - self.icm_eta) * inverse_loss

        return curiosity_loss, forward_loss, inverse_loss

    def get_intr_rew(self, state, next_state, action):
        # get predicted next state and action
        pred_next_state, pred_action, state_latent, next_state_latent = self.icm(state, next_state, action)

        # Normalize latent state ?
        # state_latent = F.normalize(state_latent, dim=-1)
        # next_state_latent = F.normalize(next_state_latent, dim=-1)

        # calculate intrinsic reward
        intrinsic_reward = self.reward_scale / 2 * (next_state_latent - pred_next_state).norm(2, dim=-1).pow(2)

        return intrinsic_reward

    def update(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_states_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # Normalizing state:
        # old_states = (old_states - old_states.mean()) / (old_states.std() + 1e-7)

        # Calculate intrinsic rewards
        intrinsic_rewards = self.get_intr_rew(old_states[:-1],
                                              old_states[1:],
                                              old_actions[1:])

        intrinsic_rewards = torch.cat((torch.zeros(1).to(device), intrinsic_rewards), dim=0)
        intrinsic_rewards = intrinsic_rewards.float().detach()

        # Normalizing the intrinsic rewards:
        intrinsic_rewards = (intrinsic_rewards - intrinsic_rewards.mean()) / (intrinsic_rewards.std() + 1e-7)

        # Calculate total combined rewards
        combined_rewards = (1. - self.intrinsic_reward_integration) * rewards + self.intrinsic_reward_integration * intrinsic_rewards

        # # Print intrinsic reward associated with states[0] values sorted
        # sort_indx = old_states[:, 0].sort()[1]
        # next_s_intr = torch.cat([old_states[sort_indx],
        #                          combined_rewards.clone().detach()[sort_indx].unsqueeze(dim=1)], dim=1)
        # print("next state & intrinsic: ", next_s_intr)

        # Compute (Generalized) Advantages
        if self.use_gae:
            # Calculate Generalized Advantage Estimation:
            advantages = torch.zeros_like(combined_rewards)
            for t in range(int(combined_rewards.size(0))):
                for l in range(int(20)):
                    if t + l < len(combined_rewards):
                        advantages[t] += (0.99 * 0.95) ** l * (combined_rewards[t + l] - old_states_values[t])

            # Convert advantages to tensor
            advantages = advantages.detach().to(device)
        else:
            advantages = combined_rewards.detach() - old_states_values.detach()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # # Print Advantages associated with states[0] values sorted
        # sort_indx = old_states[:, 0].sort()[1]
        # next_s_intr = torch.cat([old_states[sort_indx],
        #                          advantages.clone().detach()[sort_indx].unsqueeze(dim=1)], dim=1)
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
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Extrinsic reward loss:
            # Compute curiosity loss
            curiosity_loss, forward_loss, inverse_loss = self.icm_loss(old_states[:-1], old_states[1:], old_actions[1:])

            pol_loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss_pol(state_values, combined_rewards) - 0.01*dist_entropy
            # total_loss = self.policy_weight * pol_loss + curiosity_loss

            # take pol gradient step
            self.optimizer_pol.zero_grad()
            pol_loss.mean().backward()
            self.optimizer_pol.step()

            # take forward gradient step
            self.optimizer_forw.zero_grad()
            forward_loss.mean().backward()
            self.optimizer_forw.step()

            # take inverse gradient step
            self.optimizer_inv.zero_grad()
            inverse_loss.mean().backward()
            self.optimizer_inv.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    # save and load model parameters
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path + '_ppo_icm.pth')
        torch.save(self.icm.state_dict(), checkpoint_path + '_icm.pth')

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path + '_ppo_icm.pth'))
        self.icm.load_state_dict(torch.load(checkpoint_path + '_icm.pth'))


