import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from dynamics.utils import DynamicsBuffer, plot_smoothed, MC_se, setup_plot

# Setup agent and directory for logging results
env_name = 'Mountain Car'
my_agents = ['Random', 'PPO',  # Random Expl and π-Entropy
             'PPO_ICM', 'PPO_VIME',  # Reactive ICM and VIME
             'PPO_MAX_1', 'PPO_GP_1', 'PPO_DKL_1',  # 1-step
             'PPO_MAX', 'PPO_GP', 'PPO_DKL']  # H-step
labels = dict(zip(my_agents,
                  ['Random', 'π-Entropy',  # Random Expl and π-Entropy
                   '$\ell^2$ Error', 'VIME',  # Reactive ICM and VIME
                   'BAE DeepEns 5h', 'BAE GP 5h', 'BAE DK 5h',  # 1-step
                   'BAE DeepEns 100h', 'BAE GP 100h', 'BAE DK 100h']  # H-step
                  ))

# Setup plot and colors
fig, axs, colors = setup_plot(my_agents)
fig.suptitle('Mountain Car', fontsize=16, fontweight='bold')

# Create dict and assign keys
percs = dict()
for agent in my_agents:
    percs[agent] = []

sts_rews = ['State', 'Reward']

for sts_rew in sts_rews:

    # For loop over all agents
    for agent, color in zip(my_agents, colors):

        # Get all files in the directory
        dir_path = 'logs/' + sts_rew + '/' + agent
        files = os.listdir(dir_path)

        # Load all files in the dict
        for file in files:
            # Load the file
            percs[agent].append(np.load(dir_path + '/' + file))

        # Stack the list arrays into a tensor of shape (n_runs, n_steps)
        tsr = np.array(percs[agent])
        tsr = torch.tensor(tsr).squeeze(1)


        print('\n\nAgent: {}'.format(agent))
        # Print last percentage of states visited and last reward
        if sts_rew == 'State':
            print('Last percentage of states visited: ', tsr[0, -1])
        else:
            print('Last reward: ', tsr[0, -1])


        # Plot the mean and standard error
        # Plot percentage of states visited and rewards
        if sts_rew == 'State':
            if agent in ['Random', 'PPO',  'PPO_ICM', 'PPO_VIME']:
                plot_smoothed(axs[0], tsr, labels[agent], 'States visited (%)', color=color)
            else:
                plot_smoothed(axs[1], tsr, labels[agent], 'States visited (%)', color=color)

        # Compute mean and MC standard error of the percentage of states visited and write to file
        _mean = tsr[:, -1].mean(dim=0)
        if sts_rew == 'Reward':
            _mean -= 200

        _se = MC_se(tsr[:, -1], 10)

        with open('plots/{}_{}_Runs{}_FINAL.csv'.format(env_name, sts_rew, 10), 'a') as f:
            f.write('{},{},{}\n'.format(agent, _mean, _se))

plt.savefig('plots/{}_Runs{}_FINAL_BALE.pdf'.format(env_name, 10),
            bbox_inches='tight')

