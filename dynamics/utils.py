import torch
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import scipy.stats as sts

import warnings


def MC_se(x, B):
    return sts.t.ppf(0.9, B - 1) * np.std(np.array(x)) / np.sqrt(B)


class Measure:
    def __call__(self, states, actions, next_states, next_state_means, next_state_vars, model):
        """
        compute utilities of each policy

        Args:
            states: (n_actors, d_state)
            actions: (n_actors, d_action)
            next_state_means: (n_actors, ensemble_size, d_state)
            next_state_vars: (n_actors, ensemble_size, d_state)

        Returns:
            utility: (n_actors)
        """

        raise NotImplementedError


class UtilityMeasure(Measure):
    def __init__(self, action_norm_penalty=0):
        self.action_norm_penalty = action_norm_penalty

    def compute_utility(self, states, actions, next_states, next_state_means, next_state_vars, model):
        raise NotImplementedError

    def __call__(self, states, actions, next_states, next_state_means, next_state_vars, model):
        """
        compute utilities of each policy
        Args:
            states: (n_actors, d_state)
            actions: (n_actors, d_action)
            next_state_means: (n_actors, ensemble_size, d_state)
            next_state_vars: (n_actors, ensemble_size, d_state)

        Returns:
            utility: (n_actors)
        """

        utility = self.compute_utility(states, actions, next_states, next_state_means, next_state_vars, model)

        if not np.allclose(self.action_norm_penalty, 0):
            action_norms = actions ** 2                                            # shape: (n_actors, d_action)
            action_norms = action_norms.sum(dim=1)                                 # shape: (n_actors)
            utility = utility - self.action_norm_penalty * action_norms            # shape: (n_actors)

        if torch.any(torch.isnan(utility)).item():
            warnings.warn("NaN in utilities!")

        if torch.any(torch.isinf(utility)).item():
            warnings.warn("Inf in utilities!")
        return utility


class TransitionNormalizer:
    def __init__(self):
        """
        Maintain moving mean and standard deviation of state, action and state_delta
        for the formulas see: https://www.johndcook.com/blog/standard_deviation/
        """

        self.state_mean = None
        self.state_sk = None
        self.state_stdev = None
        self.action_mean = None
        self.action_sk = None
        self.action_stdev = None
        self.state_delta_mean = None
        self.state_delta_sk = None
        self.state_delta_stdev = None
        self.count = 0

    @staticmethod
    def update_mean(mu_old, addendum, n):
        mu_new = mu_old + (addendum - mu_old) / n
        return mu_new

    @staticmethod
    def update_sk(sk_old, mu_old, mu_new, addendum):
        sk_new = sk_old + (addendum - mu_old) * (addendum - mu_new)
        return sk_new

    def update(self, state, action, state_delta):
        self.count += 1

        if self.count == 1:
            # first element, initialize
            self.state_mean = state.clone()
            self.state_sk = torch.zeros_like(state)
            self.state_stdev = torch.zeros_like(state)
            self.action_mean = action.clone()
            self.action_sk = torch.zeros_like(action)
            self.action_stdev = torch.zeros_like(action)
            self.state_delta_mean = state_delta.clone()
            self.state_delta_sk = torch.zeros_like(state_delta)
            self.state_delta_stdev = torch.zeros_like(state_delta)
            return

        state_mean_old = self.state_mean.clone()
        action_mean_old = self.action_mean.clone()
        state_delta_mean_old = self.state_delta_mean.clone()

        self.state_mean = self.update_mean(self.state_mean, state, self.count)
        self.action_mean = self.update_mean(self.action_mean, action, self.count)
        self.state_delta_mean = self.update_mean(self.state_delta_mean, state_delta, self.count)

        self.state_sk = self.update_sk(self.state_sk, state_mean_old, self.state_mean, state)
        self.action_sk = self.update_sk(self.action_sk, action_mean_old, self.action_mean, action)
        self.state_delta_sk = self.update_sk(self.state_delta_sk, state_delta_mean_old, self.state_delta_mean, state_delta)

        self.state_stdev = torch.sqrt(self.state_sk / self.count)
        self.action_stdev = torch.sqrt(self.action_sk / self.count)
        self.state_delta_stdev = torch.sqrt(self.state_delta_sk / self.count)

    @staticmethod
    def setup_vars(x, mean, stdev):
        assert x.size(-1) == mean.size(-1), f'sizes: {x.size()}, {mean.size()}'

        mean, stdev = mean.clone().detach(), stdev.clone().detach()
        mean, stdev = mean.to(x.device), stdev.to(x.device)

        while len(x.size()) < len(mean.size()):
            mean, stdev = mean.unsqueeze(0), stdev.unsueeze(0)

        return mean, stdev

    def _normalize(self, x, mean, stdev):
        mean, stdev = self.setup_vars(x, mean, stdev)
        n = x - mean
        n = n / stdev
        return n

    def normalize_states(self, states):
        return self._normalize(states, self.state_mean, self.state_stdev)

    def normalize_actions(self, actions):
        return self._normalize(actions, self.action_mean, self.action_stdev)

    def normalize_state_deltas(self, state_deltas):
        return self._normalize(state_deltas, self.state_delta_mean, self.state_delta_stdev)

    def denormalize_state_delta_means(self, state_deltas_means):
        mean, stdev = self.setup_vars(state_deltas_means, self.state_delta_mean, self.state_delta_stdev)
        return state_deltas_means * stdev + mean

    def denormalize_state_delta_vars(self, state_delta_vars):
        mean, stdev = self.setup_vars(state_delta_vars, self.state_delta_mean, self.state_delta_stdev)
        return state_delta_vars * (stdev ** 2)

    def renormalize_state_delta_means(self, state_deltas_means):
        mean, stdev = self.setup_vars(state_deltas_means, self.state_delta_mean, self.state_delta_stdev)
        return (state_deltas_means - mean) / stdev

    def renormalize_state_delta_vars(self, state_delta_vars):
        mean, stdev = self.setup_vars(state_delta_vars, self.state_delta_mean, self.state_delta_stdev)
        return state_delta_vars / (stdev ** 2)

    def get_state(self):
        state = {'state_mean': self.state_mean.clone(),
                 'state_sk': self.state_sk.clone(),
                 'state_stdev': self.state_stdev.clone(),
                 'action_mean': self.action_mean.clone(),
                 'action_sk': self.action_sk.clone(),
                 'action_stdev': self.action_stdev.clone(),
                 'state_delta_mean': self.state_delta_mean.clone(),
                 'state_delta_sk': self.state_delta_sk.clone(),
                 'state_delta_stdev': self.state_delta_stdev.clone(),
                 'count': self.count}
        return state

    def set_state(self, state):
        self.state_mean = state['state_mean'].clone()
        self.state_sk = state['state_sk'].clone()
        self.state_stdev = state['state_stdev'].clone()
        self.action_mean = state['action_mean'].clone()
        self.action_sk = state['action_sk'].clone()
        self.action_stdev = state['action_stdev'].clone()
        self.state_delta_mean = state['state_delta_mean'].clone()
        self.state_delta_sk = state['state_delta_sk'].clone()
        self.state_delta_stdev = state['state_delta_stdev'].clone()
        self.count = state['count']

    def __getstate__(self):
        return self.get_state()

    def __setstate__(self, state):
        self.set_state(state)


class DynamicsBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []

        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]

        del self.is_terminals[:]

    def add(self, state, action, next_state, is_terminal):
        self.actions.append(action)
        self.states.append(state)
        self.next_states.append(next_state)

        self.is_terminals.append(is_terminal)


class JR_Div(UtilityMeasure):
    def __init__(self, decay=0.1, action_norm_penalty=0):
        super().__init__(action_norm_penalty=action_norm_penalty)
        self.decay = decay

    def rescale_var(self, var, min_log_var, max_log_var):
        min_var, max_var = torch.exp(min_log_var), torch.exp(max_log_var)
        return max_var - self.decay * (max_var - var)

    def compute_utility(self, states, actions, next_states, next_state_means, next_state_vars, model):
        state_delta_means = next_state_means - states.to(next_state_means.device).unsqueeze(1)
        state_delta_means = model.normalizer.renormalize_state_delta_means(state_delta_means)
        state_delta_vars = model.normalizer.renormalize_state_delta_vars(next_state_vars)

        mu, var = state_delta_means, next_state_vars                         # shape: both (n_actors, ensemble_size, d_state)
        n_act, es, d_s = mu.size()                                            # shape: (n_actors, ensemble_size, d_state)

        var = self.rescale_var(var, model.min_log_var, model.max_log_var)

        # entropy of the mean
        mu_diff = mu.unsqueeze(1) - mu.unsqueeze(2)                           # shape: (n_actors, ensemble_size, ensemble_size, d_state)
        var_sum = var.unsqueeze(1) + var.unsqueeze(2)                         # shape: (n_actors, ensemble_size, ensemble_size, d_state)

        err = (mu_diff * 1 / var_sum * mu_diff)                               # shape: (n_actors, ensemble_size, ensemble_size, d_state)
        err = torch.sum(err, dim=-1)                                          # shape: (n_actors, ensemble_size, ensemble_size)
        det = torch.sum(torch.log(var_sum), dim=-1)                           # shape: (n_actors, ensemble_size, ensemble_size)

        log_z = -0.5 * (err + det)                                            # shape: (n_actors, ensemble_size, ensemble_size)
        log_z = log_z.reshape(n_act, es * es)                                 # shape: (n_actors, ensemble_size * ensemble_size)
        mx, _ = log_z.max(dim=1, keepdim=True)                                # shape: (n_actors, 1)
        log_z = log_z - mx                                                    # shape: (n_actors, ensemble_size * ensemble_size)
        exp = torch.exp(log_z).mean(dim=1, keepdim=True)                      # shape: (n_actors, 1)
        entropy_mean = -mx - torch.log(exp)                                   # shape: (n_actors, 1)
        entropy_mean = entropy_mean[:, 0]                                     # shape: (n_actors)

        # mean of entropies
        total_entropy = torch.sum(torch.log(var), dim=-1)                     # shape: (n_actors, ensemble_size)
        mean_entropy = total_entropy.mean(dim=1) / 2 + d_s * np.log(2.) / 2    # shape: (n_actors)

        # jensen-renyi divergence
        utility = entropy_mean - mean_entropy                                 # shape: (n_actors)

        return utility


def kl_div_diag(diag_cov1, mean1, diag_cov2, mean2):
    """
    Compute the KL divergence between two multivariate Gaussian distributions with diagonal covariance matrices.

    Parameters:
    diag_cov1 (torch.Tensor): Diagonal covariance matrix of the first Gaussian distribution.
    mean1 (torch.Tensor): Mean of the first Gaussian distribution.
    diag_cov2 (torch.Tensor): Diagonal covariance matrix of the second Gaussian distribution.
    mean2 (torch.Tensor): Mean of the second Gaussian distribution.

    Returns:
    kl_divergence (torch.Tensor): KL divergence value.
    """
    log_cov_ratio = torch.sum(torch.log(diag_cov2 / diag_cov1))
    trace_ratio = torch.sum((diag_cov1 + (mean1 - mean2) ** 2) / diag_cov2)
    term1 = 0.5 * (log_cov_ratio + trace_ratio - len(mean1))

    kl_divergence = term1
    return kl_divergence


def plot_smoothed(ax, y, agent_label, y_label, color, x_label='Time step'):

    mean_y = y.median(dim=0)[0]
    upper95_y = np.percentile(y, 75, axis=0)
    lower95_y = np.percentile(y, 25, axis=0)

    mean_y_smooth = gaussian_filter1d(mean_y, sigma=2)
    upper95_y_smooth = gaussian_filter1d(upper95_y, sigma=2)
    lower95_y_smooth = gaussian_filter1d(lower95_y, sigma=2)

    ax.plot(mean_y_smooth, label=agent_label, color=color)
    ax.fill_between(np.arange(len(mean_y_smooth)), lower95_y_smooth, upper95_y_smooth, alpha=0.3, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # put legend up left
    ax.legend(loc='upper left')


def plot_smoothed_both(ax, y, agent_label, y_label, color, x_label='Time step'):

    mean_y = y.median(dim=0)[0]
    upper95_y = np.percentile(y, 80, axis=0)
    lower95_y = np.percentile(y, 20, axis=0)

    mean_y_smooth = gaussian_filter1d(mean_y, sigma=2)
    upper95_y_smooth = gaussian_filter1d(upper95_y, sigma=2)
    lower95_y_smooth = gaussian_filter1d(lower95_y, sigma=2)

    ax.plot(mean_y_smooth, label=agent_label, color=color)
    ax.fill_between(np.arange(len(mean_y_smooth)), lower95_y_smooth, upper95_y_smooth, alpha=0.3, color=color)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # put legend up left
    ax.legend(loc='upper left')


# # Define EIG estimation function
# def bald_estimation(model, likelihood, test_x):
#     """
#     Estimate the expected information gain (EIG) of a test point.
#
#     Section C.2 paper EPIG derivation of BALD for  GP regression.
#
#     Parameters:
#         model (gpytorch.models.ApproximateGP): Trained GP model.
#         likelihood (gpytorch.likelihoods.Likelihood): Likelihood of the GP model.
#         test_x (torch.Tensor): Test input data (1-D tensor).
#         num_samples (int): Number of samples to use for Monte Carlo estimation.
#
#     Returns:
#         eig (torch.Tensor): EIG of the test point.
#     """
#     # Compute entropy of the predictive distribution
#     f_dist = model(test_x)
#     pred_covar = f_dist.variance.clone().detach()
#
#     # Compute entropy of the joint distribution
#     y_dist = likelihood(f_dist)
#     y_covar = y_dist.variance.clone().detach()
#
#     # Compute EIG
#     bald = 0.5 * (torch.log(pred_covar) - torch.log(y_covar))
#     bald = bald.mean(dim=-1)
#
#     # # Define the desired min and max values
#     # min_value = torch.tensor(-1.0)
#     # max_value = torch.tensor(1.0)
#     #
#     # # max_value = torch.exp(torch.tensor(-.5))
#     #
#     # # # Scale BALD between the specified min and max values
#     # bald = (max_value - min_value) * (bald - bald.min()) / (bald.max() - bald.min()) + min_value
#
#     return bald.clone().detach()


# def bald_estimation(model, likelihood, test_x, n_model_samples=10):
#     """
#     Estimate the expected information gain (EIG) of a test point through BALD.
#     """
#
#     # Compute marginal entropy
#     marg_entr = []
#
#     for i in range(test_x.size(0)):
#
#         if torch.cuda.is_available():
#             test_x_i = test_x[i].unsqueeze(0).cuda()
#             f_dist = model(test_x_i)
#             f_dist = f_dist.cpu()
#
#         else:
#             f_dist = model(test_x[i].unsqueeze(0))
#
#         marg_entr.append(f_dist.entropy())
#
#     marg_entr = torch.stack(marg_entr)
#
#     # Compute conditional entropy
#     f_dist = model(test_x)
#
#     outputs = f_dist.sample(torch.Size([n_model_samples]))  # [K, N, D]
#     outputs = outputs.permute(1, 0, 2)  # [N, K, D]
#
#     y_dist = likelihood(outputs)
#     cond_entr = y_dist.entropy().mean(dim=1)  # [N, D]
#
#     # # If marg_entr is nan, Inf or -Inf, set it to 0
#     # marg_entr = replace_nan_inf(marg_entr)
#     # cond_entr = replace_nan_inf(cond_entr)
#
#     # Rescale marg_entr and cond_entr
#     marg_entr = rescale_var(marg_entr, min_log_var=-5., max_log_var=-1., decay=0.1)
#     cond_entr = rescale_var(cond_entr, min_log_var=-5., max_log_var=-1., decay=0.1)
#
#     # Compute BALD
#     bald = marg_entr - cond_entr  # [N, D]
#
#     # # Define the desired min and max values
#     # min_value = torch.tensor(-1.0)
#     # max_value = torch.tensor(1.0)
#     #
#     # # # Scale BALD between the specified min and max values
#     # bald = (max_value - min_value) * (bald - bald.min()) / (bald.max() - bald.min()) + min_value
#     #
#     # # If bald is nan, Inf or -Inf, set it to 0
#     # bald = replace_nan_inf(bald)
#
#     return bald
#
#
# def eig_gauss(model, likelihood, test_x):
#     """
#     Estimate the expected information gain (EIG) of a test point.
#
#     Section C.2 paper EPIG derivation of BALD for  GP regression.
#
#     Parameters:
#         model (gpytorch.models.ApproximateGP): Trained GP model.
#         likelihood (gpytorch.likelihoods.Likelihood): Likelihood of the GP model.
#         test_x (torch.Tensor): Test input data (1-D tensor).
#
#     Returns:
#         eig (torch.Tensor): EIG of the test point.
#     """
#     # Compute entropy of the predictive distribution
#     f_dist = model(test_x)
#     pred_covar = f_dist.variance.clone().detach()
#
#     y_covar = []
#     # Compute entropy of the joint distribution
#     for j in range(10):
#         # Compute entropy of the joint distribution
#         y_dist = likelihood(f_dist)
#         y_covar.append(y_dist.variance.clone().detach())
#
#     y_covar = torch.stack(y_covar)
#     y_covar = y_covar.mean(dim=0)
#
#     # Compute EIG
#     eig = 0.5 * (torch.log(torch.prod(pred_covar, dim=-1)) - torch.log(torch.prod(y_covar, dim=-1)))
#
#     # Define the desired min and max values
#     min_value = torch.tensor(-1.0)
#     max_value = torch.tensor(1.0)
#
#     # max_value = torch.exp(torch.tensor(-.5))
#
#     # # Scale EIG between the specified min and max values
#     # eig = (max_value - min_value) * (eig - eig.min()) / (eig.max() - eig.min()) + min_value
#
#     return eig.clone().detach()




def eig_gauss(model, likelihood, test_x, n_model_samples=10):
    """
    Estimate the expected information gain (EIG) of a test point through BALD.
    """

    # Compute marginal entropy
    marg_entr = []

    for i in range(test_x.size(0)):

        if torch.cuda.is_available():
            test_x_i = test_x[i].unsqueeze(0).cuda()
            f_dist = model(test_x_i)
            f_dist = f_dist.cpu()

        else:
            f_dist = model(test_x[i].unsqueeze(0))

        marg_entr.append(f_dist.entropy())

    marg_entr = torch.stack(marg_entr)

    # Compute conditional entropy
    f_dist = model(test_x)

    outputs = f_dist.sample(torch.Size([n_model_samples]))  # [K, N, D]
    outputs = outputs.permute(1, 0, 2)  # [N, K, D]

    y_dist = likelihood(outputs)
    cond_entr = y_dist.entropy().mean(dim=1)  # [N, D]

    # # If marg_entr is nan, Inf or -Inf, set it to 0
    # marg_entr = replace_nan_inf(marg_entr)
    # cond_entr = replace_nan_inf(cond_entr)

    # Rescale marg_entr and cond_entr
    marg_entr = rescale_var(marg_entr)
    cond_entr = rescale_var(cond_entr)

    # Compute BALD
    bald = marg_entr - cond_entr  # [N, D]

    # # Define the desired min and max values
    # min_value = torch.tensor(-1.0)
    # max_value = torch.tensor(1.0)
    #
    # # # Scale BALD between the specified min and max values
    # bald = (max_value - min_value) * (bald - bald.min()) / (bald.max() - bald.min()) + min_value
    #
    # # If bald is nan, Inf or -Inf, set it to 0
    # bald = replace_nan_inf(bald)

    return bald





def setup_plot(my_agents):
    # Set the default style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams.update({'font.size': 12})

    # Start two subplots (one for percentage of states visited, one for reward)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].set_title('Reactive', fontweight='bold')
    axs[1].set_title('Active', fontweight='bold')

    # Set y axis limits
    axs[0].set_ylim([-0.05, 1.05])
    # axs[1].set_ylim([-0.05, 1.05])

    # Extract colors from tuple to list
    colors = matplotlib.colormaps['Set3'].colors
    colors = [colors[i] for i in range(len(my_agents))]

    return fig, axs, colors



def setup_plot_both(my_agents):
    # Set the default style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams.update({'font.size': 12})

    # Start two subplots (one for percentage of states visited, one for reward)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].set_title('States Visited (%)', fontweight='bold')
    axs[1].set_title('Rewards', fontweight='bold')

    # Set y axis limits
    axs[0].set_ylim([-0.05, 1.05])

    # Extract colors from tuple to list
    colors = matplotlib.colormaps['Dark2'].colors
    colors = [colors[i] for i in range(len(my_agents))]

    return fig, axs, colors



# Function to replace nan, inf and -inf values with 0
def replace_nan_inf(x):

    # If marg_entr is nan, Inf or -Inf, set it to 0
    x[torch.isnan(x)] = 0
    x[torch.isinf(x)] = 0
    x[torch.isneginf(x)] = 0

    return x


def rescale_var(var, min_log_var=-10., max_log_var=5., decay=1.):

    min_log_var = torch.tensor(min_log_var)
    max_log_var = torch.tensor(max_log_var)

    min_var, max_var = torch.exp(min_log_var), torch.exp(max_log_var)
    return max_var - decay * (max_var - var)
