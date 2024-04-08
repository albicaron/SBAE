import torch
import gpytorch
import numpy as np

from algs.ppo_vime import *
from dynamics.deep_ensemble import *
from dynamics.utils import DynamicsBuffer

from dynamics.gps import IndependentMultitaskSVGPModel, IndependentMultitaskSVDKLModel, DUE_model
from dynamics.gps import setup_model, train_model, predict_model


def create_dyn_model(conf_env, my_agent, dyn_hidden=128, dyn_layers=3, latent_dim=10, ensemble_size=25,
                     num_batches=32, update_every=100, num_ind_pts=50):

    # Initialize dynamics model
    global dyn_model

    # Deep Ensemble dynamics
    if conf_env.has_continuous_action_space:
        action_dim_dyn = conf_env.action_dim
    else:
        action_dim_dyn = 1

    if my_agent == 'PPO_VIME':

        # BNN dynamics
        dyn_model = BNN(n_in=int(conf_env.state_dim + action_dim_dyn), n_hidden=[dyn_hidden]*dyn_layers,
                        n_out=conf_env.state_dim,
                        n_batches=num_batches, learning_rate=0.0001)

    elif 'PPO_MAX' in my_agent:

        # Deep Ensemble dynamics
        dyn_model = DeepEnsemble(action_dim_dyn, conf_env.state_dim, n_hidden=dyn_hidden, n_layers=dyn_layers,
                                 ensemble_size=ensemble_size, non_linearity='leaky_relu')

    elif my_agent in ['PPO_GP', 'PPO_DKL', 'PPO_GP_1', 'PPO_DKL_1']:

        # GP dynamics
        if 'PPO_GP' in my_agent:

            dyn_model = IndependentMultitaskSVGPModel(input_dim=conf_env.state_dim + action_dim_dyn,
                                                      output_dim=conf_env.state_dim,
                                                      num_ind_pts=num_ind_pts, mean_mode='linear')

            likelihood, objective_function, optimizer = setup_model(dyn_model, conf_env.state_dim, update_every,
                                                                    learning_rate=0.01, mode='SVGP ELBO')

            if torch.cuda.is_available():
                dyn_model = dyn_model.cuda()
                likelihood = likelihood.cuda()

        else:

            # hidden_dims = [dyn_hidden] * (dyn_layers - 1)
            # hidden_dims.append(2)
            # hidden_dims = [50, 10]
            #
            # dyn_model = IndependentMultitaskSVDKLModel(input_dim=conf_env.state_dim + action_dim_dyn,
            #                                            output_dim=conf_env.state_dim, num_ind_pts=num_ind_pts,
            #                                            hidden_units=hidden_dims, mean_mode='linear')

            dyn_model = DUE_model(input_dim=conf_env.state_dim + action_dim_dyn, output_dim=conf_env.state_dim,
                                  num_ind_pts=num_ind_pts, num_units=32, depth=2, mean_mode='constant')

            likelihood, objective_function, optimizer = setup_model(dyn_model, conf_env.state_dim,
                                                                    update_every, learning_rate=0.01, mode='SVDKL Bi-L')

            if torch.cuda.is_available():
                dyn_model = dyn_model.cuda()
                likelihood = likelihood.cuda()

        # return model dict
        dyn_model = dict([("model", dyn_model),
                          ("likelihood", likelihood),
                          ("objective_function", objective_function),
                          ("optimizer", optimizer)])

    else:
        dyn_model = None

    return dyn_model


def compute_intrinsic_reward(dynamics, p, _inputs, _targets, dim_like, num_steps, kl_batch_size, second_order_update,
                             n_itr_update, use_replay_pool):

    kl = np.zeros((len(dim_like), 1))

    for k in range(p * num_steps,
                   int((p * num_steps) + np.ceil(num_steps / float(kl_batch_size)))):

        # Save old params for every update.
        dynamics.save_old_params()
        start = k * kl_batch_size
        end = np.minimum(
            (k + 1) * kl_batch_size, _targets.shape[0] - 1)

        if second_order_update:
            # We do a line search over the best step sizes using
            # step_size * invH * grad
            #                 best_loss_value = np.inf
            for step_size in [0.01]:
                dynamics.save_old_params()
                loss_value = dynamics.train_update_fn(
                    _inputs[start:end], _targets[start:end], second_order_update, step_size)
                loss_value = loss_value.detach()
                kl_div = np.clip(loss_value, 0, 1000)
                # If using replay pool, undo updates.
                if use_replay_pool:
                    dynamics.reset_to_old_params()
        else:
            # Update model weights based on current minibatch.
            for _ in range(n_itr_update):
                dynamics.train_update_fn(
                    _inputs[start:end], _targets[start:end], second_order_update)
            # Calculate current minibatch KL.
            kl_div = np.clip(
                float(dynamics.f_kl_div_closed_form().detach()), 0, 1000)

        for k in range(start, end):
            index = k % num_steps
            kl[index][p] = kl_div

        # If using replay pool, undo updates.
        if use_replay_pool:
            dynamics.reset_to_old_params()

    return kl
