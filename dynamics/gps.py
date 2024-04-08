import math
import torch
import gpytorch
import matplotlib.pyplot as plt

from dynamics.utils import TransitionNormalizer
import torch.nn as nn
import torch.nn.functional as F

from dynamics.layers.spectral_norm_fc import spectral_norm_fc


# Indipendent Multioutput SVGP (no correlations)
class IndependentMultitaskSVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim, output_dim, num_ind_pts, mean_mode='constant'):
        # Let's use a different set of inducing points for each task
        inducing_points = torch.rand(output_dim, num_ind_pts, input_dim)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([output_dim])
        )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=output_dim
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        # Tried also linear mean function, but it does not work really well on unichain
        if mean_mode == 'linear':
            self.mean_module = gpytorch.means.LinearMean(input_size=input_dim,
                                                         batch_shape=torch.Size([output_dim]))
        else:
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([output_dim]))

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim, batch_shape=torch.Size([output_dim])),
            batch_shape=torch.Size([output_dim]), num_dims=input_dim
        )

        # Set transition normalizer
        self.normalizer = TransitionNormalizer()

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# NN part of the DKL model
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, input_size, hidden_units):
        super().__init__()

        all_units = [input_size] + hidden_units

        for i in range(1, len(all_units)):
            self.add_module(f"linear_{i}", torch.nn.Linear(all_units[i - 1], all_units[i]))
            if i < len(all_units) - 1:
                self.add_module(f"relu_{i}", torch.nn.ReLU())




# We exclude DKL model as it does not perform well and use SVDKL instead
class IndependentMultitaskSVDKLModel(gpytorch.Module):
    def __init__(self, input_dim, output_dim, num_ind_pts=10, hidden_units=[20, 5], mean_mode='constant'):
        super(IndependentMultitaskSVDKLModel, self).__init__()

        # Feature extractor
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
        self.feature_extractor = LargeFeatureExtractor(input_dim, hidden_units)

        self.gp_layer = IndependentMultitaskSVGPModel(hidden_units[-1], output_dim, num_ind_pts,
                                                      mean_mode=mean_mode)

        # Set transition normalizer
        self.normalizer = TransitionNormalizer()

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch

        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        # Concatenate action
        res = self.gp_layer(projected_x)

        return res


class FCResNet(nn.Module):
    def __init__(
        self,
        input_dim,
        features,
        depth,
        spectral_normalization,
        coeff=0.95,
        n_power_iterations=1,
        dropout_rate=0.01,
        num_outputs=None,
        activation="relu",
    ):
        super().__init__()
        """
        ResFNN architecture

        Introduced in SNGP: https://arxiv.org/abs/2006.10108
        """
        self.first = nn.Linear(input_dim, features)
        self.residuals = nn.ModuleList(
            [nn.Linear(features, features) for i in range(depth)]
        )
        self.dropout = nn.Dropout(dropout_rate)

        if spectral_normalization:
            self.first = spectral_norm_fc(
                self.first, coeff=coeff, n_power_iterations=n_power_iterations
            )

            for i in range(len(self.residuals)):
                self.residuals[i] = spectral_norm_fc(
                    self.residuals[i],
                    coeff=coeff,
                    n_power_iterations=n_power_iterations,
                )

        self.num_outputs = num_outputs
        if num_outputs is not None:
            self.last = nn.Linear(features, num_outputs)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        else:
            raise ValueError("That acivation is unknown")

    def forward(self, x):
        x = self.first(x)

        for residual in self.residuals:
            x = x + self.dropout(self.activation(residual(x)))

        if self.num_outputs is not None:
            x = self.last(x)

        return x



class DUE_model(gpytorch.Module):
    def __init__(self, input_dim, output_dim, num_ind_pts=10, num_units=20, depth=2, mean_mode='constant'):

        super(DUE_model, self).__init__()

        # Feature extractor
        self.feature_extractor = FCResNet(
                                            input_dim=input_dim,
                                            features=num_units,
                                            depth=depth,
                                            spectral_normalization=True,
                                            coeff=0.95,
                                            n_power_iterations=1,
                                            dropout_rate=0.01
                                        )

        # Define GP layer
        self.gp_layer = IndependentMultitaskSVGPModel(num_units, output_dim, num_ind_pts,
                                                      mean_mode=mean_mode)

        # Set transition normalizer
        self.normalizer = TransitionNormalizer()

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch

        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        res = self.gp_layer(projected_x)

        return res







# Setup likelihood, objective function, and optimizer
def setup_model(gp_model, state_dim, data_dim, learning_rate=0.1, mode='SVGP ELBO'):

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=state_dim)

    if mode == 'SVGP ELBO':
        objective_function = gpytorch.mlls.VariationalELBO(likelihood, gp_model, num_data=data_dim)
    elif mode == 'SVGP PredLL':
        objective_function = gpytorch.mlls.PredictiveLogLikelihood(likelihood, gp_model, num_data=data_dim)
    elif mode in ['SVDKL ELBO', 'SVDKL Bi-L']:
        objective_function = gpytorch.mlls.VariationalELBO(likelihood, gp_model.gp_layer, num_data=data_dim)
    elif mode == 'SVDKL PredLL':
        objective_function = gpytorch.mlls.PredictiveLogLikelihood(likelihood, gp_model.gp_layer, num_data=data_dim)
    else:
        raise ValueError('Mode not recognized')

    if mode == 'Exact GP':
        optimizer = torch.optim.Adam(gp_model.parameters(), lr=learning_rate)
    else:
        try:
            optimizer = torch.optim.Adam(list(gp_model.parameters()) + list(gp_model.feature_extractor()) + list(likelihood.parameters()),
                                         lr=learning_rate)
        except:
            optimizer = torch.optim.Adam(list(gp_model.parameters()) + list(likelihood.parameters()), lr=learning_rate)

    return likelihood, objective_function, optimizer


# Train model
def train_model(gp_model, state_action, next_state, training_iterations=50):
    # Train
    gp_model['model'].train()
    gp_model['likelihood'].train()
    for _ in range(training_iterations):
        output = gp_model['model'](state_action)
        loss = -gp_model['objective_function'](output, next_state)
        loss.backward()
        gp_model['optimizer'].step()
        gp_model['optimizer'].zero_grad()



# Predict next state
def predict_model(gp_model, likelihood, state_action):
    gp_model.eval()
    likelihood.eval()
    with torch.no_grad():

        f_dist = gp_model(state_action)
        y_dist = likelihood(f_dist)
        y_covar = y_dist.variance

        mean = f_dist.mean
        covar = f_dist.variance
        # covar_ = covar.view(state_action.size(0), state_dim, state_action.size(0), state_dim)

        f_lower, f_upper = f_dist.confidence_region()

        # y_dist = likelihood(f_dist)
        # y_lower, y_upper = y_dist.confidence_region()

    return mean, covar, y_covar, f_lower, f_upper

