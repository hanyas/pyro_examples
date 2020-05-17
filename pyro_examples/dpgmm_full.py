import torch
from torch.distributions import Gamma

import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm

from pyro.distributions import *

import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive

assert pyro.__version__.startswith('1')
pyro.enable_validation(True)
pyro.set_rng_seed(1337)

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

data = torch.cat((MultivariateNormal(-2 * torch.ones(2), 0.1 * torch.eye(2)).sample([25]),
                  MultivariateNormal(2 * torch.ones(2), 0.1 * torch.eye(2)).sample([25]),
                  MultivariateNormal(torch.tensor([0., 0.]), 0.1 * torch.eye(2)).sample([25])))

data = data.to(device)

N = data.shape[0]
D = data.shape[1]


def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


def model(data, **kwargs):
    with pyro.plate("beta_plate", T - 1):
        beta = pyro.sample("beta", Beta(1, alpha))

    zeta = 2. * torch.ones(T * D, device=device)
    delta = 2. * torch.ones(T * D, device=device)
    with pyro.plate("prec_plate", T * D):
        prec = pyro.sample("prec", Gamma(zeta, delta))

    corr_chol = torch.zeros(T, D, D, device=device)
    for t in pyro.plate("corr_chol_plate", T):
        corr_chol[t, ...] = pyro.sample("corr_chol_{}".format(t), LKJCorrCholesky(d=D, eta=torch.ones(1, device=device)))

    with pyro.plate("mu_plate", T):
        _std = torch.sqrt(1. / prec.view(-1, D))
        sigma_chol = torch.bmm(torch.diag_embed(_std), corr_chol)
        mu = pyro.sample("mu", MultivariateNormal(torch.zeros(T, D, device=device), scale_tril=sigma_chol))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(mix_weights(beta)))
        pyro.sample("obs", MultivariateNormal(mu[z], scale_tril=sigma_chol[z]), obs=data)


def guide(data, **kwargs):
    gamma = pyro.param('gamma', alpha * torch.ones(T - 1, device=device), constraint=constraints.positive)

    zeta = pyro.param('zeta', lambda: Uniform(1., 2.).sample([T * D]).to(device),  constraint=constraints.positive)
    delta = pyro.param('delta', lambda: Uniform(1., 2.).sample([T * D]).to(device), constraint=constraints.positive)

    psi = pyro.param('psi', lambda: Uniform(1., 2.).sample([T]).to(device), constraint=constraints.positive)

    tau = pyro.param('tau', lambda: MultivariateNormal(torch.zeros(D), 10 * torch.eye(2)).sample([T]).to(device))
    pi = pyro.param('pi', torch.ones(N, T, device=device) / T, constraint=constraints.simplex)

    with pyro.plate("beta_plate", T - 1):
        q_beta = pyro.sample("beta", Beta(torch.ones(T - 1, device=device), gamma))

    with pyro.plate("prec_plate", T * D):
        q_prec = pyro.sample("prec", Gamma(zeta, delta))

    q_corr_chol = torch.zeros(T, D, D, device=device)
    for t in pyro.plate("corr_chol_plate", T):
        q_corr_chol[t, ...] = pyro.sample("corr_chol_{}".format(t), LKJCorrCholesky(d=D, eta=psi[t]))

    with pyro.plate("mu_plate", T):
        _q_std = torch.sqrt(1. / q_prec.view(-1, D))
        q_sigma_chol = torch.bmm(torch.diag_embed(_q_std), q_corr_chol)
        q_mu = pyro.sample("mu", MultivariateNormal(tau, scale_tril=q_sigma_chol))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(pi))


T = 5

optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO(num_particles=35))


def train(num_iterations):
    losses = []
    pyro.clear_param_store()

    # fig = plt.figure(figsize=(5, 5))

    for j in tqdm(range(num_iterations)):
        loss = svi.step(data)
        losses.append(loss)

        # if (j % 100) == 0:
        #     centers, covars = marginal(guide, num_samples=250)
        #     animate(fig.gca(), centers, covars)
        #     plt.draw()
        #     plt.axis('equal')
        #     plt.pause(0.001)
        #     plt.clf()

    return losses


def truncate(alpha, centers, perc, corrs, weights):
    threshold = alpha**-1 / 100.
    true_centers = centers[weights > threshold]

    prec = perc.view(T, D)
    true_prec = prec[weights > threshold]

    true_corrs = corrs[weights > threshold, ...]

    _stds = torch.sqrt(1. / true_prec.view(-1, D))
    _sigmas = torch.bmm(torch.diag_embed(_stds), true_corrs)

    true_sigmas = torch.zeros(len(_sigmas), D, D)
    for n in range(len(_sigmas)):
        true_sigmas[n, ...] = torch.mm(_sigmas[n, ...], _sigmas[n, ...].T)

    true_weights = weights[weights > threshold] / torch.sum(weights[weights > threshold])
    return true_centers, true_sigmas, true_weights


def marginal(guide, num_samples=25):
    posterior_predictive = Predictive(guide, num_samples=num_samples)
    posterior_samples = posterior_predictive.forward(data)

    mu_mean = posterior_samples['mu'].detach().mean(dim=0)
    prec_mean = posterior_samples['prec'].detach().mean(dim=0)

    corr_mean = torch.zeros(T, D, D)
    for t in range(T):
        corr_mean[t, ...] = posterior_samples['corr_chol_{}'.format(t)].detach().mean(dim=0)

    beta_mean = posterior_samples['beta'].detach().mean(dim=0)
    weights_mean = mix_weights(beta_mean)

    centers, sigmas, _ = truncate(alpha, mu_mean, prec_mean, corr_mean, weights_mean)

    return centers, sigmas


def animate(axes, centers, covars):
    plt.scatter(data[:, 0], data[:, 1], color="blue", marker="+")

    from math import pi
    t = torch.arange(0, 2 * pi, 0.01)
    circle = torch.stack([torch.sin(t), torch.cos(t)], dim=0)

    axes.scatter(centers[:, 0], centers[:, 1], color="red")
    for n in range(len(covars)):
        ellipse = torch.mm(torch.cholesky(covars[n, ...]), circle)
        axes.plot(ellipse[0, :] + centers[n, 0], ellipse[1, :] + centers[n, 1],
                  linestyle='-', linewidth=2, color='g', alpha=1.)


alpha = 0.1 * torch.ones(1, device=device)
elbo = train(5000)

# plt.figure()
# plt.plot(elbo)
