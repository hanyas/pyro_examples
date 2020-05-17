import torch
from torch.distributions import Gamma

import matplotlib.pyplot as plt
from tqdm import tqdm

from pyro.distributions import *

import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive

assert pyro.__version__.startswith('1')
pyro.enable_validation(True)
pyro.set_rng_seed(1337)


N, D = 250, 2
data = MultivariateNormal(torch.zeros(D), 1. * torch.eye(D)).sample([N])


def model(data, **kwargs):
    alpha = 2. * torch.ones(D, )
    beta = 2. * torch.ones(D, )

    with pyro.plate("prec_plate", D):
        prec = pyro.sample("prec", Gamma(alpha, beta))

    # with pyro.plate("corr_chol_plate", 1):
    corr_chol = pyro.sample("corr_chol", LKJCorrCholesky(d=D, eta=torch.ones(1)))

    _std = torch.sqrt(1. / prec.squeeze())
    sigma_chol = torch.mm(torch.diag(_std), corr_chol.squeeze())

    mu = pyro.sample("mu", MultivariateNormal(torch.zeros(D), scale_tril=sigma_chol))

    with pyro.plate("data", N, subsample_size=16) as ind:
        pyro.sample("obs", MultivariateNormal(mu, scale_tril=sigma_chol), obs=data[ind])


def guide(data, **kwargs):
    alpha = pyro.param('alpha', lambda: Uniform(1., 2.).sample([D]),  constraint=constraints.positive)
    beta = pyro.param('beta', lambda: Uniform(1., 2.).sample([D]), constraint=constraints.positive)

    psi = pyro.param('psi', lambda: Uniform(1., 2.).sample(), constraint=constraints.positive)
    tau = pyro.param('tau', lambda: MultivariateNormal(torch.zeros(D), torch.eye(D)).sample())

    with pyro.plate("prec_plate", D):
        q_prec = pyro.sample("prec", Gamma(alpha, beta))

    # with pyro.plate("corr_chol_plate", 1):
    q_corr_chol = pyro.sample("corr_chol", LKJCorrCholesky(d=D, eta=psi))

    _q_std = torch.sqrt(1. / q_prec.squeeze())
    q_sigma_chol = torch.mm(torch.diag(_q_std), q_corr_chol.squeeze())

    q_mu = pyro.sample("mu", MultivariateNormal(tau, scale_tril=q_sigma_chol))


optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO(num_particles=10))


def train(num_iterations):
    losses = []
    pyro.clear_param_store()

    # fig = plt.figure(figsize=(5, 5))
    # plt.scatter(data[:, 0], data[:, 1], color="blue", marker="+")

    # center, covar = marginal(guide, num_samples=100)
    # artist = animate(fig.gca(), None, center, covar)

    for j in tqdm(range(num_iterations)):
        loss = svi.step(data)
        losses.append(loss)

        # if (j % 500) == 0:
        #     center, covar = marginal(guide, num_samples=100)
        #     artist = animate(fig.gca(), artist, center, covar)
        #     plt.draw()
        #     plt.axis('equal')
        #     plt.pause(0.001)

    return losses


def marginal(guide, num_samples=25):
    posterior_predictive = Predictive(guide, num_samples=num_samples)
    posterior_samples = posterior_predictive.forward(data)

    mu_mean = posterior_samples['mu'].detach().mean(dim=0).squeeze()
    prec_mean = posterior_samples['prec'].detach().mean(dim=0).squeeze()
    corr_chol_mean = posterior_samples['corr_chol'].detach().mean(dim=0).squeeze()

    _std_mean = torch.sqrt(1. / prec_mean)
    _sigma_chol_mean = torch.mm(torch.diag(_std_mean), corr_chol_mean)
    sigma_mean = torch.mm(_sigma_chol_mean, _sigma_chol_mean.T)

    return mu_mean, sigma_mean


def animate(axes, artist, center, covar):
    from math import pi
    t = torch.arange(0, 2 * pi, 0.01)
    circle = torch.stack([torch.sin(t), torch.cos(t)], dim=0)
    ellipse = torch.mm(torch.cholesky(covar), circle)

    if artist is None:
        point = axes.scatter(center[0], center[1], color="red")
        line = axes.plot(ellipse[0, :] + center[0], ellipse[1, :] + center[1],
                         linestyle='-', linewidth=2, color='g', alpha=1.)[0]
    else:
        line, point = artist
        point.set_offsets(center)
        line.set_xdata(ellipse[0, :] + center[0])
        line.set_ydata(ellipse[1, :] + center[1])

    return tuple([line, point])


elbo = train(5000)

# plt.figure()
# plt.plot(elbo)

import numpy as np

mu, sigma = marginal(guide, num_samples=2500)
print('emp_mu', np.mean(data.numpy(), axis=0))
print('vi_mu', mu)
print('emp_sigma', np.cov(data.numpy(), rowvar=False))
print('vi_sigma', sigma)
