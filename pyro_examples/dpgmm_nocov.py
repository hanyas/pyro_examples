import torch
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
torch.set_num_threads(1)


data = torch.cat((MultivariateNormal(-8 * torch.ones(2), torch.eye(2)).sample([25]),
                  MultivariateNormal(8 * torch.ones(2), torch.eye(2)).sample([25]),
                  MultivariateNormal(torch.tensor([1.5, 2]), torch.eye(2)).sample([25])))

N = data.shape[0]


def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


def model(data, **kwargs):
    with pyro.plate("beta_plate", T - 1):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("mu_plate", T):
        mu = pyro.sample("mu", MultivariateNormal(torch.zeros(2), 10 * torch.eye(2)))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(mix_weights(beta)))
        pyro.sample("obs", MultivariateNormal(mu[z], torch.eye(2)), obs=data)


def guide(data, **kwargs):
    gamma = pyro.param('gamma', alpha * torch.ones(T - 1), constraint=constraints.positive)
    tau = pyro.param('tau', lambda: MultivariateNormal(torch.zeros(2), 10 * torch.eye(2)).sample([T]))
    pi = pyro.param('pi', torch.ones(N, T) / T, constraint=constraints.simplex)

    with pyro.plate("beta_plate", T - 1):
        q_beta = pyro.sample("beta", Beta(torch.ones(T - 1), gamma))

    with pyro.plate("mu_plate", T):
        q_mu = pyro.sample("mu", MultivariateNormal(tau, torch.eye(2)))

    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(pi))


T = 10

optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO(num_particles=10))


def train(num_iterations):
    losses = []
    pyro.clear_param_store()

    fig = plt.figure(figsize=(5, 5))
    plt.scatter(data[:, 0], data[:, 1], color="blue", marker="+")

    centers = marginal(guide, num_samples=100)
    artist = animate(fig.gca(), None, centers)

    for j in tqdm(range(num_iterations)):
        loss = svi.step(data)
        losses.append(loss)

        if (j % 100) == 0:
            centers = marginal(guide, num_samples=100)
            artist = animate(fig.gca(), artist, centers)
            plt.draw()
            plt.axis('equal')
            plt.pause(0.001)

    return losses


def truncate(alpha, centers, weights):
    threshold = alpha**-1 / 100
    true_centers = centers[weights > threshold]
    true_weights = weights[weights > threshold]\
                   / torch.sum(weights[weights > threshold])
    return true_centers, true_weights


def marginal(guide, num_samples=100):
    posterior_predictive = Predictive(guide, num_samples=num_samples)
    posterior_samples = posterior_predictive.forward(data)

    mu_mean = posterior_samples['mu'].detach().mean(dim=0)

    beta_mean = posterior_samples['beta'].detach().mean(dim=0)
    weights_mean = mix_weights(beta_mean)

    centers, _ = truncate(alpha, mu_mean, weights_mean)

    return centers


def animate(axes, artist, centers):
    if artist is not None:
        artist.set_offsets(centers)
    else:
        artist = axes.scatter(centers[:, 0], centers[:, 1], color="red")
    return artist


alpha = 0.1
elbo = train(5000)

plt.figure()
plt.plot(elbo)
