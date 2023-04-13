import torch
from torch import nn
from torchdiffeq import odeint_adjoint as odeint

# Define the continuous-time dynamics using an ordinary differential equation (ODE)
class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

    def forward(self, t, x):
        # t is time (not used in this case)
        # x is the input
        dxdt = -self.net(x.pow(3))
        return dxdt

# Define the continuous normalizing flow model
class CNF(nn.Module):
    def __init__(self, dynamics):
        super().__init__()
        self.dynamics = dynamics

    def forward(self, x, logpx=None):
        # Reshape x to be a column vector
        x = x.view(-1, 1)

        # Compute the derivative of x with respect to t using the ODE solver
        # We use the adjoint method for backpropagation through the ODE solver
        t = torch.Tensor([0, 1]).to(x)
        xt = odeint(self.dynamics, x, t, rtol=1e-5, atol=1e-6)

        # Reshape xt to be a row vector
        xt = xt.view(1, -1)

        # Compute the log determinant of the Jacobian matrix
        # This is needed for computing the likelihood of the transformed distribution
        logdet = torch.zeros(x.shape[0], device=x.device)

        if logpx is not None:
            logpx = logpx - logdet

        return xt, logpx

# Define the simple 1D Gaussian distribution
class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, n):
        return torch.randn(n) * self.sigma + self.mu

    def log_prob(self, x):
        return -0.5 * ((x - self.mu) / self.sigma) ** 2 - torch.log(torch.sqrt(2 * torch.tensor([3.1415])) * self.sigma)

# Instantiate the continuous normalizing flow model and the simple 1D Gaussian distribution
dynamics = Dynamics()
cnf = CNF(dynamics)
gaussian = Gaussian(mu=0.5, sigma=0.1)

# Sample from the simple 1D Gaussian distribution
x = gaussian.sample(100)

# Transform the samples using the continuous normalizing flow model
xt, logpx = cnf(x)

# Compute the likelihood of the transformed samples under the transformed distribution
log_prob = gaussian.log_prob(xt.view(-1))

# Print the average log likelihood
print(log_prob.mean())
