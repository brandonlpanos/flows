# Experiments with continuous normalizing flows (CNF)

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq # library for ODE solvers

# The base distribution is a standard normal distribution in 2D
# The target distribution is a mixture of 2D Gaussian's with different means and variances

# Define the first order differential equation used to transform the input distribution over time
class ODEfunc(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ODEfunc, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Define the layers to transform x1 into x2
        self.layers = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim // 2),
            nn.Tanh(),
        )

    def forward(self, t, x):
        # Split x into two parts: x1 and x2
        x1, x2 = x.chunk(2, dim=-1)
        
        # Transform x2 using the layers defined above
        x2_transformed = x2 * torch.exp(self.layers(x1))
        
        # Compute the derivatives of x1 and x2 with respect to time t
        dx1_dt = x2_transformed
        dx2_dt = -x1
        
        # Concatenate the derivatives of x1 and x2 into a single tensor and return it
        return torch.cat((dx1_dt, dx2_dt), dim=-1)
    
# Define the continuous normalizing flow model that uses the ODE function defined above to transform the input distribution
class ContinuousNormalizingFlow(nn.Module):
    def __init__(self, num_steps, input_dim, hidden_dim):
        super(ContinuousNormalizingFlow, self).__init__()
        self.num_steps = num_steps
        # Create an instance of the ODE function
        self.func = ODEfunc(input_dim, hidden_dim)

    def forward(self, x):
        # Define the start and end times for the ODE solver
        t0 = torch.tensor([0.0])
        t1 = torch.tensor([1.0])
        
        # Set x0 to be the input distribution
        x0 = x
        
        # Solve the ODE using the torchdiffeq library and return the final distribution and the log determinant of the Jacobian matrix
        xs = torchdiffeq.odeint(self.func, x0, torch.linspace(t0, t1, self.num_steps))
        log_det = torch.sum(torch.log(torch.abs(torch.autograd.functional.jacobian(xs[-1], x0))))
        return xs[-1], log_det
    
# Define the target distribution that we want to approximate
def target_density(x):
    return (torch.exp(-0.5 * ((x[:, 0] - 2) ** 2 + (x[:, 1] - 2) ** 2) / 0.2 ** 2) +
            torch.exp(-0.5 * ((x[:, 0] + 2) ** 2 + (x[:, 1] + 2) ** 2) / 0.2 ** 2) +
            torch.exp(-0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2) / 0.2 ** 2))

# Define the prior distribution that we will start with (a standard normal distribution)
prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

# How do I now train the above flow?
