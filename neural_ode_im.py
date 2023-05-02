import torch 
import pickle
import numpy as np
import torch.nn as nn
from sklearn.datasets import make_circles
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint_adjoint as odeint

# Set the device to GPU since flows are computationally expensive
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Simple Gaussian since is easy to sample and has a known density 
p_z0 = torch.distributions.MultivariateNormal(
    loc=torch.tensor([0.0, 0.0]).to(device),
    covariance_matrix=torch.tensor([[0.2, 0.0], [0.0, 0.2]]).to(device))

def get_batch_targets(num_samples):
    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)
    return(x, logp_diff_t1)

class CircleDataset(Dataset):
    def __init__(self, data):
        self.target = data[0]
        self.log_diff = data[1]
    def __len__(self):
        return len(self.target)
    def __getitem__(self, indx):
        return torch.Tensor(self.target[indx]), torch.Tensor(self.log_diff[indx])
targetloader = DataLoader(CircleDataset(get_batch_targets(12800)), batch_size=64, shuffle=True, pin_memory=True)

class CNF(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]
        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            W, B, U = self.hyper_net(t)
            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)
            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)
            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)
    
    
def trace_df_dz(f, z):
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


class HyperNetwork(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]
    
# Define the training loop
def train(model, optimizer, n_epochs, batch_size):
    best_loss = 1e10
    num_samples = 64
    for epoch in range(n_epochs):
        epoch_loss = 0
        for x, logp_diff_t1 in targetloader:
            optimizer.zero_grad()
            
            z_t, logp_diff_t = odeint(
                model,
                (x, logp_diff_t1),
                torch.tensor([t1, t0]).type(torch.float32).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
            
            logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
            loss = -logp_x.mean(0)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {epoch_loss/len(targetloader)*batch_size}")
        if epoch_loss/len(targetloader)*batch_size < best_loss:
            best_loss = epoch_loss/len(targetloader)*batch_size
            torch.save(model.state_dict(), "best_model_im.pth")

if __name__ == "__mian__":
    # Time grid
    t0, t1 = 0, 10
    # Initialize the model and optimizer
    ode_func = CNF(in_out_dim=2, hidden_dim=32, width=64).to(device)
    optimizer = torch.optim.Adam(ode_func.parameters(), lr=1e-4)
    # Train the model
    train(ode_func, optimizer, n_epochs=100, batch_size=64)