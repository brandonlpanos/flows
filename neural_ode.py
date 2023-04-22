import torch 
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint_adjoint as odeint

'''
Basic Neural ODE model for generating spectra from noise. Paper resource: https://arxiv.org/pdf/1806.07366.pdf
'''

# Set the device to GPU since flows are computationally expensive
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load the targets as a few hundred thousand Mg II flare spectra and rap into a dataloader for easy sampling
# We downsample and normalize to one
with open('spectra.p', 'rb') as f: targets = pickle.load(f)
indices = np.random.choice(targets.shape[0], 2560, replace=False)
targets = targets[indices]
targets = targets / np.max(targets, axis=1).reshape(targets.shape[0],1)

class SpectralData(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, indx):
        return torch.Tensor(self.data[indx])
targetloader = DataLoader(SpectralData(targets), batch_size=64, shuffle=True, pin_memory=True)

# Define the ODE function for the time dynamic, i.e., this defines the vector field / differential equation
class ODEFunc(torch.nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(240, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 240),
        )
    def forward(self, t, y):
        return self.net(y)

# Define the loss function (measures the discrepancy between end products of noise following the field and the targets)
def nll_loss(y_pred, target):
    log_probs = -0.5 * ((y_pred - target) ** 2).sum(dim=1)
    return -log_probs.mean()

# Define the time grid for solving the initial value ODE, 
# Always between 0 and 1, finer means more accurate but longer training
t = torch.linspace(0.0, 1.0, 100).to(device)

# Define the training loop
# Train a continuous flow model to generate spectra from noise
def train(model, optimizer, n_epochs, batch_size):
    for epoch in range(n_epochs):
        epoch_loss = 0
        for targets in targetloader:
            optimizer.zero_grad()
            base_vecs = torch.normal(mean=0, std=0.1, size=(batch_size, 240)).to(device)
            y_pred = odeint(model, base_vecs, t).to(device)
            loss = nll_loss(y_pred[-1], targets.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(targetloader)*batch_size}")

# Generate new samples: samples are newly generated spectra from trained model
# dz_evolution is the evolution of an initial noise vector into the generated spectra
def generate_samples(model, t, n_samples):
    with torch.no_grad():
        dz_evolution = odeint(model, torch.randn(n_samples, 240), t)
        samples = dz_evolution[-1]
    return dz_evolution, samples

# Initialize the model and optimizer
ode_func = ODEFunc().to(device)
optimizer = torch.optim.Adam(ode_func.parameters(), lr=1e-3)
# Train the model
train(ode_func, optimizer, n_epochs=100, batch_size=64)
# Save the model
torch.save(ode_func.state_dict(), "my_model.pth")