import torch 
from sklearn.datasets import make_circles
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint_adjoint as odeint

# Set the device to GPU since flows are computationally expensive
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# collect target data
targets, _ = make_circles(n_samples=12800, factor=0.4, noise=0.05, random_state=0)

# base distribution 
base_vecs = torch.normal(mean=0, std=0.8, size=(12800, 2)).to(device)

class CircleDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, indx):
        return torch.Tensor(self.data[indx])
targetloader = DataLoader(CircleDataset(targets), batch_size=64, shuffle=True, pin_memory=True)

# Define the ODE function for the time dynamic, i.e., this defines the vector field / differential equation
class ODEFunc(torch.nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 2),
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
    best_loss = 1e10
    for epoch in range(n_epochs):
        epoch_loss = 0
        for targets in targetloader:
            optimizer.zero_grad()
            base_vecs = torch.normal(mean=0, std=0.1, size=(batch_size, 2)).to(device)
            y_pred = odeint(model, base_vecs, t).to(device)
            loss = nll_loss(y_pred[-1], targets.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(targetloader)*batch_size}")
        if epoch_loss/len(targetloader)*batch_size < best_loss:
            best_loss = epoch_loss/len(targetloader)*batch_size
            torch.save(model.state_dict(), "best_model_im.pth")

# Initialize the model and optimizer
ode_func = ODEFunc().to(device)
optimizer = torch.optim.Adam(ode_func.parameters(), lr=1e-3)
# Train the model
train(ode_func, optimizer, n_epochs=100, batch_size=64)