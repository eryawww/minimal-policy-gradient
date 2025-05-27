import torch

# OBS_DIM = 2 # Removed

class MLPModule(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64): # Modified
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim) # Modified
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim) # Modified
    
    def forward(self, state) -> torch.Tensor:
        # state = torch.tensor(state, dtype=torch.float32) # (OBS_DIM,)
        x = torch.as_tensor(state, dtype=torch.float32)
        if x.ndim == 0: # Handle scalar input
            x = x.unsqueeze(0)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

class LinearModule(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int): # Modified
        super().__init__()
        # self.t = torch.nn.Parameter(torch.zeros(OBS_DIM, dtype=torch.float32)) # Modified
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, state) -> torch.Tensor:
        # state = torch.tensor(state, dtype=torch.float32) # (OBS_DIM,)
        # return torch.dot(self.t, state) # (1,) # Modified
        x = torch.as_tensor(state, dtype=torch.float32)
        if x.ndim == 0: # Handle scalar input
            x = x.unsqueeze(0)
        return self.linear(x)