import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch

class Conv(nn.Module):
    def __init__(self, input_channels, num_actions, grid_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        # Calcular el tamaño de salida de las capas convolucionales dinámicamente
        self._conv_output_size = self._get_conv_output(grid_size, input_channels)
        print(f"[DEBUG] Flattened conv output size: {self._conv_output_size}")

        self.fc1 = nn.Linear(self._conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def _get_conv_output(self, shape, channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, *shape)
            output_feat = self.conv_layers(dummy_input)
            print(f"[DEBUG] conv output shape: {output_feat.shape}")
            return output_feat.view(1, -1).size(1)

    def conv_layers(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr)
        self.criterion = nn.MSELoss()

    

    def train_step(self, state, action, reward, next_state, done):
        # Convert if `state` is a list or tuple
        if isinstance(state, (list, tuple)):
            state = torch.stack([
                torch.from_numpy(s).float() if isinstance(s, np.ndarray) 
                else torch.tensor(s, dtype=torch.float32) 
                for s in state
            ])
            next_state = torch.stack([
                torch.from_numpy(s).float() if isinstance(s, np.ndarray) 
                else torch.tensor(s, dtype=torch.float32) 
                for s in next_state
            ])
            action = torch.tensor(action, dtype=torch.long)
            reward = torch.tensor(reward, dtype=torch.float)
            done   = torch.tensor(done,   dtype=torch.bool)
    
        else:
            # If it's a single NumPy array, convert here
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            if isinstance(next_state, np.ndarray):
                next_state = torch.from_numpy(next_state).float()
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).long()
            if isinstance(reward, np.ndarray):
                reward = torch.from_numpy(reward).float()
            if isinstance(done, np.ndarray):
                done = torch.from_numpy(done).bool()
