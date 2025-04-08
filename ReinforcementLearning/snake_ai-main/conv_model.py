import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch
import numpy as np

class Conv(nn.Module):
    def __init__(self, input_channels, num_actions, grid_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        # Calcular el tamaño de salida de las capas convolucionales dinámicamente
        self._conv_output_size = self._get_conv_output(grid_size, input_channels)

        self.fc1 = nn.Linear(self._conv_output_size, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def save(self, file_name='conv_model.pth'):
        if not os.path.exists('./model'):
            os.makedirs('./model')
        file_name = os.path.join('./model', file_name)
        torch.save(self.state_dict(), file_name)
    def _get_conv_output(self, shape, channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, *shape)
            output_feat = self.conv_layers(dummy_input)
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
    # Asegurar que todos son tensores con tipo y shape correctos
        state = self._to_tensor(np.array(state), dim=4)         # (B, C, H, W)
        next_state = self._to_tensor(np.array(next_state), dim=4)
        if isinstance(action,torch.Tensor) and action.numel() > 1:
            action = action[0]
        action = self._to_tensor(np.array(action), dim=1, dtype=torch.long)
        reward = self._to_tensor(np.array(reward), dim=1)
        done = self._to_tensor(np.array(done), dim=1, dtype=torch.bool)

        # Asegurarse de que `action` tenga la forma (batch_size, 1)
        if len(action.shape) == 1:
            action = action.unsqueeze(1)

        # 1. Predicción Q(s, a)
        pred = self.model(state)  # Esto tiene forma (batch_size, num_actions)
        print("Q values:", pred)

        # 2. Q target: r + γ * max(Q(s’, a’))
        with torch.no_grad():
            next_pred = self.model(next_state)
            max_next_q = torch.max(next_pred, dim=1)[0].unsqueeze(1)  # Aseguramos que max_next_q tiene forma (batch_size, 1)
            target = reward.view(-1,1) + self.gamma * max_next_q * (~done)

        # 3. Actualizar solo Q(s, a)
        # Aquí `action` tiene forma (batch_size, 1) y `pred` tiene forma (batch_size, num_actions)
        pred_q = pred.gather(1, action)  # Esto debe funcionar ahora sin errores

        # 4. Backpropagation
        loss = self.criterion(pred_q, target)  # Calculamos la pérdida entre predicción y target
        self.optimizer.zero_grad()  # Limpiamos gradientes
        loss.backward()  # Retropropagación
        self.optimizer.step()  # Actualización de pesos

    def _to_tensor(self, data, dim, dtype=torch.float):
        if isinstance(data, list):
            data = np.array(data)

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).type(dtype)

        elif not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=dtype)

        while data.dim() < dim:
            data = data.unsqueeze(0)

        return data
