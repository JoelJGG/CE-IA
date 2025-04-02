import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch

class ConvNet(nn.Module):
    def __init__(self,height, width, output_size):
        super().__init__()
        self.h = height
        self.w = width
        self.conv1 = nn.Conv2d(3,32, kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32 * (self.h // 2) * (self.w // 2), 256)
        # En el constructor después de las convs:
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 24, 32)  # shape del state
            dummy_output = self.pool(F.relu(self.conv2(F.relu(self.conv1(dummy_input)))))
            flat_size = dummy_output.view(1, -1).size(1)

            self.fc2 = nn.Linear(256, output_size) 

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return  self.fc2(x) 
    def save(self,file_name='model.pth'):
        if not os.path.exists('./model'):
            os.makedirs('./model')
            file_name = os.path.join('./model', file_name)
            torch.save(self.state_dict(),file_name)
"""
class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self,file_name='model.pth'):
        if not os.path.exists('./model'):
            os.makedirs('./model')
            file_name = os.path.join('./model', file_name)
            torch.save(self.state_dict(),file_name)

    """
"""

class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr)
        self.criterion = nn.MSELoss()

    def train_step(self,state,action,reward,next_state,done):
        state = torch.tensor(state,dtype=torch.float)
        next_state = torch.tensor(next_state,dtype=torch.float)
        action = torch.tensor(action,dtype=torch.long)
        reward = torch.tensor(reward,dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done,)

        pred = self.model(state)

        target = pred.clone()

        for index in range(len(done)):
            Q = reward[index]
            if not done[index]:
                Q = reward[index] + self.gamma * torch.max(self.model(next_state[index]))
            target[index][torch.argmax(action[index]).item()] = Q

        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()

        self.optimizer.step()


    """
    
class QTrainer:
    def __init__(self,model,lr,gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(),lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # Convertir listas a tensores directamente
        state = torch.stack([torch.tensor(s, dtype=torch.float32) for s in state])
        next_state = torch.stack([torch.tensor(s, dtype=torch.float32) for s in next_state])
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
    
        if len(state.shape) == 3:
            # Agrega batch dimension si es solo 1 muestra
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done,)
    
        # → predicción con CNN
        pred = self.model(state)
    
        # Copia el resultado actual
        target = pred.clone()
    
        for idx in range(len(done)):
            Q = reward[idx]
            if not done[idx]:
                Q = reward[idx] + self.gamma * torch.max(self.model(next_state[idx].unsqueeze(0)))
            action_idx = torch.argmax(action[idx]).item()
            target[idx][action_idx] = Q
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


