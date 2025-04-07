"""
Giving him info about his body
Using a conv instead of a Linear
Tweaking the Reward function
"""
from collections import deque
import random
from conv_snake_game import Point, Direction, SnakeGame
from conv_model import Conv, QTrainer
import numpy as np
import torch

LR = 0.001

class Agent:
    def __init__(self):
        self.game_counter = 0
        self.gamma = 1 - LR
        self.model = Conv(input_channels=6, grid_size=(24, 32), num_actions=3)
        self.trainer = QTrainer(self.model, LR, self.gamma)
        self.memory = deque(maxlen=100000)
    
    def state(self, game):

        return game.get_conv_state()
    
    def action(self, state):
        move = [0, 0, 0]
        epsilon = 100 - self.game_counter
        if random.randint(0, 200) < epsilon:
            index = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            print("State shape:", state.shape)
            print("Tensor shape:", state_tensor.shape)
            pred = self.model(state_tensor)
            index = torch.argmax(pred).item()
        move[index] = 1
        return move
    
    def train(self, state, action, reward, new_state, done):
        self.trainer.train_step(state, action, reward, new_state, done)

    def save_step(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def train_long(self):
        if len(self.memory) > 1000:
            sample = random.sample(self.memory, 1000)
        else:
            sample = self.memory
        states, actions, rewards, new_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, new_states, dones)

    
def play():
    total_score = 0
    max_score = 0
    agent = Agent()
    game = SnakeGame()
    while True:
        state = agent.state(game)
        action = agent.action(state)
        reward, done, score = game.play_step(action)
        new_state = agent.state(game)

        agent.train(state, action, reward, new_state, done)
        agent.save_step(state, action, reward, new_state, done)

        if done:
            game.reset()
            agent.game_counter += 1
            agent.train_long()

            if score > max_score:
                max_score = score
                agent.model.save()

            print('Game: ', agent.game_counter, 'Score: ', score, 'Max Score: ', max_score)

if __name__ == '__main__':
    play()
