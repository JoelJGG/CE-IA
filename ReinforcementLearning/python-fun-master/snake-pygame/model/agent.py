from collections import deque
import random
from snake_game import Point, Direction, SnakeGame
from model import ConvNet, QTrainer
#from model import Network
import numpy as np
import torch

LR = 0.001

class Agent:
    def __init__(self):
        self.game_counter = 0
        self.gamma = 1 - LR
        #self.model = Network(17,1024,3)
        self.model = ConvNet(height=24,width=32,output_size=3)

        self.trainer = QTrainer(self.model, LR, self.gamma)
        self.memory = deque(maxlen=100000)

    """
    def state(self,game):
        head = game.snake[0]
        point_l = Point(head.x - game.BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - game.BLOCK_SIZE)
        point_r = Point(head.x + game.BLOCK_SIZE, head.y)
        point_d = Point(head.x, head.y + game.BLOCK_SIZE)

        dir_l =  game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        die_l = (dir_u and game.is_collision(point_l)) or \
                (dir_l and game.is_collision(point_d)) or \
                (dir_d and game.is_collision(point_r)) or \
                (dir_r and game.is_collision(point_u)) 

        die_s = (dir_u and game.is_collision(point_u)) or \
                (dir_l and game.is_collision(point_l)) or \
                (dir_d and game.is_collision(point_d)) or \
                (dir_r and game.is_collision(point_r))

        die_r = (dir_u and game.is_collision(point_r)) or \
                (dir_l and game.is_collision(point_u)) or \
                (dir_d and game.is_collision(point_l)) or \
                (dir_r and game.is_collision(point_d)) 

        apple_l = game.food.x < head.x
        apple_r = game.food.x > head.x
        apple_u = game.food.y < head.y
        apple_d = game.food.y > head.y

        positions = game.snake_position()
        snake_pos = np.array([(int(p.x),int(p.y)) for p in positions], dtype=int).flatten()
        print(snake_pos)
        print(type(snake_pos))
        state = [
            *snake_pos,
            dir_l,
            dir_u,
            dir_r,
            dir_d,
            die_l,
            die_s,
            die_r,
            apple_l,
            apple_u,
            apple_r,
            apple_d
        ]

        return np.array(state, dtype=int)
    """
    def state(self,game):
        grid_width = game.w // game.BLOCK_SIZE
        grid_height = game.h // game.BLOCK_SIZE

        state = np.zeros((3,grid_height,grid_width),dtype=np.float32)

        for segment in game.snake:
            x = int(segment.x // game.BLOCK_SIZE)
            y = int(segment.y // game.BLOCK_SIZE)
            if 0 <= y < grid_height and 0 <= x < grid_width:
                state[0, y, x] = 1.0

        fx = int(game.food.x // game.BLOCK_SIZE)
        fy = int(game.food.y // game.BLOCK_SIZE)

        if 0 <= fy < grid_height and 0 <= fx < grid_width:
            state[1, fy, fx] = 1.0

        head = game.snake[0]
        hx = int(head.x // game.BLOCK_SIZE)
        hy = int(head.y // game.BLOCK_SIZE)
        if 0 <= hy < grid_height and 0 <= hx < grid_width:
            state[2, hy, hx] = 1.0

        return state
    def action(self,state):
        move = [0, 0, 0]
        epsilon = max(0.01, 0.9 * (0.995 ** self.game_counter))
        
        if random.randint(0, 200) < epsilon:
            index = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            pred = self.model(state_tensor)
            index = torch.argmax(pred[0]).item()

        if index >= len(move):
            print(f"⚠️ índice inválido: {index} → pred.shape = {pred.shape}")
            index = 0
        move[index] = 1
        return move

    def train(self,state,action,reward,new_state,done):
        self.trainer.train_step(state,action,reward,new_state,done)

    def save_step(self,state,action,reward,new_state,done):
        self.memory.append((state,action,reward,new_state,done))

    def train_long(self):
        if len(self.memory) > 1000:
            sample = random.sample(self.memory,1000)
        else:
            sample = self.memory
        states,actions,rewards,new_states,dones = zip(*sample)
        self.trainer.train_step(states,actions,rewards,new_states,dones)


def play():
    max_score = 0 
    agent = Agent()
    game = SnakeGame()
    while True:
        state = agent.state(game)
        action = agent.action(state)
        reward, done, score = game.play_step(action)
        new_state = agent.state(game)

        agent.train(state,action,reward,new_state,done)
        agent.save_step(state,action,reward,new_state,done)
        
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
