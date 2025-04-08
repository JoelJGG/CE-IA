"""
Mejorar la reward
1- Si se acerca a la manzana libremente subir
2- Si se acerca a su cuerpo sin sentido restar 0.1
"""
import math
import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN1 = (0, 255, 0)
GREEN2 = (0, 255, 100)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 200

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        print("Iniciando")
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.BLOCK_SIZE = BLOCK_SIZE
        self.reset()
        
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.steps = 0
        self.food = None
        self.distance_to_food = 0
        self._place_food()
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def distance_to_food(self):
        distance = self.head.distance(self.food)
        print(distance)
        return distance 
    def play_step(self, action):
        reward = 0
        self.steps += 1
        distance = self.distance_to_food
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        new_distance = self.distance_to_food
        if new_distance < distance:
            reward += 0.1
        distance = new_distance
        # 3. check if game over
        game_over = False
        if self.is_collision():
            reward = -1
            game_over = True
            return reward, game_over, self.score
            
        if self.steps > len(self.snake) * 40:
            reward = -2
            game_over = True
            return reward, game_over, self.score
        # 4. place new food or just move
        if self.head == self.food:
            reward = 1
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, point=None):
        if point is None:
            point = self.head

        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            print("Se choco con el mismo")
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        directions = [ Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP ]
        index = directions.index(self.direction)

        if np.array_equal(action, [0, 1, 0]):
            index = (index - 1) % 4
        elif np.array_equal(action, [0, 0, 1]):
            index = (index + 1) % 4
        
        self.direction = directions[index]

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
    def get_conv_state(self):
        rows = self.h // self.BLOCK_SIZE
        cols = self.w // self.BLOCK_SIZE
    
        grid = np.zeros((6, rows, cols), dtype=np.float32)
    
        # Snake body
        for pt in self.snake:
            x = int(pt.x // self.BLOCK_SIZE)
            y = int(pt.y // self.BLOCK_SIZE)
            if 0 <= x < cols and 0 <= y < rows:  # Verifica límites
                grid[0, y, x] = 1.0
    
        # Food
        fx = int(self.food.x // self.BLOCK_SIZE)
        fy = int(self.food.y // self.BLOCK_SIZE)
        if 0 <= fx < cols and 0 <= fy < rows:  # Verifica límites
            grid[1, fy, fx] = 1.0
    
        # Direction indicators
        head = self.head
        point_l = Point(head.x - self.BLOCK_SIZE, head.y)
        point_r = Point(head.x + self.BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - self.BLOCK_SIZE)
        point_d = Point(head.x, head.y + self.BLOCK_SIZE)
    
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
    
        # Direcciones relativas a la actual
        left = None
        right = None
        forward = None
    
        if dir_r:
            left = point_u
            right = point_d
            forward = point_r
        elif dir_l:
            left = point_d
            right = point_u
            forward = point_l
        elif dir_u:
            left = point_l
            right = point_r
            forward = point_u
        elif dir_d:
            left = point_r
            right = point_l
            forward = point_d
    
        hx = int(head.x // self.BLOCK_SIZE)
        hy = int(head.y // self.BLOCK_SIZE)
    
        # Verifica colisiones y límites antes de actualizar el grid
        if left and 0 <= left.x < self.w and 0 <= left.y < self.h :
            grid[2, hy, hx] = 1.0  # izquierda disponible
        if forward and 0 <= forward.x < self.w and 0 <= forward.y < self.h:
            grid[3, hy, hx] = 1.0  # adelante disponible
        if right and 0 <= right.x < self.w and 0 <= right.y < self.h :
            grid[4, hy, hx] = 1.0  # derecha disponible
    
        # Dirección actual 
        if 0 <= hx < cols and 0 <= hy < rows:  # Verifica límites
            grid[5, hy, hx] = 1.0
    
        return grid
