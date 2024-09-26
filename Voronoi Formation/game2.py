import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

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
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20 # size of the rect
SPEED = 40  # robot move speed in one step

class RobotsGameAI:

    def __init__(self, w=640, h=480, n=10):
        self.w = w
        self.h = h
        self.n = n
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Path finder robots')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.robot = Point(self.w/2, self.h/2)

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
    

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food == self.robot:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the position
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 200*len(self.robot):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.robot == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.robot
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits other agents to check
        # if pt in self.robot[1:]:
        #     return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        # for pt in self.robot:
        # draw the single robot
        pygame.draw.rect(self.display, BLUE1, pygame.Rect(self.robot.x, self.robot.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, BLUE2, pygame.Rect(self.robot.x+4, self.robot.y+4, 12, 12))
        
        #draw the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # Define the possible directions.
        directions = {
            (1, 0, 0, 0): Direction.UP,
            (0, 1, 0, 0): Direction.RIGHT,
            (0, 0, 1, 0): Direction.DOWN,
            (0, 0, 0, 1): Direction.LEFT
        }
        
        # Set the new direction based on the action array.
        for key, direction in directions.items():
            if np.array_equal(action, key):
                self.direction = direction
                break

        x = self.robot.x
        y = self.robot.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.robot = Point(x, y)