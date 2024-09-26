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

    def __init__(self, w=640, h=480, n=2):
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

        self.score = 0
        self.robots = []  
        self.foods = []
        self.scores = [False] * self.n
        self._place_robots()
        self._place_food()
        self.frame_iteration = 0
    
            
    def _place_robots(self):
        center_point = Point(self.w // 2, self.h // 2)
        self.robots.append(center_point)  # Start by placing the first robot at the center
        while len(self.robots) < self.n:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            new_robot = Point(x, y)
            # Check that the new robot point doesn't overlap with existing robots or food points
            if new_robot not in self.robots and new_robot not in self.foods:
                self.robots.append(new_robot)
                
    def _place_food(self):
        self.foods = []  # Initialize an empty list to store food points
        self.scores = [False] * self.n
        while len(self.foods) < self.n:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            new_food = Point(x, y)
            # Check that the new food doesn't overlap with the robot or existing food
            if new_food not in self.foods and new_food not in self.robots:
                self.foods.append(new_food)

    def check_food_collision(self, reward):
        for i, robot in enumerate(self.robots):
            new_foods = []
            for food in self.foods:
                if robot == food:
                    self.scores[i] = True
                    self.score += 1  # Increase the score for the robot
                    reward = 50 if self.score % 2 == 0 else 25
                else:
                    new_foods.append(food)  # If not eaten, keep the food
            self.foods = new_foods  # Update the food list after checking all foods

            # If all food is eaten, place new food
            if not self.foods:
                self._place_food()
        return reward
    
    def play_step(self, actions):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(actions) # update the position
        
        # 3. check if game over
        reward = 0
        reward = reward - 1
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.robots):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        reward = self.check_food_collision(reward)
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


                    
    def is_collision(self, pts=None):
        if pts is None:
            pts = self.robots
        for pt in pts:
            # Check for collision with boundaries
            if pt.x >= self.w - BLOCK_SIZE or pt.x < 0 or pt.y >= self.h - BLOCK_SIZE or pt.y < 0:
                return True  # Collision with boundary detected

            # Check for collision with other robots
            for other in pts:
                if pt != other and pt.x == other.x and pt.y == other.y:
                    return True  # Collision with another robot detected

        return False  # No collision detected
    def is_collision_single_pt(self, pt):
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.robots[1:]:
            return True

        return False

    def closest_food_location(self, pt):
        closest_food = None
        min_distance = float('inf')

        # Loop through all food points to find the closest one
        for food in self.foods:
            distance = (pt.x - food.x) ** 2 + (pt.y - food.y) ** 2
            if distance < min_distance:
                min_distance = distance
                closest_food = food

        return closest_food
    
    def _update_ui(self):
        self.display.fill(BLACK)

        # for pt in self.robot:
        # draw the single robot
        for robot in self.robots:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(robot.x, robot.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(robot.x+4, robot.y+4, 12, 12))
        
        #draw the food
        for food in self.foods:
            pygame.draw.rect(self.display, RED, pygame.Rect(food.x, food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, actions):
        # Ensure self.robots and actions are lists and have the same length
        if not isinstance(actions, list) or not isinstance(self.robots, list) or len(actions) != len(self.robots):
            raise ValueError("Actions must be a list with the same number of elements as there are robots.")
        
        # Define the possible directions.
        directions = {
            (1, 0, 0, 0): 'UP',
            (0, 1, 0, 0): 'RIGHT',
            (0, 0, 1, 0): 'DOWN',
            (0, 0, 0, 1): 'LEFT',
            (0, 0, 0, 0): 'STAY',
        }

        # Loop over each robot and its corresponding action
        for i, action in enumerate(actions):
            # Set the new direction based on the action array
            direction = None
            for key, dir_value in directions.items():
                if np.array_equal(action, key):
                    direction = dir_value
                    break
            
            # Skip if direction is None or not in directions
            if direction not in directions.values():
                continue

            # Get the current position of the robot
            robot = self.robots[i]
            x, y = robot.x, robot.y

            # Update the position based on the direction
            if direction == 'RIGHT':
                x += BLOCK_SIZE
            elif direction == 'LEFT':
                x -= BLOCK_SIZE
            elif direction == 'DOWN':
                y += BLOCK_SIZE
            elif direction == 'UP':
                y -= BLOCK_SIZE

            # Update the robot's position
            self.robots[i] = Point(x, y)