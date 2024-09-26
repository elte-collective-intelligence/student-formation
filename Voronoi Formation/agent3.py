import torch
import random
import numpy as np
from collections import deque
from game3 import RobotsGameAI, Direction, Point
from model3_1 import Linear_QNet1, QTrainer1
from model3_2 import Linear_QNet2, QTrainer2
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self, id,Linear_QNet, QTrainer):
        self.id = id
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        robot = game.robots[self.id]
        point_l = Point(robot.x - 20, robot.y)
        point_r = Point(robot.x + 20, robot.y)
        point_u = Point(robot.x, robot.y - 20)
        point_d = Point(robot.x, robot.y + 20)
            
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        food = game.closest_food_location(robot)
        state = [
            # Danger straight
            (dir_r and game.is_collision_single_pt(point_r)) or 
            (dir_l and game.is_collision_single_pt(point_l)) or 
            (dir_u and game.is_collision_single_pt(point_u)) or 
            (dir_d and game.is_collision_single_pt(point_d)),

            # Danger right
            (dir_u and game.is_collision_single_pt(point_r)) or 
            (dir_d and game.is_collision_single_pt(point_l)) or 
            (dir_l and game.is_collision_single_pt(point_u)) or 
            (dir_r and game.is_collision_single_pt(point_d)),

            # Danger left
            (dir_d and game.is_collision_single_pt(point_r)) or 
            (dir_u and game.is_collision_single_pt(point_l)) or 
            (dir_r and game.is_collision_single_pt(point_u)) or 
            (dir_l and game.is_collision_single_pt(point_d)),
                
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
                
            # Food location 
            food.x < robot.x,  # food left
            food.x > robot.x,  # food right
            food.y < robot.y,  # food up
            food.y > robot.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, game):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        if game.scores[self.id]:
            final_move = [0,0,0,0] # to make it not to move until other robots find their target         
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent1 = Agent(0,Linear_QNet1, QTrainer1)
    agent2 = Agent(1,Linear_QNet2, QTrainer2)
    game = RobotsGameAI()
    while True:
        # get old state
        state_old1 = agent1.get_state(game)
        state_old2 = agent2.get_state(game)
        # get move
        final_move1 = agent1.get_action(state_old1,game)
        final_move2 = agent2.get_action(state_old2,game)
        
        final_moves = []
        final_moves.append(final_move1)
        final_moves.append(final_move2)
        
        # perform move and get new state
        reward, done, score = game.play_step(final_moves)
        
        state_new1 = agent1.get_state(game)
        state_new2 = agent2.get_state(game)
        
        # train short memory
        agent1.train_short_memory(state_old1, final_move1, reward, state_new1, done)
        agent2.train_short_memory(state_old2, final_move2, reward, state_new2, done)
        
        # remember
        agent1.remember(state_old1, final_move1, reward, state_new1, done)
        agent2.remember(state_old2, final_move2, reward, state_new2, done)
        
        if done:
            # train long memory, plot result
            game.reset()
            
            agent1.n_games += 1
            agent2.n_games += 1
            
            agent1.train_long_memory()
            agent2.train_long_memory()
            
            if score > record:
                record = score
                agent1.model.save()
                agent2.model.save()
                
            print('Game', agent1.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent1.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()