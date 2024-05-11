import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

from TrafficSimulator import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.number_games = 0
        self.epsilon = 0
        self.gamma = 0.85
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(11, 512, 2)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        # state list
        # 1. num_vehicle_each_roads (4 elements)
        # 2. current traffic configuration (4 elements)
        # 3. num_vehicle_pass (1 element)
        # 4. time_last_change (1 element)

        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y-20)
        point_d = Point(head.x, head.y+20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            #Move direction
            dir_l, dir_r, dir_u, dir_d,

            #Food loc
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        #rand moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.number_games
        final_move = [0,0] # [keep_light, next_light]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        #get old state
        state_old = agent.get_state(game)
        #get move
        final_move = agent.get_action(state_old)
        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new= agent.get_state(game)
        #train_short_memory
        agent.train_short_memory(state_old, final_move, reward,state_new, done)
        #remember
        agent.remember(state_old, final_move, reward,state_new, done)

        if done:
            #train long memory and plot result
            game.reset()
            agent.number_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.number_games, "Score", score, "Record", record)

            #plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.number_games
            plot_mean_score.append(mean_score)
            plot(plot_scores, plot_mean_score)



if __name__ == "__main__":
    train()


