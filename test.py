
import pandas as pd
from TrafficSimulation.trafficsim import TrafficSim


df = pd.read_csv('clean_data.csv')

# trafsim = TrafficSim()
# for i in range(len(df)):
#     trafsim.update_state(df)
#     print(trafsim.get_log())



import torch
import random
import numpy as np
from collections import deque
# from game import SnakeGameAI, Direction, Point
from deepqnet.model import Linear_QNet, QTrainer
from deepqnet.helper import plot

# from TrafficSimulator import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1
LR = 0.001

class Agent:

    def __init__(self):
        self.number_sims = 0
        self.epsilon = 0
        self.gamma = 0.85
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(10, 512, 2)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.last_time_state = 0
        self.last_traffic_configuration = [False, False, False, False]


    def get_state(self, sim):
        # state list
        # 1. num_vehicle_each_roads (4 elements)
        # 2. current traffic configuration (4 elements)
        # 3. num_vehicle_pass (1 element)
        # 4. time_last_change (1 element)

        log = sim.get_log()
        vehicles_present = log['vehicles_present']
        traffic_light_cycle = log['traffic_light_cycle']
        traffic_light_cycle = np.asarray(traffic_light_cycle).astype(int).tolist()
        vehicles_passed = log['vehicles_passed']
        last_time_change = log['time'] - self.last_time_state
        if self.last_traffic_configuration != log['traffic_light_cycle']:
            self.last_time_state = log['time']
            self.last_traffic_configuration = log['traffic_light_cycle']

        state = [a for a in vehicles_present]
        state = state + [vehicles_passed, last_time_change] + traffic_light_cycle
        state = np.array(state)
        # print(traffic_light_cycle)
        return state.reshape(1, -1)

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
        self.epsilon = 2 - self.number_sims
        if state[0,5] > 2:
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 2)
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
        else:
            move = 0
        return move


def reward_model(sim, state_old, last_time):
    log = sim.get_log()
    vehicles_present = log['vehicles_present']
    vehicles_passed = log['vehicles_passed']
    time = int(log['time']) // 2
    if (sum(vehicles_present)/len(vehicles_present)) < max(vehicles_present):
        reward = -10
    elif min(vehicles_present) < max(vehicles_present):
        reward = -5
    elif last_time > 50:
        reward = -(last_time % 10)
    else:
        reward = 10
    if time > 4674:
        done = True
    else: 
        done = False
    # current_score += reward
    return reward, done


def log_files(sim, iteration):
    log = sim.get_log()
    vehicles_present = log['vehicles_present']
    traffic_light_cycle = log['traffic_light_cycle']
    traffic_light = [index for index, value in enumerate(traffic_light_cycle) if value]
    vehicles_passed = log['vehicles_passed']
    time = log['time'] // 2
    iteration = iteration
    out = f'{iteration}, {time}, {vehicles_present[0]}, {vehicles_present[1]}, {vehicles_present[2]}, {vehicles_present[3]}, {vehicles_passed}, {traffic_light[0]}'
    return out




def train():
    plot_scores = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    sim = TrafficSim()
    sim.update_state(df)
    files = open('log.csv', 'a')
        # Append a new line
    new_line = 'iteration, time, vehicles_present[0], vehicles_present[1], vehicles_present[2], vehicles_present[3], vehicles_passed, traffic_light[0]'
    files.write(new_line + '\n')
    while True:
        #get old state
        state_old = agent.get_state(sim)
        #get move
        final_move = agent.get_action(state_old)
        #perform move and get new state
        sim.update_state(df, output=final_move)

        reward, done = reward_model(sim, state_old, agent.last_time_state)
        state_new= agent.get_state(sim)
        
        #remember
        agent.remember(state_old, final_move, reward, state_new, done)

        #train_short_memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        log = log_files(sim, agent.number_sims)
        files.write(log + '\n')
        if done:
            #train long memory and plot result
            
            agent.number_sims += 1
            agent.train_long_memory()
            sim = TrafficSim()
            sim.update_state(df)

            # if score > record:
            #     record = score
            agent.model.save()

            print("sim", agent.number_sims)

            # #plot
            # plot_scores.append(score)
            # total_score += score
            # mean_score = total_score/agent.number_sims
            # plot_mean_score.append(mean_score)
            # plot(plot_scores, plot_mean_score)



if __name__ == "__main__":
    train()


