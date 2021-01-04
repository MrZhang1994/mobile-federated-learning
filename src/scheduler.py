# import libraries
import re
import socket
import argparse
import csv
import numpy as np
import pandas as pd
import time
import copy

import ddpg_mpn
import pg_mpn
import config

# get some global variables
MEMORY_CAPACITY = config.MEMORY_CAPACITY
MAXIMUM_ITERATION_NUM = config.MAXIMUM_ITERATION_NUM
EMBEDDING_DIMENSION = config.EMBEDDING_DIMENSION
HIDDEN_DIMENSION = config.HIDDEN_DIMENSION
LSTM_LAYERS_NUM = config.LSTM_LAYERS_NUM

# set channel_data
channel_data = config.channel_data

# set the logger
logger = config.logger_sch

# set the csv
scheduler_csv = config.scheduler_csv

# used for round robin
queue = []
# used for sch_loss
prev_cars = []

class Reward:
    def __init__(self):
        self.F_r = 0
        self.F_r_last = 0

    def calculate_reward(self, loss_locals, selection, FPF, time_length):
        """
        loss_local: 1*M array
        selection: 1*M array, binary
        FPF: 1*M
        Calculate reward
        """
        if np.sum(selection) == 0:
            return 0
        ALPHA = 1
        BETA = 1
        M = len(loss_locals[0])

        self.F_r = np.matmul(selection,loss_locals.T)/(np.sum(selection))
        self.F_r = float(self.F_r)

        Reward = ALPHA*(self.F_r_last-self.F_r)/(time_length*self.F_r)+BETA*float(np.matmul(FPF,(loss_locals.T))/np.sum(selection)-np.sum(FPF)/M)
        self.F_r_last = self.F_r
        Reward = 100000*Reward
        return Reward
        
class Environment:
    '''
    about the environment
    '''
    def __init__(self): 
        self.a = None

    def update(self, time_counter):
        cars = list(channel_data[channel_data['Time'] == time_counter]['Car'])
        Distance = list(channel_data[channel_data['Time'] == time_counter]['Distance to BS(4982,905)'])

        channel_state = np.zeros((1, len(cars)))
        available_car = np.zeros((1, len(cars)))

        for i in range(len(cars)):
            if Distance[i] == 0:
                channel_state[0, i] = 1
            else:
                channel_state[0, i] = 1/Distance[i]
            available_car[0, i] = cars[i]
        return channel_state, available_car

    
class Scheduler_MPN:
    def __init__(self):
        self.rwd = Reward()
        RL = pg_mpn.PG if config.NAIVE_PG else ddpg_mpn.DDPG
        self.agent = RL(MAXIMUM_ITERATION_NUM, EMBEDDING_DIMENSION, HIDDEN_DIMENSION, LSTM_LAYERS_NUM)
        self.env = Environment()

        self.FPF1_idx_lst = []
        self.time_counter_last = 0

        self.state_last = None
        self.action_last = None
        self.available_car = None

    def sch_mpn_empty(self, round_idx, time_counter):
        channel_state, self.available_car = self.env.update(time_counter)
        state = np.zeros((1, len(self.available_car[0]), 3))
        for i in range(len(self.available_car[0])):
            state[0, i, 0] = channel_state[0, i]
            state[0, i, 1] = 0
            state[0, i, 2] = 0
        itr_num, pointer, hidden_states = self.agent.choose_action_withAmender(state)

        self.action_last = [itr_num, pointer, hidden_states]
        self.state_last = state

        client_indexes = []
        if len(pointer) != 0:
            for i in range(len(pointer)):
                client_indexes.append(int(self.available_car[0, pointer[i]]))
        local_itr = itr_num

        # write to the scheduler csv
        with open(scheduler_csv, mode = "w+", encoding='utf-8', newline='') as file:
            csv_writer = csv.writer(file)
            if round_idx == 0:
                csv_writer.writerow(['time counter', 'available car', 'channel_state', 'pointer', 'client index', 'iteration', 'reward', 'loss_a', 'loss_c'])
            csv_writer.writerow([time_counter, str(self.available_car[0].tolist()),\
                                    str(state[0].tolist()), str(pointer),\
                                    str(client_indexes), itr_num])
            file.flush()
        return client_indexes, local_itr        

    def sch_mpn_test(self, round_idx, time_counter):
        channel_state, self.available_car = self.env.update(time_counter)
        state = np.zeros((1, len(self.available_car[0]), 3))
        for i in range(len(self.available_car[0])):
            state[0, i, 0] = channel_state[0, i]
            state[0, i, 1] = 0
            state[0, i, 2] = 0
        itr_num, pointer, hidden_states = self.agent.choose_action(state)

        self.action_last = [itr_num, pointer, hidden_states]
        self.state_last = state

        client_indexes = []
        if len(pointer) != 0:
            for i in range(len(pointer)):
                client_indexes.append(int(self.available_car[0, pointer[i]]))
        local_itr = itr_num

        return client_indexes, local_itr   

    def sch_mpn(self, round_idx, time_counter, loss_locals, FPF_idx_lst, local_loss_lst):
        # ================================================================================================
        # calculate reward
        selection = np.zeros((1, len(self.available_car[0])))
        FPF = np.zeros((1, len(self.available_car[0])))
        loss_local = np.zeros((1, len(self.available_car[0])))
        pointer = self.action_last[1]

        if len(pointer)>0:
            for i in range(len(pointer)):
                selection[0, int(pointer[i])] = 1
                FPF[0, int(pointer[i])] = FPF_idx_lst[int(self.available_car[0, int(pointer[i])])]
                loss_local[0, int(pointer[i])] = loss_locals[i]
        
        time_length = time_counter - self.time_counter_last
        reward = self.rwd.calculate_reward(loss_local, selection, FPF, time_length)
        # ================================================================================================
        # update state
        channel_state, self.available_car = self.env.update(time_counter)

        state = np.zeros((1, len(self.available_car[0]), 3))
        for i in range(len(self.available_car[0])):
            state[0, i, 0] = channel_state[0, i]
            if FPF_idx_lst==[]:
                state[0, i, 1] = 0
            else:
                state[0, i, 1] = FPF_idx_lst[int(self.available_car[0, i])]
            state[0, i, 2] = local_loss_lst[0, int(self.available_car[0, i])]
        # del available_car_last
        # ================================================================================================
        # train agent
        self.agent.store_transition(self.state_last, self.action_last, [reward], state)
        loss = []
        if self.agent.memory:
            loss_a, td_error = self.agent.learn()
            loss = [loss_a, td_error]
        # ================================================================================================
        # produce action and mes
        if round_idx < config.AMEND_ITER:
            itr_num, pointer, hidden_states = self.agent.choose_action_withAmender(state)
        else:
            itr_num, pointer, hidden_states = self.agent.choose_action(state)
        
        client_indexes = []
        if len(pointer) != 0:
            for i in range(len(pointer)):
                client_indexes.append(int(self.available_car[0, pointer[i]]))
        local_itr = itr_num    
        # ================================================================================================
        # record action,state,time_counter
        self.action_last = [itr_num, pointer, hidden_states]
        self.state_last = state
        self.time_counter_last = time_counter

        # write to the scheduler csv
        with open(scheduler_csv, mode = "w+", encoding='utf-8', newline='') as file:
            csv_writer = csv.writer(file)
            if round_idx == 0:
                csv_writer.writerow(['time counter', 'available car', 'channel_state', 'pointer', 'client index', 'iteration', 'reward', 'loss_a', 'loss_c'])
            csv_writer.writerow([time_counter, str(self.available_car[0].tolist()),\
                                    str(state[0].tolist()), str(pointer),\
                                    str(client_indexes), itr_num])
            file.flush()
        return client_indexes, local_itr, reward

def sch_random(round_idx, time_counter):
    # set the seed
    np.random.seed(round_idx)

    # random sample clients
    cars = list(channel_data[channel_data['Time'] == time_counter]['Car'])
    client_indexes = []
    if cars:
        client_indexes = list(np.random.choice(cars, max(int(len(cars) / 2), 1), replace=False).ravel())
    logger.info("client_indexes = " + str(client_indexes) + "; time_counter = " + str(time_counter))

    # random local iterations
    local_itr = np.random.randint(2) + 1
    return client_indexes, local_itr

def sch_channel(round_idx, time_counter):
    # set the seed
    np.random.seed(round_idx)

    # sample only based on channel condition
    curr_channel = channel_data[channel_data['Time'] == time_counter]
    curr_channel = curr_channel.sort_values(by=['Distance to BS(4982,905)'],
                                            ascending=True)  # sort by the channel condition

    client_indexes = ((curr_channel.iloc[:int((len(curr_channel) + 1) / 2),:]).loc[:, 'Car']).tolist()

    # random local iterations
    local_itr = np.random.randint(2) + 1
    return client_indexes, local_itr
 
def sch_rrobin(round_idx, time_counter):
    # set the seed
    np.random.seed(round_idx)

    cars = list(channel_data[channel_data['Time'] == time_counter]['Car'])
    queue.extend(cars)
    client_indexes = []
    num_client = int(len(cars) / 2) + 1

    while len(client_indexes) < num_client:
        car_temp = queue.pop(0)
        if car_temp in cars:  # add the car exist in the current time
            client_indexes.append(car_temp)

    # random local iterations
    local_itr = np.random.randint(2) + 1
    return client_indexes, local_itr

def sch_loss(round_idx, time_counter):
    cars = list(channel_data[channel_data['Time'] == time_counter]['Car'])

    if len(loss_locals) == 0:  # no loss value before, random choose
        client_indexes = list(np.random.choice(cars, max(int(len(cars) / 2), 1), replace=False).ravel())
    else:
        client_indexes = []
        while len(loss_locals) != 0:  # find the car with max loss value and exist in the current time counter
            max_idx = loss_locals.index(max(loss_locals))
            tmp = prev_cars[max_idx]
            if tmp in cars:
                client_indexes.append(tmp)
                break
            else:
                del loss_locals[max_idx]
        if len(client_indexes) == 0:  # no such car
            client_indexes = list(np.random.choice(cars, max(int(len(cars) / 2), 1), replace=False).ravel())

    logging.info("client_indexes = " + str(client_indexes) + "; time_counter = " + str(time_counter))
    prev_cars.clear()
    prev_cars.extend(cars)  # used for next time calling scheduler
    # random local iterations
    local_itr = np.random.randint(2) + 1
    return client_indexes, local_itr