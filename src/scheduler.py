# ************************************************************************************************************ #
# import libraries
import re
import socket
import argparse
import numpy as np
import pandas as pd
import os
import logging
import time
import copy

import ddpg_mpn
import pg_mpn
import config

MEMORY_CAPACITY = config.MEMORY_CAPACITY
MAXIMUM_ITERATION_NUM = config.MAXIMUM_ITERATION_NUM
EMBEDDING_DIMENSION = config.EMBEDDING_DIMENSION
HIDDEN_DIMENSION = config.HIDDEN_DIMENSION
LSTM_LAYERS_NUM = config.LSTM_LAYERS_NUM



# set the logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('schedule')

# set the data dir & read the data
channel_data_dir = "../data"
channel_data = pd.concat(
    [pd.read_csv(os.path.join(channel_data_dir, csv_name), error_bad_lines=False) for csv_name in os.listdir(channel_data_dir)],
    ignore_index=True)

# used for round robin
queue = []
# used for sch_loss
prev_cars = []
loss_locals = []



class Reward:
    def __init__(self):
        self.F_r = 0
        self.F_r_last = 0

    def calculate_reward(self, loss_locals, selection, FPF, time_length):
        '''
        Calculate reward
        '''
        # loss_local: 1*M array
        # selection: 1*M array, binary
        # FPF: 1*M
        #===============================================
        # M = len(loss_locals[0])
        # print(M)
        # print(selection)
        # print(loss_locals)
        # print((selection))
        # print((loss_locals.T).shape)
        if np.sum(selection) == 0:
            return 0
        ALPHA = 1
        BETA = 1
        M = len(loss_locals[0])

        self.F_r = np.matmul(selection,loss_locals.T)/(np.sum(selection))
        # self.F_r = np.sum(loss_locals)/(np.sum(selection))
        
        self.F_r = float(self.F_r)
        # print("self.F_r="+str(self.F_r))
        # float(np.matmul(FPF,(loss_locals.T)))
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
        self.agent = RL(LSTM_LAYERS_NUM, EMBEDDING_DIMENSION, HIDDEN_DIMENSION, LSTM_LAYERS_NUM)
        self.env = Environment()


        self.FPF1_idx_lst = []
        self.time_counter_last = 0


        self.state_last = None
        self.action_last = None
        self.available_car = None


    def sch_mpn_initial(self, round_idx, time_counter, csv_writer2):
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

        csv_writer2.writerow([time_counter, str(self.available_car[0].tolist()),\
                                str(state[0].tolist()), str(pointer),\
                                str(client_indexes), itr_num])
        return client_indexes, local_itr        
        # s_client_indexes = str(list(client_indexes))[1:-1].replace(',', '')
        # s_local_itr = str(local_itr)                

        # return s_client_indexes + "," + s_local_itr         

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


    def sch_mpn(self, round_idx, time_counter, loss_locals, FPF_idx_lst, local_loss_lst, csv_writer2):
        # ================================================================================================
        # calculate reward
        csv_writer2_line = []
        csv_writer2_line.append(time_counter)
        selection = np.zeros((1, len(self.available_car[0])))
        FPF = np.zeros((1, len(self.available_car[0])))
        loss_local = np.zeros((1, len(self.available_car[0])))
        pointer = self.action_last[1]
        
        # print(self.available_car[0])
        # print(pointer)
        # print(FPF_idx_lst)
        # print(len(FPF_idx_lst))
        # print(loss_locals)

        if len(pointer)>0:
            for i in range(len(pointer)):
                selection[0, int(pointer[i])] = 1
                FPF[0, int(pointer[i])] = FPF_idx_lst[int(self.available_car[0, int(pointer[i])])]
                loss_local[0, int(pointer[i])] = loss_locals[i]
        
        # print(pointer)
        # print(selection[0])
        time_length = time_counter - self.time_counter_last
        reward = self.rwd.calculate_reward(loss_local, selection, FPF, time_length)
        # print(reward)
        # ================================================================================================
        # update state
        channel_state, self.available_car = self.env.update(time_counter)

        csv_writer2_line.append(str(self.available_car[0].tolist()))
        csv_writer2_line.append(str(channel_state[0].tolist()))

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
        if len(pointer)>0:
            self.agent.store_transition(self.state_last, self.action_last, [reward], state)
        loss = []
        if self.agent.pointer > MEMORY_CAPACITY:
            loss_a, td_error = self.agent.learn()
            loss = [loss_a, td_error]
        # ================================================================================================
        # produce action and mes
        if round_idx<100:
            itr_num, pointer, hidden_states = self.agent.choose_action_withAmender(state)
        else:
            itr_num, pointer, hidden_states = self.agent.choose_action(state)
        csv_writer2_line.append(str(pointer))
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
        csv_writer2_line.append(str(client_indexes))
        csv_writer2_line.append(local_itr)
        csv_writer2_line.append(reward)
        csv_writer2.writerow(csv_writer2_line+loss)

        return client_indexes, local_itr


def sch_random(round_idx, time_counter, csv_writer2):
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
    csv_writer2.writerow([time_counter, str(client_indexes), local_itr])
    return client_indexes, local_itr



def sch_channel(round_idx, time_counter, csv_writer2):
    # set the seed
    np.random.seed(round_idx)

    # sample only based on channel condition
    curr_channel = channel_data[channel_data['Time'] == time_counter]
    curr_channel = curr_channel.sort_values(by=['Distance to BS(4982,905)'],
                                            ascending=True)  # sort by the channel condition
    # print(curr_channel)
    # print(int((len(curr_channel)+1)/2))

    client_indexes = ((curr_channel.iloc[:int((len(curr_channel) + 1) / 2),:]).loc[:, 'Car']).tolist()

    # random local iterations
    local_itr = np.random.randint(2) + 1
    csv_writer2.writerow([time_counter, str(client_indexes), local_itr])
    return client_indexes, local_itr
 

def sch_rrobin(round_idx, time_counter, csv_writer2):
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

    # logger.info("client_indexes = " + str(client_indexes) + "; time_counter = " + str(time_counter))

    # random local iterations
    local_itr = np.random.randint(2) + 1
    csv_writer2.writerow([time_counter, str(client_indexes), local_itr])
    return client_indexes, local_itr


def sch_loss(round_idx, time_counter, csv_writer2):
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

    csv_writer2.writerow([time_counter, str(client_indexes), local_itr])
    return client_indexes, local_itr