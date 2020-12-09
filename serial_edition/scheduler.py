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
MEMORY_CAPACITY = 1



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
        if np.sum(selection) == 0:
            return 0
        ALPHA = 1
        BETA = 1
        M = len(loss_locals[0])
        # print(M)
        # print(selection)
        print(loss_locals)
        print((selection))
        # print((loss_locals.T).shape)
        self.F_r = np.matmul(selection,loss_locals.T)/(np.sum(selection))
        # self.F_r = np.sum(loss_locals)/(np.sum(selection))
        
        self.F_r = float(self.F_r)
        # print("self.F_r="+str(self.F_r))
        # float(np.matmul(FPF,(loss_locals.T)))
        Reward = ALPHA*(self.F_r_last-self.F_r)/(time_length*self.F_r)+BETA*float(np.matmul(FPF,(loss_locals.T))/np.sum(selection)-np.sum(FPF)/M)
        self.F_r_last = self.F_r
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
        self.agent = ddpg_mpn.DDPG(4, 16, 16, 1)
        self.env = Environment()


        self.FPF1_idx_lst = []
        self.time_counter_last = 0


        self.state_last = None
        self.action_last = None
        self.available_car = None


    def sch_mpn_initial(self, round_idx, time_counter):
        channel_state, self.available_car = self.env.update(time_counter)
        state = np.zeros((1, len(self.available_car[0]), 2))
        for i in range(len(self.available_car[0])):
            state[0, i, 0] = channel_state[0, i]
            state[0, i, 1] = 0
        itr_num, pointer, hidden_states = self.agent.choose_action(state)

        self.action_last = [itr_num, pointer, hidden_states]
        self.state_last = state

        client_indexes = []
        if len(pointer) != 0:
            for i in range(len(pointer)):
                client_indexes.append(int(self.available_car[0, pointer[i]]))
        local_itr = itr_num

        return client_indexes, local_itr        
        # s_client_indexes = str(list(client_indexes))[1:-1].replace(',', '')
        # s_local_itr = str(local_itr)                

        # return s_client_indexes + "," + s_local_itr             



    def sch_mpn(self, round_idx, time_counter, loss_locals, FPF_idx_lst):
        # ================================================================================================
        # calculate reward
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
        

        time_length = time_counter - self.time_counter_last
        reward = self.rwd.calculate_reward(loss_local, selection, FPF, time_length)
        # ================================================================================================
        # update state
        channel_state, self.available_car = self.env.update(time_counter)
        state = np.zeros((1, len(self.available_car[0]), 2))
        for i in range(len(self.available_car[0])):
            state[0, i, 0] = channel_state[0, i]
            if FPF_idx_lst==[]:
                state[0, i, 1] = 0
            else:
                state[0, i, 1] = FPF_idx_lst[int(self.available_car[0, i])]
        # ================================================================================================
        # train agent
        if len(pointer)>0:
            self.agent.store_transition(self.state_last, self.action_last, [reward], state)
        if self.agent.pointer > MEMORY_CAPACITY:
            loss_a, td_error = self.agent.learn()
        # ================================================================================================
        # produce action and mes
        itr_num, pointer, hidden_states = self.agent.choose_action(state)
        client_indexes = []
        if len(pointer) != 0:
            for i in range(len(pointer)):
                client_indexes.append(int(self.available_car[0, pointer[i]]))
        local_itr = itr_num
        return client_indexes, local_itr
        # s_client_indexes = str(list(client_indexes))[1:-1].replace(',', '')
        # s_local_itr = str(local_itr)                
        # # ================================================================================================
        # # record action,state,time_counter
        # self.action_last = [itr_num, pointer, hidden_states]
        # self.state_last = state
        # self.time_counter_last = time_counter

        # return s_client_indexes + "," + s_local_itr




