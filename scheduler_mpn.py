# # Libs used
import numpy as np
import socket
import csv
import random
import ddpg_mpn

MEMORY_CAPACITY = 1

class Environment:
    '''
    about the environment
    '''
    def __init__(self, Date):
        self.Date = Date
        f = open('../data/Car_info_'+str(Date)+'.txt','r')
        a = f.read()
        self.car_info = eval(a)
        self.one_day_length = len(self.car_info)     
        # print(self.car_info)

    def update(self, time_slot):
        channel_state = np.zeros((1, len(self.car_info[time_slot])))
        available_car = np.zeros((1, len(self.car_info[time_slot])))

        for i in range(len(channel_state[0])):
            if self.car_info[time_slot][i][4] == 0:
                channel_state[0, i] = 1
            else:
                channel_state[0, i] = 1/self.car_info[time_slot][i][4]
            available_car[0, i] = self.car_info[time_slot][i][2]
        return channel_state, available_car

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
        # print(loss_locals.T)
        self.F_r = np.matmul(selection,(loss_locals.T))/(np.sum(selection))
        self.F_r = float(self.F_r)
        # print("self.F_r="+str(self.F_r))
        float(np.matmul(FPF,(loss_locals.T)))
        Reward = ALPHA*(self.F_r_last-self.F_r)/(time_length*self.F_r)+BETA*float(np.matmul(FPF,(loss_locals.T))/np.sum(selection)-np.sum(FPF)/M)
        self.F_r_last = self.F_r
        return Reward


# host = socket.gethostname()
# server = socket.socket()
# server.bind((host, 8999)) # bind the server to port 8999. Before running the code, make sure there is no program running under this port.
# server.listen(5) # maximum connections 5.

rwd = Reward()
agent = ddpg_mpn.DDPG(4, 16, 16, 1)

for Date in range(1001, 1011):

    print('Date = '+str(Date))
    env = Environment(Date)

    time_counter_last = 0
    time_counter = 0

    channel_state, available_car = env.update(0)


    state = np.zeros((1, len(available_car[0]), 2))
    for i in range(len(available_car[0])):
        state[0, i, 0] = channel_state[0, i]
        state[0, i, 1] = 0


    # for t in range(1, env.one_day_length):
    for t in range(2000, 2005): 
        itr_num, pointer, hidden_states = agent.choose_action(state)
        print("time:  "+str(t))
        print(itr_num)
        print(pointer)

        # itr_num: a number
        # pointer: a array sized choosen car num
        # hidden_states: a 1*((itr+1)*hidden_dim) array



        client_indexes = []
        if len(pointer) != 0:
            for i in range(len(pointer)):
                client_indexes.append(int(available_car[0, pointer[i]]))
        local_itr = itr_num
        # print(available_car)
        # print(client_indexes)

        # socket
        client, addr = server.accept() # connected to client. 


        s_client_indexes = str(list(client_indexes))[1:-1].replace(',', '')
        s_local_itr = str(local_itr)
        mes = s_client_indexes + "," + s_local_itr               
        # socket send action
        client.send(mes.encode()) # send the message to the connected client.
        # waiting for respond
        response = client.recv(1000000).decode()
        loss_locals = []
        FPF1_idx_lst = []
        if response != "nothing":
            response = response.split(',')
            if response[0] != '':
                time_counter = int(float(response[0]))
            if response[1] != '':
                loss_locals = [float(i) for i in response[1].split(' ')]
            if response[2] != '':
                FPF1_idx_lst = [float(i) for i in response[2].split(' ')]
        client.close()

        # loss_locals = []
        # FPF1_idx_lst = []
        # for i in range(len(available_car[0])):
        #     loss_locals.append(100*random.random())
        # for i in range(10000):
        #     FPF1_idx_lst.append(random.random())
        # time_counter = 10*t

        # Calculate reward
        selection = np.zeros((1, len(available_car[0])))
        FPF = np.zeros((1, len(available_car[0])))

        if len(pointer)>0:
            for i in range(len(pointer)):
                selection[0, int(pointer[i])] = 1
                FPF[0, int(pointer[i])] = FPF1_idx_lst[int(available_car[0, int(pointer[i])])]
                
        time_length = time_counter - time_counter_last
        reward = rwd.calculate_reward(np.array([loss_locals]), selection, FPF, time_length)
        print(reward)
        channel_state, available_car = env.update(t)
        # print(available_car)
        # print(channel_state)
        state_ = np.zeros((1, len(available_car[0]), 2))
        # print(state_)
        for i in range(len(available_car[0])):
            state_[0, i, 0] = channel_state[0, i]
            # state_[0, i, 1] = FPF1_idx_lst[available_car[0MEMORY_CAPACITY, i]] 
            state_[0, i, 1] = 0

        # print(state_)
        # store experience and train
        if len(pointer)>0:
            agent.store_transition(state, [itr_num, pointer, hidden_states], [reward], state_)

        if agent.pointer > MEMORY_CAPACITY:
            loss_a, td_error = agent.learn()
            # csv_loss.writerow([t, agent.learn_time, loss_a, td_error])
        time_counter_last = time_counter

        state = state_ 

        # break
    break