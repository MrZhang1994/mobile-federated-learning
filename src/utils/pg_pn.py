import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.PointerNet as PointerNet
import config


# Parameters for pg
MEMORY_CAPACITY = config.MEMORY_CAPACITY    # size of experience pool
LR_A = config.LR_A                          # learning rate for actor
use_gpu = config.use_gpu                    # use GPU or not

# print(device)
# Parameters for multi-layer PointerNetwork
FEATURE_DIMENSION = config.FEATURE_DIMENSION
MAXIMUM_CLIENT_NUM_PLUS_ONE = config.MAXIMUM_CLIENT_NUM_PLUS_ONE
AMEND_RATE = config.AMEND_RATE

def Amender(pointer, state):
    # print(type(pointer))
    # print(pointer)
    channel_state = state[0, :, 0]
    channel_state_avg = np.mean(channel_state)
    amended_pointer = copy.deepcopy(pointer)
    if type(amended_pointer) == np.ndarray:
        amended_pointer = amended_pointer.tolist()
    # amended_pointer = []
    # if len(pointer)>0:
    #     for i in range(len(pointer)):
    #         amended_pointer.append(pointer[i])
    for i in range(len(channel_state)):
        if (channel_state[i] >= channel_state_avg) and (i not in pointer) and (random.random()<AMEND_RATE):
            amended_pointer.append(i)
        elif (channel_state[i] < channel_state_avg) and (i in pointer) and (random.random()<AMEND_RATE):
            amended_pointer.remove(i)
    return amended_pointer

class ANetProb(nn.Module):
    def __init__(self, embedding_dimension, hidden_dimension, lstm_layers_num):
        super(ANetProb, self).__init__()

        self.embedding_dim = embedding_dimension
        self.hidden_dim = hidden_dimension

        self.pointer_network = PointerNet.PointerNet(input_dim = FEATURE_DIMENSION,
                                                        embedding_dim = embedding_dimension,
                                                        hidden_dim = hidden_dimension,
                                                        lstm_layers = lstm_layers_num,
                                                        dropout = 0,
                                                        bidir=False)

    def forward(self, state, action=None):
        '''
        state.shape = batch_size*max_carnum*feature_num
        input_lengths is the real carnum in each element of a batch
        '''
        # print(state.shape)

        if action is None:
            output, pointer, hidden_state = self.pointer_network(state, False)
        else:
            output, pointer, hidden_state = self.pointer_network(state, False, action)

        # pointer = pointer[0]

        log_prob = self.pointer_network.log_prob
        entropy = self.pointer_network.entropy

        return pointer[0], ((log_prob, entropy), pointer)



class PG(object):
    def __init__(self, embedding_dim, hidden_dim, lstm_layers):

        self.device = torch.device("cuda:" + str(config.device_No) if torch.cuda.is_available() else "cpu")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        self.Actor = ANetProb(embedding_dim, hidden_dim, lstm_layers)
        self.atrain = torch.optim.Adam(self.Actor.parameters(), lr=LR_A, weight_decay=1e-4)
        self.loss_td = nn.MSELoss()

        if use_gpu:
            self.Actor = self.Actor.to(self.device)
            self.loss_td = self.loss_td.to(self.device)

        self.pointer = 0
        self.learn_time = 0
        self.memory = []

    def choose_action_withAmender(self, state):
        return self.choose_action(state, True)

    def choose_action(self, state, amender=False):
        state = state.astype(np.float32)
        state_tensor = torch.FloatTensor(state)

        if use_gpu:
            state_tensor = state_tensor.to(self.device)

        with torch.no_grad():
            pointer, hidden_states = self.Actor(state_tensor)

        if use_gpu:
            pointer = pointer.cpu()
            (log_prob, _), pointer = hidden_states
            hidden_states = ((log_prob.cpu(), None), pointer.cpu())

        pointer = pointer.detach().numpy()
        # print(pointer)
        # pointer = pointer[0: np.argwhere(pointer == state.shape[1])]
        # ================================================================================================
        # # Amender
        if amender:
            pointer = Amender(pointer, state)
        # ================================================================================================      
        return pointer, hidden_states

    def learn(self):
        if config.DONT_TRAIN:
            return 0, 0
        self.learn_time += 1
        loss_a = []
        for bt in self.memory:
            bs = torch.FloatTensor(bt[0].astype(np.float32))
            (blog_prob, _), baction = bt[1][1]
            br = torch.FloatTensor(bt[2])
            bs_ = torch.FloatTensor(bt[3].astype(np.float32))

            if use_gpu:
                bs = bs.to(self.device)
                baction = baction.to(self.device)
                blog_prob = blog_prob.to(self.device)
                br = br.to(self.device) 
                bs_ = bs_.to(self.device) 

            pointer, ((log_prob, entropy), _) = self.Actor(bs, baction)
            data_term = -(br * torch.exp(log_prob - blog_prob)).detach() * log_prob
            reg_term = - config.REG_FACTOR * entropy
            loss_a.append(data_term + reg_term)
        loss_a = torch.stack(loss_a).mean()
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        if use_gpu:
            loss_a = loss_a.cpu()
        return float(loss_a), 0

    def store_transition(self, s, a, r, s_):

        if self.pointer < MEMORY_CAPACITY:
            self.memory.append([s, a, r, s_])
        else:  
            index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
            self.memory[index] = [s, a, r, s_]
        self.pointer += 1 

    def save_model(self, path):
        torch.save(self.Actor.cpu().state_dict(), path+'/Actor.pkl')
