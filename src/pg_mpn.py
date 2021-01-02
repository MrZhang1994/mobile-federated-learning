import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
# import config
# import convlstm
import PointerNet
import config

from ddpg_mpn import ANet

# Parameters for ddpg
MEMORY_CAPACITY = config.MEMORY_CAPACITY    # size of experience pool
LR_A = config.LR_A                          # learning rate for actor
GAMMA = config.GAMMA                        # reward discount
TAU = config.TAU                            # soft replacement
use_gpu = config.use_gpu                    # use GPU or not
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
# print(device)
# Parameters for multi-layer PointerNetwork
FEATURE_DIMENSION = config.FEATURE_DIMENSION
MAXIMUM_CLIENT_NUM_PLUS_ONE = config.MAXIMUM_CLIENT_NUM_PLUS_ONE


class ANetProb(ANet):
    def forward(self, state, action=None):
        '''
        state.shape = batch_size*max_carnum*feature_num
        input_lengths is the real carnum in each element of a batch
        '''
        # print(state.shape)
        itr = torch.ones((1,state.shape[1],1))
        eof = torch.zeros((1,1,FEATURE_DIMENSION))
        state_list = []

        if use_gpu:
            itr = itr.to(device) 
            eof = eof.to(device) 

        for i in range(self.max_itr_num):
            state_list.append(torch.cat((state, (i+1)*itr), dim = 2))
            state_list[i] = torch.cat((state_list[i], eof), dim = 1)

        outputs = []
        pointers = []
        hidden_states = []


        for j in range(self.max_itr_num):
            if action is None:
                output, pointer, hidden_state = self.pointer_network_layer1[j](state_list[j], False)
            else:
                output, pointer, hidden_state = self.pointer_network_layer1[j](state_list[j], False, action[0][j])

            outputs.append(output[:,0,:])
            pointers.append(pointer)
            hidden_states.append(hidden_state)

        outputs = torch.cat(outputs, 0)
        outputs = torch.unsqueeze(outputs, dim = 0)

        output_60 = torch.zeros((1,self.max_itr_num,MAXIMUM_CLIENT_NUM_PLUS_ONE))
        if use_gpu:
            output_60 = output_60.to(device) 
        output_60[:,:,0:outputs.size(2)] = outputs

        if action is None:
            output2, pointer2, hidden_state2 = self.pointer_network_layer2(output_60, False, None, True)
        else:
            output2, pointer2, hidden_state2 = self.pointer_network_layer2(output_60, False, action[1], True)

        itr_num = pointer2[0, 0]  
        pointer = pointers[itr_num][0]
        log_prob = self.pointer_network_layer2.log_prob + self.pointer_network_layer1[j].log_prob
        entropy = self.pointer_network_layer2.entropy + self.pointer_network_layer1[j].entropy

        return itr_num, pointer, ((log_prob, entropy), (pointers, pointer2))


class PG(object):
    def __init__(self, max_itr_num, embedding_dim, hidden_dim, lstm_layers):

        self.max_itr_num = max_itr_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        self.Actor = ANetProb(self.max_itr_num, embedding_dim, hidden_dim, lstm_layers)

        self.atrain = torch.optim.Adam(self.Actor.parameters(), lr=LR_A, weight_decay=1e-4)
        self.loss_td = nn.MSELoss()
        if use_gpu:
            self.Actor = self.Actor.to(device)
            for i in range(self.Actor.max_itr_num):
                self.Actor.pointer_network_layer1[i] = self.Actor.pointer_network_layer1[i].to(device)
            self.loss_td = self.loss_td.to(device)

        self.pointer = 0
        self.learn_time = 0
        self.memory = []

    def choose_action_withAmender(self, state):
        return self.choose_action(state)

    def choose_action(self, state):
        state = state.astype(np.float32)
        ss = torch.FloatTensor(state)
        num_clients = state.shape[1]

        if use_gpu:
            ss = ss.to(device)   
        # print(ss)
        with torch.no_grad():
            itr_num, pointer, hidden_states = self.Actor(ss)
        # print("MPN:")
        # print(itr_num)
        # print(pointer)
        if use_gpu:
            itr_num = itr_num.cpu()
            pointer = pointer.cpu()
            (log_prob, _), (ptrs_1, ptr2) = hidden_states
            hidden_states = (log_prob.cpu(), None), ([p.cpu() for p in ptrs_1], ptr2.cpu())

        itr_num = itr_num.detach().numpy() + 1
        pointer = pointer.detach().numpy()
        count = 0
        for i in range(len(pointer)):
            if pointer[i] == num_clients:
                count = i
        if count == 0:
            pointer = []
        else:
            pointer = pointer[0: count]
        # ================================================================================================
        # # Amender
        # itr_num, pointer = Amender(itr_num, pointer, state)
        # ================================================================================================      
        return itr_num, pointer, hidden_states

    def learn(self):
        if config.DONT_TRAIN:
            return 0, 0
        self.learn_time += 1
        loss_a = []
        for bt in self.memory:
            bs = torch.FloatTensor(bt[0].astype(np.float32))
            bitr_num = torch.FloatTensor((np.expand_dims(bt[1][0], axis=0)).astype(np.float32))
            (blog_prob, _), baction = bt[1][2]
            br = torch.FloatTensor(bt[2])
            bs_ = torch.FloatTensor(bt[3].astype(np.float32))

            if use_gpu:
                bs = bs.to(device) 
                bitr_num = bitr_num.to(device) 
                blog_prob = blog_prob.to(device)
                br = br.to(device) 
                bs_ = bs_.to(device) 

            itr_num, pointer, ((log_prob, entropy), _) = self.Actor(bs, baction)
            loss_a.append(-(br * torch.exp(log_prob - blog_prob)).detach() * log_prob - 1e-2 * entropy)
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
