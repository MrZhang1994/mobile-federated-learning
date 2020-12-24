import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import PointerNet
import config

# Parameters for ddpg
MEMORY_CAPACITY = config.MEMORY_CAPACITY    # size of experience pool
LR_A = config.LR_A                          # learning rate for actor
LR_C = config.LR_C                          # learning rate for critic
GAMMA = config.GAMMA                        # reward discount
TAU = config.TAU                            # soft replacement
use_gpu = config.use_gpu                    # use GPU or not
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
# Parameters for multi-layer PointerNetwork
FEATURE_DIMENSION = config.FEATURE_DIMENSION
MAXIMUM_CLIENT_NUM_PLUS_ONE = config.MAXIMUM_CLIENT_NUM_PLUS_ONE
AMEND_RATE = config.AMEND_RATE


def Amender(itr_num, pointer, state):
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
    amended_itr = itr_num
    for i in range(len(channel_state)):
        if (channel_state[i] >= channel_state_avg) and (i not in pointer) and (random.random()<AMEND_RATE):
            amended_pointer.append(i)
        elif (channel_state[i] < channel_state_avg) and (i in pointer) and (random.random()<AMEND_RATE):
            amended_pointer.remove(i)
    if random.random()<AMEND_RATE:
        amended_itr = math.ceil(min(1, (len(amended_pointer)/40))/(1/4))
    amended_itr = np.array(amended_itr)
    return amended_itr, amended_pointer

class ANet(nn.Module):
    def __init__(self, max_itr_num, embedding_dimension, hidden_dimension, lstm_layers_num):
        super(ANet, self).__init__()

        self.max_itr_num = max_itr_num
        self.embedding_dim = embedding_dimension
        self.hidden_dim = hidden_dimension

        self.pointer_network_layer1 = []
        for i in range(self.max_itr_num):
            self.pointer_network_layer1.append(PointerNet.PointerNet(input_dim = FEATURE_DIMENSION,
                                                                        embedding_dim = embedding_dimension,
                                                                        hidden_dim = hidden_dimension,
                                                                        lstm_layers = lstm_layers_num,
                                                                        dropout = 0,
                                                                        bidir=False))

        self.pointer_network_layer2 = PointerNet.PointerNet(input_dim = MAXIMUM_CLIENT_NUM_PLUS_ONE,
                                                                embedding_dim = embedding_dimension,
                                                                hidden_dim = hidden_dimension,
                                                                lstm_layers = lstm_layers_num,
                                                                dropout = 0,
                                                                bidir=False)
                
    def forward(self, state):
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
            output, pointer, hidden_state = self.pointer_network_layer1[j](state_list[j])
            outputs.append(output[:,0,:])
            pointers.append(pointer)
            hidden_states.append(hidden_state)

        outputs = torch.cat(outputs, 0)
        outputs = torch.unsqueeze(outputs, dim = 0)

        output_60 = torch.zeros((1,self.max_itr_num,MAXIMUM_CLIENT_NUM_PLUS_ONE))
        if use_gpu:
            output_60 = output_60.to(device) 
        output_60[:,:,0:outputs.size(2)] = outputs

        output2, pointer2, hidden_state2 = self.pointer_network_layer2(output_60)

        itr_num = pointer2[0, 0]  
        pointer = pointers[itr_num][0]

        hidden_states.append(hidden_state2)
        hidden_states = torch.cat(hidden_states, dim = 1)

        return itr_num, pointer, hidden_states

class CNet(nn.Module):
    def __init__(self, max_itr_num, embedding_dimension, hidden_dimension, lstm_layers_num):
        super(CNet, self).__init__()

        self.max_itr_num = max_itr_num
        self.embedding_dim = embedding_dimension
        self.hidden_dim = hidden_dimension

        self.pointer_network_layer1 = []
        for i in range(max_itr_num):
            self.pointer_network_layer1.append(PointerNet.PointerNet(input_dim = FEATURE_DIMENSION,
                                                                        embedding_dim = embedding_dimension,
                                                                        hidden_dim = hidden_dimension,
                                                                        lstm_layers = lstm_layers_num,
                                                                        dropout = 0,
                                                                        bidir=False))

        self.pointer_network_layer2 = PointerNet.PointerNet(input_dim = MAXIMUM_CLIENT_NUM_PLUS_ONE,
                                                                embedding_dim = embedding_dimension,
                                                                hidden_dim = hidden_dimension,
                                                                lstm_layers = lstm_layers_num,
                                                                dropout = 0,
                                                                bidir=False)

        self.fcc = nn.Linear(self.hidden_dim*(self.max_itr_num+1), self.hidden_dim)
        self.fcc.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(self.hidden_dim*(self.max_itr_num+1), self.hidden_dim)
        self.fca.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(self.hidden_dim, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, state, hidden_states_a):
        # print(s.shape)

        itr = torch.ones((1,state.shape[1],1))
        eof = torch.zeros((1,1,FEATURE_DIMENSION))
        if use_gpu:
            itr = itr.to(device) 
            eof = eof.to(device) 
        state_list = []
        for i in range(self.max_itr_num):
            # print(state.shape)
            # print(itr.shape)
            state_list.append(torch.cat((state, (i+1)*itr), dim = 2))
            

        outputs = []
        pointers = []
        hidden_states = []

        for j in range(self.max_itr_num):
            output, pointer, hidden_state = self.pointer_network_layer1[j](state_list[j])
            outputs.append(output[:,0,:])
            pointers.append(pointer)
            hidden_states.append(hidden_state)

        outputs = torch.cat(outputs, 0)
        outputs = torch.unsqueeze(outputs, dim = 0)

        output_60 = torch.zeros((1,self.max_itr_num,MAXIMUM_CLIENT_NUM_PLUS_ONE))
        if use_gpu:
            output_60 = output_60.to(device) 
        output_60[:, :, 0:outputs.size(2)] = outputs

        output2, pointer2, hidden_state2 = self.pointer_network_layer2(output_60)

        itr_num = pointer2[0, 0]  
        pointer = pointers[itr_num][0]

        hidden_states.append(hidden_state2)

        hidden_states = torch.cat(hidden_states, dim = 1)

        x1 = self.fcc(hidden_states)
        x2 = self.fca(hidden_states_a)
        x = F.relu(x1+x2)
        value = self.out(x)

        return value

class DDPG(object):
    def __init__(self, max_itr_num, embedding_dim, hidden_dim, lstm_layers):

        self.max_itr_num = max_itr_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        self.Actor_eval = ANet(self.max_itr_num, embedding_dim, hidden_dim, lstm_layers)
        self.Actor_target = ANet(self.max_itr_num, embedding_dim, hidden_dim, lstm_layers)
        self.Critic_eval = CNet(self.max_itr_num, embedding_dim, hidden_dim, lstm_layers)
        self.Critic_target = CNet(self.max_itr_num, embedding_dim, hidden_dim, lstm_layers)

        # self.Actor_eval.load_state_dict(torch.load('result/0512/0512_1015/Actor_eval.pkl'))
        # self.Actor_eval.eval()
        # self.Actor_target.load_state_dict(torch.load('result/0512/0512_1015/Actor_target.pkl'))
        # self.Actor_target.eval()
        # self.Critic_eval.load_state_dict(torch.load('result/0512/0512_1015/Critic_eval.pkl'))
        # self.Critic_eval.eval()
        # self.Critic_target.load_state_dict(torch.load('result/0512/0512_1015/Critic_target.pkl'))
        # self.Critic_target.eval()

        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr = LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr = LR_A)
        self.loss_td = nn.MSELoss()
        if use_gpu:
            self.Actor_eval = self.Actor_eval.to(device)
            self.Actor_target = self.Actor_target.to(device)
            self.Critic_eval = self.Critic_eval.to(device)
            self.Critic_target = self.Critic_target.to(device)
            for i in range(self.Actor_eval.max_itr_num):
                self.Actor_eval.pointer_network_layer1[i] = self.Actor_eval.pointer_network_layer1[i].to(device)
                self.Actor_target.pointer_network_layer1[i] = self.Actor_target.pointer_network_layer1[i].to(device)
                self.Critic_eval.pointer_network_layer1[i] = self.Critic_eval.pointer_network_layer1[i].to(device)
                self.Critic_target.pointer_network_layer1[i] = self.Critic_target.pointer_network_layer1[i].to(device)
            self.loss_td = self.loss_td.to(device)

        self.pointer = 0
        self.learn_time = 0
        self.memory = []

    def choose_action_withAmender(self, state):
        state = state.astype(np.float32)
        ss = torch.FloatTensor(state)

        if use_gpu:
            ss = ss.to(device)   
        # print(ss)
        itr_num, pointer, hidden_states = self.Actor_eval(ss)
        # print("MPN:")
        # print(itr_num)
        # print(pointer)
        if use_gpu:
            itr_num = itr_num.cpu()
            pointer = pointer.cpu()
            hidden_states = hidden_states.cpu()
        
        itr_num = itr_num.detach().numpy()+1
        pointer = pointer.detach().numpy()
        count = 0
        # print(pointer)
        for i in range(len(pointer)):
            if pointer[i] == len(pointer)-1:
                count = i
        if count == 0:
            pointer = []
        else:
            pointer = pointer[0: count]
        hidden_states = hidden_states.detach().numpy()
        # ================================================================================================
        # # Amender
        itr_num, pointer = Amender(itr_num, pointer, state)
        # ================================================================================================      
        
        return itr_num, pointer, hidden_states
        
    def choose_action(self, state):
        state = state.astype(np.float32)
        ss = torch.FloatTensor(state)

        if use_gpu:
            ss = ss.to(device)   
        # print(ss)
        itr_num, pointer, hidden_states = self.Actor_eval(ss)
        # print("MPN:")
        # print(itr_num)
        # print(pointer)
        if use_gpu:
            itr_num = itr_num.cpu()
            pointer = pointer.cpu()
            hidden_states = hidden_states.cpu()
        
        itr_num = itr_num.detach().numpy()+1
        pointer = pointer.detach().numpy()
        count = 0
        # print(pointer)
        for i in range(len(pointer)):
            if pointer[i] == len(pointer)-1:
                count = i
        if count == 0:
            pointer = []
        else:
            pointer = pointer[0: count]
        hidden_states = hidden_states.detach().numpy()
        # ================================================================================================
        # # Amender
        # itr_num, pointer = Amender(itr_num, pointer, state)
        # ================================================================================================      
        
        return itr_num, pointer, hidden_states
    def learn(self):

        self.learn_time += 1

        for target_param, param in zip(self.Critic_target.parameters(), self.Critic_eval.parameters()):
            target_param.data.copy_(target_param.data*(1.0 - TAU) + param.data*TAU)
        for target_param, param in zip(self.Actor_target.parameters(), self.Actor_eval.parameters()):
            target_param.data.copy_(target_param.data*(1.0 - TAU) + param.data*TAU)
        

        indices = np.random.randint(low = 0, high=MEMORY_CAPACITY)
        bt = self.memory[indices]
        # print(bt)
        bs = torch.FloatTensor(bt[0].astype(np.float32))
        bitr_num = torch.FloatTensor((np.expand_dims(bt[1][0], axis=0)).astype(np.float32))
        # bpointer = torch.FloatTensor(bt[1][1])
        bhidden_states = torch.FloatTensor(bt[1][2].astype(np.float32))
        br = torch.FloatTensor(bt[2])
        bs_ = torch.FloatTensor(bt[3].astype(np.float32))
        # print(br)

        # print('use_gpu = '+str(use_gpu))
        # if use_gpu:
        #     bt = bt.cuda()
        



        if use_gpu:
            bs = bs.to(device) 
            bitr_num = bitr_num.to(device) 
            # bpointer = bpointer.to(device) 
            bhidden_states = bhidden_states.to(device) 
            br = br.to(device) 
            bs_ = bs_.to(device) 

        # print(bs)
        # print(list(bs.size()))
        itr_num, pointer, hidden_states = self.Actor_eval(bs)
        q = self.Critic_eval(bs, hidden_states)
        
        loss_a = -torch.mean(q)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        itr_num_, pointer_, hidden_states_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_, hidden_states_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        # print(q_)
        # print(br)
        q_target = br+GAMMA*q_  # q_target = 负的
        q_v = self.Critic_eval(bs, bhidden_states)
        # print(q_target)
        # print(q_v)
        td_error = self.loss_td(q_target, q_v)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

        if use_gpu:
            loss_a = loss_a.cpu()
            td_error = td_error.cpu()

        return float(loss_a), float(td_error)


    def store_transition(self, s, a, r, s_):

        if self.pointer < MEMORY_CAPACITY:
            self.memory.append([s, a, r, s_])
        else:  
            index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
            self.memory[index] = [s, a, r, s_]
        self.pointer += 1 


    def save_model(self, path):
        if use_gpu:
            self.Actor_eval = self.Actor_eval.cpu()
            self.Actor_target = self.Actor_target.cpu()
            self.Critic_eval = self.Critic_eval.cpu()
            self.Critic_target = self.Critic_target.cpu()            
        torch.save(self.Actor_eval.state_dict(), path+'/Actor_eval.pkl')
        torch.save(self.Actor_target.state_dict(), path+'/Actor_target.pkl')
        torch.save(self.Critic_eval.state_dict(), path+'/Critic_eval.pkl')
        torch.save(self.Critic_target.state_dict(), path+'/Critic_target.pkl')
        if use_gpu:
            self.Actor_eval = self.Actor_eval.to(device) 
            self.Actor_target = self.Actor_target.to(device) 
            self.Critic_eval = self.Critic_eval.to(device) 
            self.Critic_target = self.Critic_target.to(device) 
