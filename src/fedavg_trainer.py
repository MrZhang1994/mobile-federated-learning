# newly added libraries
import copy
import wandb
import time
import math
import csv
from tqdm import tqdm

import torch
import numpy as np
import multiprocessing as mp

from client import Client
from config import *
import scheduler as sch

class FedAvgTrainer(object):
    def __init__(self, dataset, model, device, args):
        self.device = device
        self.args = args
        
        [client_num, train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        # record the client number of the dataset
        self.client_num = client_num 
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict)
        # time counter starts from the first line
        self.time_counter = channel_data['Time'][0]
        self.cum_time = self.time_counter

        self.model_global = model
        self.model_global.train()

        # initialize the scheduler function
        if self.args.method == "sch_mpn" or self.args.method == "sch_mpn_empty":
            for _ in range(100):
                self.scheduler = sch.Scheduler_MPN()
                client_indexes, local_itr = self.scheduler.sch_mpn_test(1, 2002)
                if len(client_indexes) > 5:
                    break
        elif self.args.method == "sch_random":
            self.scheduler = sch.sch_random
        elif self.args.method == "sch_channel":
            self.scheduler = sch.sch_channel
        elif self.args.method == "sch_rrobin":
            self.scheduler = sch.sch_rrobin
        elif self.args.method == "sch_loss":
            self.scheduler = sch.sch_loss
        else:
            self.scheduler = sch.sch_random
 
 
    def setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict):
        logger.debug("############setup_clients (START)#############")
        for client_idx in range(client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device)
            self.client_list.append(c)
        logger.debug("############setup_clients (END)#############")


    def train(self):
        """
        Global initialized values
        """
        # maintain a lst for local losses
        local_loss_lst = np.zeros((1, client_num_in_total)) 
        # counting days
        counting_days = 0
        # Initialize A for calculating FPF2 index
        w_diff_mat = torch.zeros((len(self.model_global.state_dict().keys()), int(client_num_in_total))).to(self.device)
        local_itr_lst = torch.zeros(self.args.comm_round, int(client_num_in_total)).to(self.device)  # historical local iterations.
        A_mat = dict() 
        for para in self.model_global.state_dict().keys():
            weight_shape = self.model_global.state_dict()[para].numpy().ravel().shape[0]
            A_mat[para] = torch.ones(weight_shape).to(self.device) # initial the value of A with zero.
        G_mat = torch.zeros((1, int(client_num_in_total))).to(self.device) # initial the value of G with zero
        """
        starts training, entering the loop of command round.
        """
        for round_idx in range(self.args.comm_round):
            logger.info("################Communication round : {}".format(round_idx))
            logger.info("time_counter: {}".format(self.time_counter))
   
            self.model_global.train()
            
            # get client_indexes from scheduler
            reward = 0
            if self.args.method == "sch_mpn" or self.args.method == "sch_mpn_empty":
                if self.args.method == "sch_mpn_empty" or round_idx == 0:
                    client_indexes, local_itr = self.scheduler.sch_mpn_empty(round_idx, self.time_counter)
                else:
                    client_indexes, local_itr, reward = self.scheduler.sch_mpn(round_idx, self.time_counter, loss_locals, FPF2_idx_lst[0], local_loss_lst)
            else:
                client_indexes, local_itr = self.scheduler(round_idx, self.time_counter)
                # write to the scheduler csv
                with open(scheduler_csv, mode = "a+", encoding='utf-8', newline='') as file:
                    csv_writer = csv.writer(file)
                    if round_idx == 0:
                        csv_writer.writerow(['time counter', 'client index', 'iteration'])
                    csv_writer.writerow([self.time_counter, str(client_indexes), local_itr])
                    file.flush()
            logger.info("client_indexes = " + str(client_indexes))
            
            # write one line to trainer_csv
            trainer_csv_line = [round_idx, self.time_counter, str(client_indexes)]
            
            # Update local_itr_lst
            if client_indexes and local_itr > 0: # only if client_idx is not empty and local_iter > 0, then I will update following values
                local_itr_lst[round_idx, list(client_indexes)] = float(local_itr)

            # contribute to time counter
            self.tx_time(client_indexes) # transmit time
            
            # store the last model's training parameters.
            last_w = self.model_global.cpu().state_dict() 
            # local Initialization
            w_locals, loss_locals, time_interval_lst, loss_list = [], [], [], []
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            for idx in range(len(client_indexes)):
                # update dataset
                client = self.client_list[idx]
                client_idx = client_indexes[idx]
                dataset_idx = client_idx % self.client_num
                client.update_local_dataset(dataset_idx, self.train_data_local_dict[dataset_idx],
                                            self.test_data_local_dict[dataset_idx],
                                            self.train_data_local_num_dict[dataset_idx])

                # train on new dataset
                # add a new parameter "local_itr" to the funciton "client.train()"
                # add a new return value "time_interval" which is the time consumed for training model in client.
                w, loss, time_interval = client.train(net=copy.deepcopy(self.model_global).to(self.device), local_iteration = local_itr)
                
                # record current time interval into time_interval_lst
                time_interval_lst.append(time_interval)
                # record current w into w_locals
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                # record current loss into loss_locals
                loss_locals.append(loss)
                # update the local_loss_lst
                local_loss_lst[0, client_idx] = loss
                # update w_diff_mat
                for i, para in enumerate(self.model_global.state_dict().keys()):
                    w_diff_mat[i, idx] = torch.norm((w[para].to(self.device).reshape((-1, )) - last_w[para].to(self.device).reshape((-1, )))  * A_mat[para])

                # loss 
                logger.info('Client {:3d}, loss {:.3f}'.format(client_idx, loss))
                loss_list.append(loss)

            # update FPF index list
            FPF2_idx_lst = (w_diff_mat.sum(dim = 0) / G_mat).cpu().numpy()
            FPF2_idx_lst[np.bitwise_or(np.isnan(FPF2_idx_lst), np.isinf(FPF2_idx_lst))] = 0
            
            # write FPF index list to csv
            with open(FPF_csv, mode = "a+", encoding='utf-8', newline='') as file:
                csv_writer = csv.writer(file)
                if round_idx == 0:
                    csv_writer.writerow(['time counter'] + ["car_"+str(i) for i in range(client_num_in_total)])
                csv_writer.writerow([self.time_counter]+FPF2_idx_lst[0].tolist())
                file.flush()

            # update global weights
            w_glob = self.aggregate(w_locals)
            # copy weight to net_glob
            self.model_global.load_state_dict(w_glob)
            
           # update A_mat
            for para in w_glob.keys():
                A_mat[para] = A_mat[para] * (1 - 1/G2) + (w_glob[para].to(self.device).reshape((-1, )) - last_w[para].to(self.device).reshape((-1, ))) / G2
            # update G_mat
            G_mat = G_mat * (1 - 1 / G1) + local_itr_lst.sum(dim=0).reshape((1, -1)) / G1

            # update the time counter
            if time_interval_lst:
                self.time_counter += math.ceil(TIME_COMPRESSION_RATIO*(sum(time_interval_lst) / len(time_interval_lst)))
                self.cum_time += math.ceil(TIME_COMPRESSION_RATIO*(sum(time_interval_lst) / len(time_interval_lst)))
            logger.debug("time_counter after training: {}".format(self.time_counter))
            
            trainer_csv_line += [self.time_counter-trainer_csv_line[1], np.var(local_loss_lst), str(loss_list)]
            
            # if current time_counter has exceed the channel table, I will simply stop early
            if self.time_counter >= channel_data["Time"].max():
                logger.info("################schedualing restarts")
                if counting_days == RESTART_DAYS:
                    for key in w_glob.keys():
                        w_glob[key] = torch.rand(w_glob[key].size())
                    counting_days = 0
                else:
                    
                    counting_days += 1                
                self.time_counter = 0
            # set the time_counter 
            self.time_counter = np.array(channel_data['Time'][channel_data['Time'] > self.time_counter])[0]
            
            # print loss
            if not loss_locals:
                logger.info('Round {:3d}, Average loss None'.format(round_idx))
                
                trainer_csv_line.append('None')
            else:
                loss_avg = sum(loss_locals) / len(loss_locals)
                logger.info('Round {:3d}, Average loss {:.3f}'.format(round_idx, loss_avg))
                
                trainer_csv_line.append(loss_avg)

            if round_idx and round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
                test_acc = self.local_test_on_all_clients(self.model_global, round_idx)
                
                trainer_csv_line.append(test_acc)
            
            # write headers for csv
            with open(trainer_csv, mode = "a+", encoding='utf-8', newline='') as file:
                csv_writer = csv.writer(file)
                if round_idx == 0:
                    csv_writer.writerow(['round index', 'time counter', 'client index', 'train time', 'fairness', 'local loss', 'global loss', 'test accuracy'])
                csv_writer.writerow(trainer_csv_line)
                file.flush()

            wandb.log({
                "reward": reward,
                "round": round_idx,
                "cum_time": self.cum_time,
                "local_itr": local_itr,
                "client_num": len(client_indexes)
            })
           
    def tx_time(self, client_indexes):
        if not client_indexes:
            self.time_counter += 1
            self.cum_time += 1
            return 
        # read the channel condition for corresponding cars.
        channel_res = np.reshape(np.array(channel_data[channel_data['Time'] == self.time_counter * channel_data['Car'].isin(client_indexes)]["Distance to BS(4982,905)"]), (1, -1))
        logger.debug("channel_res: {}".format(channel_res))

        # linearly resolve the optimazation problem
        tmp_t = 1
        while np.sum(RES_WEIGHT * channel_res * RES_RATIO / tmp_t) > 1:
            tmp_t += 1

        # self.time_counter += tmp_t
        self.time_counter += math.ceil(TIME_COMPRESSION_RATIO*tmp_t)
        self.cum_time += math.ceil(TIME_COMPRESSION_RATIO*tmp_t)

        logger.debug("time_counter after tx_time: {}".format(self.time_counter))


    def aggregate(self, w_locals):
        if not w_locals:
            return self.model_global.cpu().state_dict()
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params


    def local_test_on_all_clients(self, model_global, round_idx, eval_on_train=False):
        logger.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        for client_idx in tqdm(range(min(int(client_num_in_total), self.client_num))):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            if eval_on_train:
                train_local_metrics = client.local_test(model_global, False)
                train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
                train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
                train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(model_global, True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

        if eval_on_train:
            # test on training dataset
            train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
            train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

            stats = {'training_acc': train_acc, 'training_loss': train_loss}
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            logger.info(stats)

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        test_loss_var = np.var(test_metrics['losses'])
        test_acc_var = np.var(np.array(test_metrics['num_correct']) / np.array(test_metrics['num_samples']))

        stats = {
            "Test/Acc": test_acc,
            "Test/Loss": test_loss,
            "Test/AccVar": test_acc_var,
            "Test/LossVar": test_loss_var,
            "round": round_idx,
            "cum_time": self.cum_time,
        }
        logger.info(stats)
        wandb.log(stats)
