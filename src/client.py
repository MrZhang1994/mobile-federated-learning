# newly added libraries, to record the time interval
import time
import torch
from torch import nn

from config import logger

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logger.debug("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device

        '''
        stackoverflow_lr is the task of multi-label classification
        please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
        https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
        https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
        '''
        if self.args.dataset == "stackoverflow_lr":
            self.criterion = nn.BCELoss(reduction = 'sum').to(device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(device)


    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number


    def get_sample_number(self):
        return self.local_sample_number


    def train(self, net, local_iteration): # add a new parameter "local_iteration".
        net.train()
        # train and update
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

        # record the start training time
        time_start = float(time.time()) 
        for epoch in range(local_iteration):
            x, labels = next(iter(self.local_training_data))
            x, labels = x.to(self.device), labels.to(self.device)
            net.zero_grad()
            log_probs = net(x)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss = loss.item()
            logger.debug('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.client_idx, epoch, epoch_loss))

        # record the end time
        time_end = float(time.time()) 

        # calculate grads.
        net.eval()
        log_probs = net(x)
        loss = self.criterion(log_probs, labels)
        loss.backward()
        grads = []
        for param in net.parameters():
            grads.append(param.grad.view(-1))
        grads = torch.cat(grads)

        # add a new return value "time_interval"
        return net.cpu().state_dict(), epoch_loss, (time_end - time_start), grads.cpu()


    def local_test(self, model_global, b_use_test_dataset=False):
        model_global.eval()
        model_global.to(self.device)
        metrics = { 
            'test_correct': 0, 
            'test_loss' : 0, 
            'test_total' : 0
        }
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)

                target = target.to(self.device)
                pred = model_global(x)
                loss = self.criterion(pred, target)
        
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        return metrics
