# newly added libraries, to record the time interval
import time
import copy
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
        # train and update
        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

        # initialize values
        time_start = float(time.time()) # record the start training time
        rho, beta = None, None # initialize with null values
        for epoch in range(local_iteration):
            # get data
            x, labels = next(iter(self.local_training_data))
            x, labels = x.to(self.device), labels.to(self.device)
            
            # get lasts
            if epoch == 0:
                net.eval()
                last_w = torch.cat([param.view(-1) for param in net.parameters()]) # get last weights
                last_loss = self.criterion(net(x), labels)
                last_loss.backward()
                last_loss = last_loss.item() # get last loss 
                last_grads = torch.cat([param.grad.view(-1) for param in net.parameters()]) # calculate grads.
            
            # get currents
            net.train()
            net.zero_grad()
            log_probs = net(x)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            loss = loss.item() # get current loss
            grads = torch.cat([param.grad.view(-1) for param in net.parameters()]) # get current grads
            optimizer.step() # updates weights
            w = torch.cat([param.view(-1) for param in net.parameters()]) # get current w
            logger.info("local client {} norm of current weights - last weights: {}".format(self.client_idx, torch.norm(w - last_w)))

            # calculate rho and update rho
            rho_tmp = abs(loss - last_loss) / torch.norm(w - last_w)
            if not rho or rho_tmp > rho:
                rho = rho_tmp
            
            # calculate beta and udpate beta
            beta_tmp = torch.norm(grads - last_grads) / torch.norm(w - last_w)
            if not beta or beta_tmp > beta:
                beta = beta_tmp

            # update last
            last_loss = loss
            last_w = w
            last_grads = grads
            
            logger.debug('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.client_idx, epoch, loss))

        # record the end time
        time_end = float(time.time()) 

        # add a new return value "time_interval"
        return net.cpu().state_dict(), loss, (time_end - time_start), beta.item(), rho.item()


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
            x, target = next(iter(test_data))
            x, target = x.to(self.device), target.to(self.device)
            pred = model_global(x)
            loss = self.criterion(pred, target)
    
            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()

            metrics['test_correct'] += correct.item()
            metrics['test_loss'] += loss.item() * target.size(0)
            metrics['test_total'] += target.size(0)

        return metrics
