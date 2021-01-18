# newly added libraries, to record the time interval
import copy
import torch
from torch import nn

from config import logger, THRESHOLD_GRADS_RATIO

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logger.debug("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device

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
        rho, beta = None, None # initialize with null values
        # get data
        x, labels = next(iter(self.local_training_data))
        x, labels = x.to(self.device), labels.to(self.device) 
        # get lasts
        net.eval()
        last_w = torch.cat([param.view(-1) for param in net.parameters()]) # get last weights
        last_loss = self.criterion(net(x), labels)
        last_loss.backward()
        last_loss = last_loss.item() # get last loss 
        last_grads = torch.cat([param.grad.view(-1) for param in net.parameters()]) # calculate grads.   
        for epoch in range(local_iteration):
            # get currents
            net.train()
            net.zero_grad()
            log_probs = net(x)
            loss = self.criterion(log_probs, labels)
            loss.backward()
            grads = torch.cat([param.grad.view(-1) for param in net.parameters()]) # get current grads
            # if grads meets something strange, we will terminate the training process.
            if torch.isnan(grads).sum() > 0 or torch.isnan(loss) or torch.norm(grads) > self.args.lr * THRESHOLD_GRADS_RATIO * torch.norm(last_w):
                logger.warning("grads {} too large than weights {} or meets nan with epoch {}".format(torch.norm(grads), torch.norm(last_w), epoch))
                return net.cpu().state_dict(), None, None, None, None
            optimizer.step() # updates weights
            loss = loss.item() # get current loss
            w = torch.cat([param.view(-1) for param in net.parameters()]) # get current w
            # calculate rho and update rho
            rho_tmp = abs(loss - last_loss) / torch.norm(w - last_w)
            if not rho or rho_tmp > rho:
                rho = rho_tmp   
            # calculate beta and udpate beta
            beta_tmp = torch.norm(grads - last_grads) / torch.norm(w - last_w)
            if not beta or beta_tmp > beta:
                beta = beta_tmp
            # update last
            last_loss, last_w, last_grads = loss, w, grads
            
            logger.debug('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.client_idx, epoch, loss))

        # get acc
        _, predicted = torch.max(log_probs, -1)
        correct = predicted.eq(labels).sum()

        return net.cpu().state_dict(), loss, beta.item(), rho.item(), correct.item() / labels.size(0)


    def local_test(self, model_global, b_use_test_dataset=False):
        model_global.eval()
        model_global.to(self.device)
        metrics = dict()
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

            metrics['test_correct'] = correct.item()
            metrics['test_loss'] = loss.item() * target.size(0)
            metrics['test_total'] = target.size(0)

        return metrics
