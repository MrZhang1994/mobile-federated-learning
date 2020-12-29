# newly added libarary, to insert delay to the program.
# system python package load.
import argparse
import logging
import os
import sys
import time
import random

# Maching learning tool chain.
import numpy as np
import torch
import wandb

from fedavg_trainer import FedAvgTrainer
from config import *

# add the root dir of FedML
sys.path.insert(0, os.path.abspath("/zzp/FedML")) 

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
from fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from fedml_api.data_preprocessing.stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from fedml_api.data_preprocessing.stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist

from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.cv.cnn import CNN_DropOut
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow

from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.cv.resnet_gn import resnet18


def add_args():
    """
    return a parser added with args required by fit
    """
    parser = argparse.ArgumentParser(description='FedAvg-standalone')

    # Training settings
    parser.add_argument('--model', type=str, default='resnet56', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--comm_round', type=int, default=1000,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=25,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--seed', type=int, default=0,
                        help='the random seed')
                        
    # set if using debug mod
    parser.add_argument("-v", "--verbose", action= "store_true", dest= "verbose", 
                        help= "enable debug info output")
    # set the scheduler method
    """
    currently only 1. sch_mpn 2. sch_mpn_empty 3. sch_random
                    4. sch_channel 5. sch_rrobin 6. sch_loss are supported.
    sch_mpn_empty means sch_mpn without training.
    """         
    parser.add_argument("--method", type= str, default="sch_random",
                        help="declare the benchmark methods you use") 
    # set if full batch
    parser.add_argument("-f", "--full_batch", action= "store_true", dest= "full_batch", 
                        help="set if use full batch") 

    args = parser.parse_args()
    return args

def load_data(args, dataset_name):
    """
    args: dict containing all program arguments.
    dataset_name: the name of the dataset like MNIST.
    return: dataset which contains [client_num, train_data_num, test_data_num, train_data_global, test_data_global,
                                    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    """
    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0 or args.full_batch:
        full_batch = True
        args.batch_size = 128 # temporary batch size
    else:
        full_batch = False

    if dataset_name == "mnist":
        logger.debug("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_mnist(args.batch_size)

    elif dataset_name == "femnist":
        logger.debug("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir)

    elif dataset_name == "shakespeare":
        logger.debug("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)

    elif dataset_name == "fed_shakespeare":
        logger.debug("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_shakespeare(args.dataset, args.data_dir)

    elif dataset_name == "fed_cifar100":
        logger.debug("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_cifar100(args.dataset, args.data_dir)

    elif dataset_name == "stackoverflow_lr":
        logger.debug("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_lr(args.dataset, args.data_dir)

    elif dataset_name == "stackoverflow_nwp":
        logger.debug("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_stackoverflow_nwp(args.dataset, args.data_dir)
        
    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            data_loader = load_partition_data_cifar10
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, client_num_in_total, args.batch_size)

    if full_batch:
        logger.info("-------------batches combine------------")
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [client_num, train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset


def combine_batches(batches):
    """
    batches: list containing (batched_x, batched_y)
    return: combined batches or called full batch.
    """
    if isinstance(batches, list):
        full_x = torch.cat([batch[0] for batch in batches], 0)
        full_y = torch.cat([batch[1] for batch in batches], 0)
    else:
        batches = torch.utils.data.DataLoader(batches.dataset, batch_size = len(batches.dataset))
        full_x, full_y = next(iter(batches))
    return [(full_x, full_y)]


def create_model(args, model_name, output_dim):
    """
    args: dict containing all program arguments.
    model_name: the name of the model, like CNN.
    output_dim: the dimension of the output of the model.
    return: model.
    """
    logger.debug("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        logger.debug("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "cnn" and args.dataset == "femnist":
        logger.debug("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logger.debug("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logger.debug("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logger.debug("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logger.debug("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim) 
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logger.debug("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    return model


def main():
    # get all the program arguments.
    args = add_args()

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # set if use logging debug version.
    logger.setLevel(logging.DEBUG)
    logger_sch.setLevel(logging.DEBUG)
    if not args.verbose:
        logger.setLevel(logging.INFO)
        logger_sch.setLevel(logging.INFO)
    logger.debug("--------DEBUG enviroment start---------")
    
    # show the upate information
    logger.debug("-------global parameters setting-------")
    logger.debug("RES_WEIGHT: {}".format(RES_WEIGHT))
    logger.debug("RES_RATIO: {}".format(RES_RATIO))
    logger.debug("G1: {}".format(G1))
    logger.debug("G2: {}".format(G2))
    logger.debug("RESTART_DAYS: {}".format(RESTART_DAYS))
    logger.debug("DATE_LENGTH {}".format(DATE_LENGTH))
    logger.debug("TIME_COMPRESSION_RATIO: {}".format(TIME_COMPRESSION_RATIO))

    logger.debug("------ordinary parameter setting-------")
    logger.info(args)

    # show other information
    logger.debug("-----------Other information-----------")
    logger.debug("CHANNEL_DATA_DIR {}".format(CHANNEL_DATA_DIR))
    logger.debug("client_num_in_total: {}".format(client_num_in_total))
    logger.debug("client_num_in_total: {}".format(client_num_per_round))

    logger.debug("---------cuda device setting-----------")
    if torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            logger.error("CUDA error, invalid device ordinal")
            exit(1)
    else:
        logger.error("Plz choose other machine with GPU to run the program")
        exit(1)
    device = torch.device("cuda:" + str(args.gpu))
    logger.debug(device)

    # load data
    logger.info("-------------dataset loading------------")
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    logger.debug("-------------model setting-------------")
    model = create_model(args, model_name=args.model, output_dim=dataset[-1])
    logger.debug(model)

    # initialize the wandb.
    wandb.init(
        project="fedavg",
        name="FedAVG-" + str(args.method)[4:] + "-r" + str(args.comm_round) + "-lr" + str(args.lr),
        config=args
    )

    logger.debug("------------finish setting-------------")

    trainer = FedAvgTrainer(dataset, model, device, args)
    trainer.train()

if __name__ == "__main__":
    main()