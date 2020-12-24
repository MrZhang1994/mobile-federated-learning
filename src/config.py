import logging
import pandas as pd
import os
from tensorboardX import SummaryWriter

# set the data dir
channel_data_dir = "../data"
date_length = 10
# read channel data at once.
channel_data = pd.concat([pd.read_csv(channel_data_dir+'/'+str(csv_name)+'.csv'
                     # error_bad_lines=False
                     ) for csv_name in range(1001, 1001+date_length)], ignore_index = True)
# restrict the number of client_num_in_total to maxmium car ID + 1
client_num_in_total = channel_data['Car'].max() + 1
client_num_per_round = 100 # number of local clients

# set the logger
logging.basicConfig(
                    # filename = "logfile",
                    # filemode = "w+",
                    format='%(name)s %(levelname)s %(message)s',
                    datefmt = "%H:%M:%S",
                    level=logging.DEBUG)
logger = logging.getLogger("training")

# setup the tensorboardX
boardX = SummaryWriter(comment="-fedavg")

# ===============================
# set hyperparameters for Trainer
# ===============================
# set the requirement.
RES_WEIGHT = 0.5
RES_RATIO = 0.1 # the ratio of radio_res
# set hyperparameter for calculating FPF2 index
G1 = 2
G2 = 2
# set the number of days until reinitialize the model
RESTART_DAYS = 1
# set the speed of the time_counter's increasing.
TIME_COMPRESSION_RATIO = 0.1

# ==========================
# Parameters for ddpg
# ==========================
MEMORY_CAPACITY = 1 # size of experience pool
LR_A = 0.01         # learning rate for actor
LR_C = 0.001        # learning rate for critic
GAMMA = 0.9         # reward discount
TAU = 0.01          # soft replacement
use_gpu = False      # use GPU or not
AMEND_RATE = 1
NAIVE_PG = True

# ==========================
# Parameters for multi-layer PointerNetwork
# ==========================
FEATURE_DIMENSION = 4
MAXIMUM_CLIENT_NUM_PLUS_ONE = 61
EMBEDDING_DIMENSION = 16
HIDDEN_DIMENSION = 16
LSTM_LAYERS_NUM = 1
MAXIMUM_ITERATION_NUM = 4