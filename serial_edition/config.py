# ************************************************************************************************************ #
import logging
import pandas as pd
import os
from tensorboardX import SummaryWriter

# set the requirement.
bandwith = 1
res_weight = 0.5
res_ratio = 0.1 # the ratio of radio_res

# the ratio of standalone training time over real distributed learning training time.
timing_ratio = 1

# set the data dir
channel_data_dir = "../data"
# read channel data at once.
channel_data = pd.concat([pd.read_csv(os.path.join(channel_data_dir, csv_name)
				     # error_bad_lines=False
				     ) for csv_name in os.listdir(channel_data_dir)], ignore_index = True)

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

# set hyperparameter for calculating FPF2 index
G1 = 2
G2 = 2

# set the number of days until reinitialize the model
restart_days = 1
# ************************************************************************************************************ #


# ==========================
# Parameters for trainer
# ==========================
TIME_COMPRESSION_RATIO = 0.1


# Parameters for ddpg_mpn

# ==========================
# Parameters for ddpg
# ==========================
MEMORY_CAPACITY = 50 # size of experience pool
LR_A = 0.01         # learning rate for actor
LR_C = 0.001        # learning rate for critic
GAMMA = 0.9         # reward discount
TAU = 0.01          # soft replacement
use_gpu = False     # use GPU or not
AMEND_RATE = 0.2
# ==========================
# Parameters for multi-layer PointerNetwork
# ==========================
FEATURE_DIMENSION = 4
MAXIMUM_CLIENT_NUM_PLUS_ONE = 61
EMBEDDING_DIMENSION = 16
HIDDEN_DIMENSION = 16
LSTM_LAYERS_NUM = 1
MAXIMUM_ITERATION_NUM = 4