import os
import csv
import logging
from datetime import datetime
import pandas as pd
from tensorboardX import SummaryWriter

# set the data dir
CHANNEL_DATA_DIR = "../data"
DATE_LENGTH = 10
# read channel data at once.
channel_data = pd.concat([pd.read_csv(CHANNEL_DATA_DIR+'/'+str(csv_name)+'.csv'
                         # error_bad_lines=False
                         ) for csv_name in range(1001, 1001+DATE_LENGTH)], ignore_index = True)
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
boardX = SummaryWriter(comment="fedavg")

# set up threads for numpy
os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
os.environ['NUMEXPR_NUM_THREADS'] = str(round(os.cpu_count() / 2))

# ===============================
# store results for training process
# ===============================
# setup directories for store paritial results
DAY =  str(datetime.now().month).zfill(2)+str(datetime.now().day).zfill(2)
MOMENT = str(datetime.now().hour).zfill(2)+str(datetime.now().minute).zfill(2)
RESULT_PATH = os.path.join("result", DAY, MOMENT)
if os.path.exists(RESULT_PATH) == False:
    os.makedirs(RESULT_PATH)

# initialize some csv_writers
csv_writer1 = csv.writer(open(
    os.path.join(RESULT_PATH, "-".join(["trainer", DAY, MOMENT]) + '.csv'), # file name
    'w+', # write mode
    encoding='utf-8', # encode mode 
    newline='')
    )
csv_writer1.writerow(['round index',
                      'time counter',
                      'client index',
                      'train time', 
                      'fairness', 
                      'local loss', 
                      'global loss', 
                      'test accuracy'])

csv_writer2 = csv.writer(open(
    os.path.join(RESULT_PATH, "-".join(["scheduler", DAY, MOMENT]) + '.csv'), # file name
    'w+', # write mode
    encoding='utf-8', # encode mode 
    newline='')
    )

csv_writer3 = csv.writer(open(
    os.path.join(RESULT_PATH, "-".join(["FPF", DAY, MOMENT]) + '.csv'), # file name
    'w+', # write mode
    encoding='utf-8', # encode mode 
    newline='')
    )
csv_writer3.writerow(['time counter'] + ["car_"+str(i) for i in range(client_num_in_total)])

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
MEMORY_CAPACITY = 50 # size of experience pool
LR_A = 0.01         # learning rate for actor
LR_C = 0.001        # learning rate for critic
GAMMA = 0.9         # reward discount
TAU = 0.01          # soft replacement
use_gpu = False      # use GPU or not
AMEND_RATE = 1

# ==========================
# Parameters for multi-layer PointerNetwork
# ==========================
FEATURE_DIMENSION = 4
MAXIMUM_CLIENT_NUM_PLUS_ONE = 80
EMBEDDING_DIMENSION = 16
HIDDEN_DIMENSION = 16
LSTM_LAYERS_NUM = 1
MAXIMUM_ITERATION_NUM = 4