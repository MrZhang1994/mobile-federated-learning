import os
import logging
from datetime import datetime
import pandas as pd

# ===============================
# process the channel data
# ===============================
# set the data dir
CHANNEL_DATA_DIR = "../data"
START_DATE = 1008
DATE_LENGTH = 20
# read channel data at once.
channel_data = pd.concat([pd.read_csv(CHANNEL_DATA_DIR+'/'+str(csv_name)+'.csv'
                         # error_bad_lines=False
                         ) for csv_name in range(START_DATE, START_DATE+DATE_LENGTH)], ignore_index = True)
time_cnt_max = [pd.read_csv(os.path.join(CHANNEL_DATA_DIR, str(csv_name)+'.csv'))["Time"].max() for csv_name in range(START_DATE, START_DATE+DATE_LENGTH)] 
# restrict the number of client_num_in_total to maxmium car ID + 1
client_num_in_total = channel_data['Car'].max() + 1
client_num_per_round = 100 # number of local clients

# ===============================
# set up threads for numpy
# ===============================
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

# ===============================
# initialize some csv_writers name
# ===============================
trainer_csv = os.path.join(RESULT_PATH, "-".join(["trainer", DAY, MOMENT]) + '.csv') # file name
scheduler_csv = os.path.join(RESULT_PATH, "-".join(["scheduler", DAY, MOMENT]) + '.csv')# file name
FPF_csv = os.path.join(RESULT_PATH, "-".join(["FPF", DAY, MOMENT]) + '.csv') # file name
reward_csv = os.path.join(RESULT_PATH, "-".join(["reward", DAY, MOMENT]) + '.csv') # file name

# ===============================
# set the logger
# ===============================
logging.basicConfig(
                    filename = os.path.join(RESULT_PATH, "-".join(["logfile", DAY, MOMENT]) + ".txt"),
                    filemode = "w+",
                    format='%(name)s %(levelname)s %(message)s',
                    datefmt = "%H:%M:%S",
                    level=logging.DEBUG)
# define a Handler which writes DEBUG messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger().addHandler(console)
logger = logging.getLogger("training")
logger_sch = logging.getLogger('schedule')

# ===============================
# set hyperparameters for Trainer
# ===============================
# set the requirement.
RES_WEIGHT = 0.5 # weight
RES_RATIO = 0.1 # 1 / Bandwith
# set hyperparameter for calculating FPF2 index
G1 = 2
G2 = 2
# set the number of days until reinitialize the model
RESTART_DAYS = 3
# set the speed of the time_counter's increasing.
TIME_COMPRESSION_RATIO = 0.1
# set if evaluate on train dataset
EVAL_ON_TRAIN = True
# set the threshold value for weight size
THRESHOLD_WEIGHT_SIZE = 100000
# set the threshold value for rho and beta
THRESHOLD_RHO = 1000
THRESHOLD_BETA = 1000
# set the threshold ratio for gradients
THRESHOLD_GRADS_RATIO = 50
# set the contant for time of training.
LOCAL_TRAINING_TIME = 1
# set the class num for each local dataset
CLASS_NUM = 3

# ===============================
# set hyperparameters for calculate iteration num
# ===============================
# set xi
XI = 0.999
# initialize ETA
ETA = None

## for cnn + FederatedMNIST
# set epsilon
EPSILON = 0.1509021520614624
# set KAI
KAI = 0.00083674144

# ==========================
# Parameters for rl
# ==========================
device_No = None
PROJECT = 'fedavg_rl4'
RL_PRESET = os.environ.get('RL_PRESET', 'pg_noamender')
assert RL_PRESET in ['ac', 'pg', 'random', 'ac_baseline', 'pg_amender', 'pg_noamender']
LR_A = 0.001         # learning rate for actor
LR_C = 0.001        # learning rate for critic
GAMMA = 0.9         # reward discount
# TAU = 0.01          # soft replacement
use_gpu = False      # use GPU or not
AMEND_RATE = 1
REG_FACTOR = 0.001
USE_AC = 'ac' in RL_PRESET
MEMORY_CAPACITY = 16               # size of experience pool
# AMEND_ITER = 100 if 'amender' not in RL_PRESET else 1e12
# if 'noamender' in RL_PRESET:
#     AMEND_ITER = 0
AMEND_ITER = 100
RL_UNIFORM = 'random' in RL_PRESET
DONT_TRAIN = 'random' in RL_PRESET or 'baseline' in RL_PRESET

# ==========================
# Parameters for multi-layer PointerNetwork
# ==========================
FEATURE_DIMENSION = 3
MAXIMUM_CLIENT_NUM_PLUS_ONE = 100
EMBEDDING_DIMENSION = 16
HIDDEN_DIMENSION = 16
LSTM_LAYERS_NUM = 1
MAXIMUM_ITERATION_NUM = 20


FAIRNESS_MULTIPLIER = None