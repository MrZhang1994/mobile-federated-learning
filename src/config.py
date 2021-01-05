import os
import logging
from datetime import datetime
import pandas as pd

# ===============================
# process the channel data
# ===============================
# set the data dir
CHANNEL_DATA_DIR = "../data"
START_DATE = 1001
DATE_LENGTH = 27
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

# initialize some csv_writers name
trainer_csv = os.path.join(RESULT_PATH, "-".join(["trainer", DAY, MOMENT]) + '.csv') # file name
scheduler_csv = os.path.join(RESULT_PATH, "-".join(["scheduler", DAY, MOMENT]) + '.csv')# file name
FPF_csv = os.path.join(RESULT_PATH, "-".join(["FPF", DAY, MOMENT]) + '.csv') # file name

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
RES_WEIGHT = 0.5
RES_RATIO = 0.1 # the ratio of radio_res
# set hyperparameter for calculating FPF2 index
G1 = 2
G2 = 2
# set the number of days until reinitialize the model
RESTART_DAYS = 3
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
MAXIMUM_CLIENT_NUM_PLUS_ONE = 100
EMBEDDING_DIMENSION = 16
HIDDEN_DIMENSION = 16
LSTM_LAYERS_NUM = 1
MAXIMUM_ITERATION_NUM = 4