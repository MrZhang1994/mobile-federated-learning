# # Libs used
import numpy as np
# import pandas as pd
# import os, shutil
# import copy
# import csv
# import random
# import socket

# # Modules used
# import ddpg_mpn

import csv



def process(Date):
    time_pointer = -1
    line = 0
    f = open('data/'+str(Date)+'.csv', 'r')
    csvreader = csv.reader(f)
    car_info = list(csvreader)[1:]
    for i in range(len(car_info)):
        for j in range(len(car_info[i])):
            car_info[i][j] = float(car_info[i][j])
    # car_info = np.array(car_info)
        # print(self.car_info)
    Car_info = {}
    for i in range(len(car_info)):
        if car_info[i][0] not in Car_info.keys():
            Car_info.update({car_info[i][0]:[car_info[i]]})
        else:
            Car_info[car_info[i][0]].append(car_info[i])
    # print(Car_info)
    return(Car_info)
    
def process2(Car_info):
    car_info2 = {}
    count = 0
    for i in range(4320):
        if i in Car_info.keys():
            car_info2.update({count:Car_info[i]})
            count = count+1
    return car_info2
for Date in range(1001, 1011):
    print(Date)
    Car_info = process(Date)
    Car_info  = process2(Car_info)
    # print(Car_info)
    # break
    f = open('data/Car_info_'+str(Date)+'.txt','w')   #把字典存在txt文件里
    f.write(str(Car_info))
    f.close()