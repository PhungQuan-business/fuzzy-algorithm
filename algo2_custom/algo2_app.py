import numpy as np
import pandas as pd
from algo2 import IntuitiveFuzzy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import preprocessing
from tabulate import tabulate
import warnings
import os
import time
from sklearn.model_selection import KFold
import statistics
from operator import itemgetter
from utils_fuzzy import Logging
from operator import itemgetter

warnings.filterwarnings("ignore")
PATH = "/Users/phunghongquan/Documents/NCS-VietAnh/algorithm_custom/data/"
LOG_PATH = "logs"

arr_data = [
    ["movement_libras", [90], 0.000]  # 1 0.334
    # ["wall",[24], 0.01] #4# 0.05 38.948
    # ["ionosphere",[34], 0.01]  #2 0.09
    # ["mfeat",[76], 0.01] # 0.1 #10 56.116
    # ["Urban",[147], 0.01] # 9 4.347
    # ["waveform2",[40], 0.01] # 3
    # ["hill-valley",[100], 0.004] # 2 18.909
    # ["Pizza",[37], 0.01] #0.8
    # ["leaf", [15], 0.01] # 0.017

    # ["spectf",[44], 0.01] #2
    # ["cmc",[4,5,8,9], 0.01]
    # ["hill-valley",[100], 0.01] #0.05
    # ["mfeat",[76], 0.01] # 0.1
    # ["movement_libras",[90], 0.01] #0.6
    # ["waveform2",[40], 0.01] #0.4
    # ["MDL",[500], 0.01] #0.7
    # ["leukemia_4",[7129], 0.05] #0.2
    # ["ORL",[1024], 0.1] #0.3  #alpha remove = 0.01
    # ["Pizza",[37], 0.01] #0.8
    # ["piechart2", [36], 0.01]
    # ["vehicle", [18], 0.01]
    # ["winequality-red",[11], 0.01]
    # ["volkert", [100], 0.001]
    # ['vowel', [0, 1, 12], 0.001]
    # ["onehun", [64], 0.01]
    # ["gesture", [32], 0.01]
    # ["fri_c2_1000",[50] , 0.01]
    # ["climate", [0,20], 0.01]
    # ["heart",[1,2,5,6,8,10,11,12,13], 0.00]
    # ["synthetic", [61], 0.01]
    # ["CLL_SUB_111",[11340], 0.05] #1
    # ["pc4",[37], 0.001] #0.02
    # ["robot-failures",[90], 0.00]
    # ["forest", [27],0.001]
    # ["gesture",[32], 0.00]
    # ["glass",[9], 0.0]
    # ["robot-failures",[90], 0.01] #1
    # ["waveform",[21], 0.01] #0.55
    # ["thyroid",[21], 0.01]
    # ["PlanTexture", [64], 0.01]
    # ["kc1",[21], 0.01]
    # ["wdbc",[30], 0.01]
    # ["warpAR10P",[2400],0.05]
    # ["madelon",[500],0.01]
    # ["sonar",[60], 0.01]
    # ["ILPD",[1,10],0.01]
    # ["heart",[1,2,5,6,8,10,11,12,13],0.01]
    # ["wine",[13], 0.01]
    # ["pollution",[15],0.025]
    # ["pyrim",[27],0.01]
    # ["ionosphere",[34], 0.025]
    # ["pc4",[37]] 0.2
    # ["robot-failures",[90]] 0.1
    # ["Urban",[147], 0.01]
    # ["hill-valley",[100]] 0.2
    # ["tecator",[124],[0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75]]
    # ["lsvt",[310],[0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75]]
    # ["pd",[754], 0.025] #0.8
    # ["ORL",[1024], 0.1] #0.8
    # ["MDL",[500], 0.05] #0.7
    # ["leukemia",[7129],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]
    # ["waveform",[21], 0.01] #0.55
    # ["warpAR10P",[2400],[0.55,0.55,0.55,0.55,0.55,0.55,0.55,0.55,0.55,0.55]]
    # ["micro",[1300]]
    # ["qsar",[41], 0.01]
    # ["spambase",[57],[0.1,0.55,0.75,0.55,0.55,0.55,0.75,0.55,0.55,0.25]]
    # ["lung",[3312]]
    # ["lung_discrete",[325]]
    # ["SRBCT",[2308],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]
    # ["robot-failures",[90]]*
    # ["Image",[135,136,137,138,139],[0.1,0.25,0.25,0.25,0.1,0.1,0.1,0.55,0.1,0.25]]
    # ["texture",[40]]
    # ["scene",[294,295,296,297,298,299],[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]]
    # ["autouniv",[11,30,39,40], 0.0]
    # ["german",[0,2,3,5,6,8,9,11,13,14,16,18,19,20], 0.001]
    # ["vehicle",[18], 0.01]
    # ["seismic-bumps",[0,1,2,7,18],0.01]
    # ["waveform2",[40], 0.01] #0.4
    # ["cmc",[4,5,8,9], 0.01]
    # ["wall",[24]] 0.2
    # ["satimage",[36], 0.01]
    # ["ozone", [72], 0.05]
    # ["qsar",[41], [0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75]],# 0.8
    # ["segmentation",[19],0.01], #0.85
    # ["mfeat",[76], 0.01]
    # ["mfeat",[76]],
    # ["sick", [29]]
    # ["triazines", [60],0.01]
    # ["agnostic",[48], 0.01]
    # ["parkinsons", [22], 0.01]
    # ["oil", [49], 0.01]
    # ["pc3",[37], 0.01]
    # ["synthetic",[61], 0.01]
    # ["spectrometer",[101], 0.025]
    # ["Pizza",[37], 0.0]
    # ["piechart3",[37], 0.01]
    #  ["person",[321], 0.01]
    # ["Yale",[1024]]
    # ["spectf",[44], 0.01]
]
min_max_scaler = preprocessing.MinMaxScaler()


def preprocessing(name_file, att_nominal_cate):
    DS = np.genfromtxt(PATH + name_file + ".csv",
                       delimiter=",", dtype=object)[:, :]
    att = DS[0].astype(int)
    att_nominal_cate = np.array(att_nominal_cate)
    att_real = np.setdiff1d(att, att_nominal_cate)
    # encode decision variable, except the value in the first row
    for i in att_nominal_cate:
        DS[1:, i] = LabelEncoder().fit_transform(DS[1:, i])
    # if len(att_real) > 0 :
        # list_index_real = [list(DS[0]).index(i) for i in att_real]
    # transformation for every row except the first row
    DS[1:, att_real] = min_max_scaler.fit_transform(DS[1:, att_real])
    # return all except the first row
    return DS[1:]


def transform_array(arr):
    # Create a copy of the input array to avoid modifying the original
    transformed_arr = arr.copy()
    # Get the number of columns in the array
    num_cols = arr.shape[1]
    # Perform transformations on all columns except the last column
    for col in range(num_cols - 1):
        # Find the standard deviation of the column
        std_dev = np.std(arr[:, col])
        # Check if standard deviation is not zero
        if std_dev != 0:
            # Subtract the minimum value from each element in the column
            transformed_arr[:, col] -= np.min(arr[:, col])
            # Divide the result by the standard deviation
            transformed_arr[:, col] /= std_dev
            # Take the floor of the result
            transformed_arr[:, col] = np.floor(transformed_arr[:, col])
    # Perform MinMaxScaler on the last column
    transformed_arr[:, -1] = min_max_scaler.fit_transform(
        arr[:, -1].reshape(-1, 1)).flatten()
    return transformed_arr


def split_data(data, number: int = 1):
    if number == 1:
        return [data]
    ldt = len(data)
    spt = int(ldt / number)
    blk = spt * number
    arrs = np.split(data[:blk], number)
    if blk != ldt:
        arrs[-1] = np.vstack((arrs[-1], data[blk:]))
    return arrs


'''
chia thành 5 phần bằng nhau(xác định = number)
đẩy từng phần vào bảng tĩnh để chạy gia tăng
'''


def split_data_icr(data):
    arrs = []
    arrs_2 = split_data(data, number=2)
    arrs.append(arrs_2[0])
    arrs_2[1] = split_data(arrs_2[1], number=5)  # change as prefer
    for arr in arrs_2[1]:
        arrs.append(arr)
    return arrs


def main(arr_data):
    start = time.time()
    a_sc = [["Data", "|C|", "|R_F|", "Acc_O",
             "std_O", "Acc_F", "std_F", "T_F", "Reduct_F"]]
    n_steps = 6
    B = []
    # F = []
    num_prev = 0
    X = [0.]
    for arr in arr_data:
        # for x in X:
        F = []
        DS = preprocessing(arr[0], arr[1])
        st = time.time()
        DS = split_data_icr(DS)
        '''
        lấy 1 phần dữ liệu đã đi qua bước 
        tiền xử lí đầu tiên và 
        đem nó vào bước tiền xử lí tiếp theo
        '''
        # print(f'shape of original dataset:', DS[0].shape)
        DS_2 = transform_array(DS[0])
        IF = IntuitiveFuzzy(DS_2, arr[2])  # sửa lại input cho cái này
        F, time_filter = IF.filter()
        # print("F", F)

        # Evaluate trên dữ liệu chỉ đi qua bước tiền xử lí đầu tiên
        sc = IF.evaluate(arr[0], DS[0], F, time_filter)
        a_sc.append(sc)
        # os.system('cls')
        print(tabulate(a_sc, headers='firstrow',
              tablefmt='pipe', stralign='center'))
        U = DS[0]
        for i in range(1, n_steps):
            dU = DS[i]
            U = np.vstack((U, dU))
            U_2 = transform_array(U)
            # print(f'this is new F', F)
            IF = IntuitiveFuzzy(U_2, arr[2], F)
            F, time_filter = IF.filter()
            sc = IF.evaluate(arr[0], U, F, time_filter)
            a_sc.append(sc)
            # os.system('cls')
            print(tabulate(a_sc, headers='firstrow',
                           tablefmt='pipe', stralign='center'))

    print(time.time()-start)


if __name__ == "__main__":
    main(arr_data)
