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
    ["movement_libras", [90], 0.001]  # 1
    # ["wall",[24], 0.01] #4#
    # ["ionosphere",[34], 0.01]  #2
    # ["mfeat",[76], 0.01] # 0.1 #10
    # ["Urban",[147], 0.0005] # 9
    # ["waveform2",[40], 0.01] # 3
    # ["hill-valley",[100], 0.001] # 2
    # ["Pizza",[37], 0.01] #0.8
    # ["leaf", [15], 0.001]

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
    # ["vehicle",[18], 0.01]
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
min_max_scaler = MinMaxScaler()


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


def algo2_preprocessing(dataset):
    '''
    với attr_real thì dùng công thức sau
    với mỗi cột:
    -   tính std của cột
    -   giá trị mới = (giá trị vị trí hiện tại - min cột)/std
    '''

    features = dataset[:, :-1]
    decision_variable = dataset[:, -1].reshape(-1, 1)
    for col in range(features.shape[1]):
        column_values = features[:, col]
        std_dev = np.std(column_values)
        col_min = np.min(column_values)
        if std_dev != 0:  # To avoid division by zero
            features[:, col] = np.floor((column_values - col_min) / std_dev)
        else:
            features[:, col] = 0
    scaled_decision_variable = min_max_scaler.fit_transform(decision_variable)
    # Combine features and scaled decision variable back into one matrix
    result_matrix = np.hstack((features, scaled_decision_variable))

    return result_matrix


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
             "std_O", "Acc_F", "std_F", "T_F", "Reduct"]]
    n_steps = 2
    B = []
    # F = []

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
        DS_2 = algo2_preprocessing(DS[0])

        # Attr reduction trên dữ liệu đã qua 2 bước tiền xử lí
        IF = IntuitiveFuzzy(DS_2, 0.000001)  # sửa lại input cho cái này
        F, time_filter = IF.filter()
        print("F", F)

        # Evaluate trên dữ liệu chỉ đi qua bước tiền xử lí đầu tiên
        sc = IF.evaluate(arr[0], DS[0], F, time_filter)
        a_sc.append(sc)
        # os.system('cls')
        print(tabulate(a_sc, headers='firstrow',
              tablefmt='pipe', stralign='center'))

    print(time.time()-start)


if __name__ == "__main__":
    main(arr_data)
    # for arr in arr_data:
    # df = algo2_preprocessing(arr[0], arr[1])
    # print(pd.DataFrame(df))
