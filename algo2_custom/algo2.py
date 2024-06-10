from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm, tree
from tabulate import tabulate
from functools import reduce
import numpy as np
import time
import os
import pickle


class IntuitiveFuzzy(object):
    def __init__(self, dataset, delta, B=None):
        self.dataset = dataset
        self.delta = delta
        self.dataset_len = len(dataset)
        # self.stop_cond = stop_cond
        self.FKG_B = None
        self.delta = delta
        self.B = B
        self.attributes = range(0, len(self.dataset[0]))

    def cal_partition(self, matrix, col_idx=None, drop=False):
        '''Calculate partition
        Input:
            matrix: Input matrix 2d-array
            col_idx(1d-array): array of one or more index of the column want to drop or retain
        Parameters:
            drop(boolean): 
                if True then keep all other columns except those in col_idx
                if False then keep only the column in the col_idx
        Output:
            a list of partition
        '''
        start_2 = time.time()
        if col_idx is not None:
            if drop == True:
                U_reduced = np.delete(matrix, col_idx, axis=1)[:, :-1]
            else:
                U_reduced = matrix[:, col_idx]
        else:
            U_reduced = matrix[:, :-1]
        row_dict = {}
        for index, row in enumerate(U_reduced):
            row_tuple = tuple(row)
            if row_tuple in row_dict:
                row_dict[row_tuple].append(matrix[index, -1])
            else:
                row_dict[row_tuple] = [matrix[index, -1]]
        result = list(row_dict.values())
        end_2 = time.time() - start_2
        # print(result)
        return result

    def cal_FKG(self, dataset_len, partition_list):
        '''Calculate FKG
        Input:
            dataset_len: number of sample in the dataset
            partition_list: a list of partition
        Output:
            FKG
        '''
        FKG = 1 / (dataset_len)**2 * \
            np.sum([(len(partition)-1) * sum(partition)
                    for partition in partition_list])
        return FKG

    def filter(self):
        # sig_Ci = 0
        while True:
            if self.B is None:
                start = time.time()
                self.stop_cond = self.cal_FKG(
                    self.dataset_len, self.cal_partition(self.dataset))
                # print(f'this is the stop condition', self.stop_cond)

                temp = {}
                # Loop over columns except the last one
                for i in range(self.dataset.shape[1] - 1):
                    FKG_Ci = self.cal_FKG(self.dataset_len, self.cal_partition(
                        self.dataset, col_idx=[i], drop=False))
                    sig_Ci = abs(FKG_Ci - self.stop_cond)
                    # sig_Ci = round(sig_Ci, 3)
                    temp[i] = sig_Ci
                max_key = max(temp, key=temp.get)
                # print(f'all value after first iter', temp.values())
                # print('\n')
                # print('-' * 100)
                # print(f'value of max key is:', temp[max_key])
                # print(max_key)
                # new_arr = [idx for idx, fkg in temp.items() if fkg ==
                #            temp[max_key]]
                # self.B = new_arr
                self.B = [max_key]
                # print(
                #     f'this is self.B after iteration 1, should be more than one', self.B)
                while True:
                    start = time.time()
                    self.FKG_B = self.cal_FKG(self.dataset_len, self.cal_partition(
                        self.dataset, col_idx=self.B, drop=False))
                    if abs(self.FKG_B - self.stop_cond) <= self.delta:
                        # if self.FKG_B == self.stop_cond:
                        print(f'done at the first iteration')
                        finish = time.time() - start
                        return self.B, finish
                    temp = {}
                    # Loop over columns except the last one
                    # TODO có thể xoá những index đã add vào B khỏi danh sách index
                    for i in range(self.dataset.shape[1] - 1):
                        if i not in self.B:
                            FKG_B_Ci = self.cal_FKG(self.dataset_len, self.cal_partition(
                                self.dataset, col_idx=self.B+[i], drop=False))
                            sig_Ci = abs(self.FKG_B - FKG_B_Ci)
                            # sig_Ci = round(sig_Ci, 3)
                            temp[i] = sig_Ci

                    max_key = max(temp, key=temp.get)

                    # new_arr = [idx for idx, fkg in temp.items() if fkg ==
                    #            temp[max_key]]
                    # self.B = self.B + new_arr
                    self.B.append(max_key)
                    # print('This is self.B:', self.B)

                    # tính lại FKG_B sau khi index mới được append
                    # self.FKG_B = self.cal_FKG(self.dataset_len, self.cal_partition(
                    #     self.dataset, col_idx=self.B, drop=False))
                    # if abs(self.FKG_B - self.stop_cond) <= 0.0001 and sig_Ci <= 0.00001:
            else:
                start = time.time()
                self.stop_cond = self.cal_FKG(
                    self.dataset_len, self.cal_partition(self.dataset))
                temp = {}
                self.FKG_B = self.cal_FKG(self.dataset_len, self.cal_partition(
                    self.dataset, col_idx=self.B, drop=False))

                if abs(self.FKG_B - self.stop_cond) <= self.delta:
                    finish = time.time() - start
                    return self.B, finish
                print(f'this is self.B from third iter should not be empty', self.B)
                for i in range(self.dataset.shape[1] - 1):
                    if i not in self.B:
                        FKG_B_Ci = self.cal_FKG(self.dataset_len, self.cal_partition(
                            self.dataset, col_idx=self.B+[i], drop=False))
                        # print(f'self.B after each addition', self.B)
                        sig_Ci = abs(self.FKG_B - FKG_B_Ci)
                        # sig_Ci = round(sig_Ci, 3)
                        temp[i] = sig_Ci
                max_key = max(temp, key=temp.get)

                # new_arr = [idx for idx, fkg in temp.items() if fkg ==
                #            temp[max_key]]
                # self.B = self.B + new_arr
                self.B.append(max_key)
                print('This is len self.B:', len(list(self.B)))

                # tính lại FKG_B sau khi index mới được append
                # self.FKG_B = self.cal_FKG(self.dataset_len, self.cal_partition(
                #     self.dataset, col_idx=self.B, drop=False))
                # if abs(self.FKG_B - self.stop_cond) <= 0.0001 and sig_Ci <= 0.00001:

    def reduce(self):
        start = time.time()
        self.stop_cond = self.cal_FKG(
            self.dataset_len, self.cal_partition(self.dataset))
        self.FKG_B = self.cal_FKG(self.dataset_len, self.cal_partition(
            self.dataset, col_idx=self.B, drop=False))

        if abs(self.FKG_B - self.stop_cond) <= self.delta:
            finish = time.time() - start
            return self.B, finish

        i = 0
        # for i in range(len(self.B)):
        while i < len(self.B) and len(self.B) > 2:
            # tính FKG_(B-b)
            # arr_excluding_index = self.B[:i] + self.B[i+1:]
            B_copy = self.B.copy()
            # print(f'this is copy of self.B', B_copy)
            # B_copy.pop(i)
            # print(f'this is copy of self.B after pop should short 1', B_copy)

            FKG_B_reduce = self.cal_FKG(self.dataset_len, self.cal_partition(
                self.dataset, col_idx=B_copy, drop=False))
            # if abs(self.FKG_B - FKG_B_reduce) <= self.delta:
            if abs(self.FKG_B - FKG_B_reduce) == 0:
                self.B.pop(i)
                i = 0
                continue
            i += 1

        finish = time.time() - start
        return self.B, finish

    def evaluate(self, name, dataset, reduct_f, time_f):
        file_name = os.path.basename(name)
        # print("reduct_f", reduct_f)
        # cf = tree.DecisionTreeClassifier()
        # cf= svm.SVC(kernel='rbf', C=1, random_state=42)
        cf = KNeighborsClassifier(n_neighbors=5)
        # y_test= self.data[:,-1]
        # y_test = y_test.astype(int)
        y_train = dataset[:, -1]
        y_train = y_train.astype(int)
        # X_test_o = dataset[:,:-1]
        X_train_o = dataset[:, :-1]
        clf_o = cf.fit(X_train_o, y_train)
        # scores_o = round(clf_o.score(X_test_o, y_test),3)
        # clf = KNeighborsClassifier(n_neighbors=10)
        H_o = cross_val_score(clf_o, X_train_o, y_train, cv=10)
        scores_o = round(H_o.mean(), 3)
        std_o = round(np.std(H_o), 3)
        # Calculate Filter
        # reduct_f = reduct_f[-1]
        # X_test = dataset[:, reduct_f]
        X_train = dataset[:, reduct_f]
        clf = cf.fit(X_train, y_train)
        # scores_f = round(clf.score(X_test, y_train),3)
        H_f = cross_val_score(clf, X_train, y_train, cv=10)
        scores_f = round(H_f.mean(), 3)
        std_f = round(np.std(H_f), 3)
        return (file_name, len(self.attributes)-1, len(reduct_f),
                scores_o, std_o, scores_f, std_f, round(time_f, 3), list(self.B))
        # return (file_name, len(self.attributes)-1, len(reduct_f),
        #         scores_o, std_o, scores_f, std_f, round(time_f, 3))
