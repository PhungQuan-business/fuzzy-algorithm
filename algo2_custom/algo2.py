'''
float32 for all
sau khi tìm được set dis thì vẫn cần phải ép kiểu numpy array

'''


""" Read Page 3 """




from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import svm, tree
from tabulate import tabulate
from functools import reduce
import numpy as np
import time, os, pickle
class IntuitiveFuzzy(object):

    def __init__(self, dataset, delta):
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.stop_cond = None
        self.FKG_B = None
        self.delta = delta
        self.B = []

        self.attributes = range(0, len(self.dataset[0]))

    def cal_partition(self, matrix, col_idx=None, drop=False):
        '''Calculate partition
        Input:
            matrix: Input matrix 2d-array
            col_idx(1d-array): array of one or more index of the column want to drop or retain

        Parameters:
            drop(boolean): 
                if False then keep only the column in the col_idx
                if True then keep all other columns except those in col_idx

        Output:
            a list of partition
        '''
        start_2 = time.time()
        if col_idx is not None:
            if drop:
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
        '''Attribute reduction
        Idea:
            khởi tạo mảng B rỗng
            Nếu B rỗng:
                -   tính phân hoạch cục bộ
                -   tính FKG cục bộ
                -   tính U/C{C_i}
                -   tìm max(sig)
                -   append index max(sig) vào B
            Nếu B không rỗng:
                -   Nếu FKG của U/B <= delta
                    -   Stop
                -   Nếu FKG của U/B !<= delta
                    -   Hợp U/B với từng C_i | i not in B
                    -   tính FKG U/B hợp C_i
                    -   tìm max(sig)
                    -   append index của max(sig) vào B
                    -   check if FKG U/B <= delta
        Output:
            index(es) of retain columns
        '''
        while True:
            if not self.B:
                start = time.time()
                self.stop_cond = self.cal_FKG(
                    self.dataset_len, self.cal_partition(self.dataset))
                print(f'this is the stop condition', self.stop_cond)
                temp = {}
                # Loop over columns except the last one
                for i in range(self.dataset.shape[1] - 1):
                    FKG_Ci = self.cal_FKG(self.dataset_len, self.cal_partition(
                        self.dataset, col_idx=[i], drop=True))
                    sig_Ci = abs(FKG_Ci - self.stop_cond)
                    temp[i] = sig_Ci
                max_key = max(temp, key=temp.get)
                self.B.append(max_key)
            else:
                start = time.time()
                self.FKG_B = self.cal_FKG(self.dataset_len, self.cal_partition(
                    self.dataset, col_idx=self.B, drop=False))

                temp = {}
                # Loop over columns except the last one
                # TODO có thể xoá những index đã add vào B khỏi danh sách index
                for i in range(self.dataset.shape[1] - 1):
                    if i not in self.B:
                        FKG_B_Ci = self.cal_FKG(self.dataset_len, self.cal_partition(
                            self.dataset, col_idx=self.B+[i], drop=False))
                        sig_Ci = abs(self.FKG_B - FKG_B_Ci)
                        temp[i] = sig_Ci
                max_key = max(temp, key=temp.get)
                self.B.append(max_key)
                print('This is self.B:', self.B)

                # tính lại FKG_B sau khi index mới được append
                self.FKG_B = self.cal_FKG(self.dataset_len, self.cal_partition(
                    self.dataset, col_idx=self.B, drop=False))
                if abs(self.FKG_B - self.stop_cond) != 0:
                    # if self.FKG_B == self.stop_cond:
                    print(f'done at the first iteration')
                    finish = time.time() - start
                    return self.B, finish

    def scores(self, reduct):
        # y_test = self.data[:,-1]
        # y_test = y_test.astype(int)
        # print(reduct)
        # list_index = [self.attributes[:-1].index(i) for i in reduct]
        # X_test = self.data[:,list_index]
        # print(list_index)

        y_train = self.dataset[:, -1]
        y_train = y_train.astype(int)
        X_train = self.dataset[:, reduct]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
        # clf = tree.DecisionTreeClassifier()
        # clf = svm.SVC(kernel='rbf', C=1, random_state=42).fit(X_train, y_train)
        # clf = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
        # clf = svm.SVC(kernel='rbf', C=1, random_state=42)
        clf = KNeighborsClassifier(n_neighbors=5)
        H = cross_val_score(clf, X_train, y_train, cv=10)
        acc = round(H.mean(), 3)
        std = round(np.std(H), 3)
        # scores = round(cross_val_score(clf, X_train, y_train, cv=10).mean(),3)
        # scores = clf.score(X_test, y_test)
        # self.arr_acc.append(scores)

        return acc, std

    def update_dataset(self, data):
        '''
                        This is a function to update incremental dataset
                        Params:
                                        - data: new dataset
        '''
        self.data = data
        # self.prev = num_prev

    def update_n_objs(self):
        '''
                        This is a function to update a new number of objects
        '''
        # self.num_prev += self.num_objs
        # self.num_delta = self.num_objs - self.num_prev
        self.num_objs = len(self.data[:, 0])
        # self.dis_tg = self.dis_tg

    def update_dis(self, dis):
        self.dis_tg = dis

    def update_n_attribute(self, B):
        '''
                        This is a function to update a new list of attributes
                        Params:lear
                                        - self.B: list of attribute after wrapper phase of previous step.
        '''
        # TODO: save reduct set into self.B
        # self.dis_tg = dis_tg
        self.B = B

    def update_retional_matrices(self):
        '''
                        This is a function to update relational_matrices
        '''
        self.relational_matrices = self._get_single_attr_IFRM(self.data)

    # TODO yêu cầu user cung cấp dataset cho function, không dùng biến cục bộ class nữa
    # def evaluate(self, name, dataset, reduct_f, time_f):
    #     # print("reduct_f", reduct_f)
    #     # cf = tree.DecisionTreeClassifier()
    #     # cf= svm.SVC(kernel='rbf', C=1, random_state=42)
    #     cf = KNeighborsClassifier(n_neighbors=5)

    #     # y_test= self.data[:,-1]
    #     # y_test = y_test.astype(int)
    #     y_train = dataset[:, -1]
    #     y_train = y_train.astype(int)

    #     # X_test_o = self.dataset[:,:-1]
    #     X_train_o = dataset[:, :-1]

    #     clf_o = cf.fit(X_train_o, y_train)
    #     # scores_o = round(clf_o.score(X_test_o, y_test),3)
    #     # clf = KNeighborsClassifier(n_neighbors=10)
    #     H_o = cross_val_score(clf_o, X_train_o, y_train, cv=10)
    #     scores_o = round(H_o.mean(), 3)
    #     std_o = round(np.std(H_o), 3)

    #     # Calculate Filter
    #     # reduct_f = reduct_f[-1]
    #     # X_test = self.dataset[:, reduct_f]
    #     X_train = dataset[:, reduct_f]

    #     clf = cf.fit(X_train, y_train)
    #     # scores_f = round(clf.score(X_test, y_train),3)
    #     H_f = cross_val_score(clf, X_train, y_train, cv=10)
    #     scores_f = round(H_f.mean(), 3)
    #     std_f = round(np.std(H_f), 3)

    #     return (name, len(self.attributes)-1, len(reduct_f),
    #             scores_o, std_o, scores_f, std_f, round(time_f, 3), list(self.B))
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
