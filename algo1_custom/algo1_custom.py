'''
float32 for all
sau khi tìm được set dis thì vẫn cần phải ép kiểu numpy array

'''


""" Read Page 3 """
import numpy as np
import time, os, pickle
import pandas as pd
from functools import reduce
from tabulate import tabulate
from sklearn import svm, tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import time
import numpy as np

class IntuitiveFuzzy:
    def __init__(self, data, att_nominal_cate=None):
        """Khởi tạo tham số đầu vào
        Input:
            2d numpy array: bảng dữ liệu đầu vào ở dạng numpy, chưa transpose

        Yields:
            condition attribute matrix: a 2d numpy array of data sample without decision attribute
            decision attribute vector: a 1d numpy vector of decision attribute
        """
        
        '''
        Input:
            data: data chưa transpose
            att_data: bảng attributes không có cate attr
            C: decision variable
        '''
        # Input numpy matrix
        self.data = data
        # for calculating
        self.att_nominal_cate = att_nominal_cate
        # self.attributes = range(0,len(self.data[0]))
        # self.arr_real = [i for i in self.attributes  if i not in att_nominal_cate]
        self.cond_attr = data.T[:-1].astype(np.float32)
        self.dec_attr = data.T[-1].astype(np.float32)
        self.attributes = range(0,len(self.data[0]))
        
        self.chose_idx = []  # List to keep track of chosen indices
        
    # đầu vào matrix không bao gồm condition attribute

    def cal_fuzzy_correlation(self):
        """Tính các matrix quan hệ mờ cho từng C_i
        Input:
            matrix đã được transpose và không bao gồm condition attribute

        Return:
            fuzzy_cor_list: list các matrix quan hệ tương quan mờ
        """
        fuzzy_corr = []
        for sample in self.cond_attr:
            arr_2d = np.broadcast_to(sample.reshape(-1, 1), (len(sample), len(sample))).astype(np.float32)
            arr_cols = np.broadcast_to(sample, (len(sample), len(sample))).astype(np.float32)
            equal = arr_2d == arr_cols
            output = np.where(equal, 1.0, np.minimum(arr_2d, arr_cols)).astype(np.float32)
            fuzzy_corr.append(output)        
        return fuzzy_corr

    def cal_M_C(self):
        """Tính matrix M_C từ các matrix tương quan mờ
        Input:
            fuzzy_corr_matrix: list of fuzzy correlation matrix

        Return:
            M_C: a 2d numpy matrix
        """
        fuzzy_corr_matrix = self.cal_fuzzy_correlation()
        M_C = np.min(fuzzy_corr_matrix, axis=0).astype(np.float32)
        return M_C
    
    def cal_lambda(self):
        """Tính lambda cho từng C_i

        Input:
            M_C: 2d numpy array
        
        Return:
            lambda_matrix: a 2d numpy matrix
        """
        M_C = self.cal_M_C()
        inverse_matrix = (1.0 - M_C).astype(np.float32)
        np.fill_diagonal(inverse_matrix, np.inf)
        min_values_matrix = np.min(inverse_matrix, axis=1).astype(np.float32)
        lambda_2d = np.broadcast_to(min_values_matrix.reshape(-1,1), (len(min_values_matrix), len(min_values_matrix))).astype(np.float32)
        return lambda_2d

    def partition_mask(self):
        """Tính matrix mask để bỏ đi những giá trị cùng phân cấp khi tính discernibility set

        Input:
            decision attribute vector: a 1d numpy array of condition attribute matrix 

        Return:
            mask: a 2d numpy array of masked valued
        """
        cond_att = self.dec_attr
        arr_2d = np.broadcast_to(cond_att.reshape(-1, 1), (len(cond_att), len(cond_att))).astype(np.float32)
        mask = (arr_2d.T == arr_2d).astype(np.float32)
        return mask

    def cal_dis_set(self):
        """Dùng cho tính toán các cặp discernibility của từng C_i.
        A list of sub-list. Each sub-list is a collection of 
        unique discernibility set for each C_i
        
        Input:
            condition attributes matrix: a 2d nunpy array of data sample

        Return:
            A list of unique discernibility set for each C_i
            a list of length of each C_i discenibility collection
        """
        mask = self.partition_mask()
        lambda_matrix = self.cal_lambda()

        all_dis_set = []
        all_len_dis_set = []
        for sample in self.cond_attr:
            arr_2d = np.broadcast_to(sample.reshape(-1, 1), (len(sample), len(sample))).astype(np.float32)
            arr_cols = np.broadcast_to(sample, (len(sample), len(sample))).astype(np.float32)
            output = (1.0 - np.minimum(arr_2d, arr_cols)).astype(np.float32)
            output[mask==1] = -1.0
            dis = np.where(output >= lambda_matrix)
            dis = list(zip(dis[0], dis[1]))
            dis_len = len(dis)
            all_len_dis_set.append(dis_len)
            all_dis_set.append(dis)
        return all_dis_set, all_len_dis_set
    
    def cal_permutation_dis_set(self): 
        '''Tính chỉnh hợp của các discernibility set

        Input:
            list of list: list of all discernibility
        
        Return:
            int: length of the permutation set
        '''
        concatenated_list = []
        all_dis_set,  all_len_dis_set = self.cal_dis_set()    
        for inner_list in all_dis_set:
            concatenated_list.extend(inner_list)
            concatenated_list = list(set(concatenated_list))
        len_concatenated_list = len(concatenated_list)
        return all_dis_set, len_concatenated_list, all_len_dis_set

    def filter(self):
        start = time.time()
        A = []  # List to hold unique elements after each merge
        all_dis_set, len_per_list, all_len_dis = self.cal_permutation_dis_set()

        max_idx = all_len_dis.index(max(all_len_dis))
        self.chose_idx.append(max_idx)
        A.extend(all_dis_set[max_idx])

        if len(A) >= len_per_list:
            finish_time = time.time() - start
            return self.chose_idx, finish_time
        else:
            temp_B = {}  # Initialize the dictionary outside the loop
            while len(A) < len_per_list:
                for idx, dis_set in enumerate(all_dis_set):
                    if idx not in self.chose_idx:
                        copy_A = A.copy()
                        copy_A.extend(dis_set)
                        copy_A = list(set(copy_A))
                        if len(copy_A) >= len_per_list:
                            self.chose_idx.append(idx)
                            finish_time = time.time() - start
                            return self.chose_idx, finish_time
                        else:
                            temp_B[idx] = len(copy_A)

                if temp_B:
                    idx_max_value = max(temp_B, key=temp_B.get)
                    A.extend(all_dis_set[idx_max_value])
                    A = list(set(A))
                    self.chose_idx.append(idx_max_value)
                    temp_B.clear()  # Clear temp_B to recalculate in the next iteration
                else:
                    print("Error: temp_B is empty. No valid combination found.")
                    break
        finish_time = time.time() - start
        return self.chose_idx, finish_time
    
    def scores(self, reduct):

		# y_test = self.data[:,-1]
		# y_test = y_test.astype(int)
		# print(reduct)
		# list_index = [self.attributes[:-1].index(i) for i in reduct]
		# X_test = self.data[:,list_index]
		# print(list_index)

        y_train = self.data[:,-1]
        y_train = y_train.astype(int)
        X_train = self.data[:, reduct]


        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
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
        #self.arr_acc.append(scores)

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
        self.num_objs = len(self.data[0])
        # self.dis_tg = self.dis_tg
	
    def update_dis(self, dis):
        self.dis_tg = dis

    def update_n_attribute(self, B):
        '''
            This is a function to update a new list of attributes
            Params:lear
                - B: list of attribute after wrapper phase of previous step.
        '''
        # TODO: save reduct set into self.B
        # self.dis_tg = dis_tg
        self.B = B

    def update_retional_matrices(self):
        '''
            This is a function to update relational_matrices
        '''
        self.relational_matrices = self._get_single_attr_IFRM(self.data)
        

    def evaluate(self, name, reduct_f, time_f):
        file_name = os.path.basename(name)
		# print("reduct_f", reduct_f)
		# cf = tree.DecisionTreeClassifier()
		# cf= svm.SVC(kernel='rbf', C=1, random_state=42)
        cf = KNeighborsClassifier(n_neighbors=5)

		# y_test= self.data[:,-1]
		# y_test = y_test.astype(int)
        y_train = self.data[:,-1]
        y_train = y_train.astype(int)
		
		
		# X_test_o = self.data[:,:-1]
        X_train_o = self.data[:,:-1]


        clf_o = cf.fit(X_train_o, y_train)
		# scores_o = round(clf_o.score(X_test_o, y_test),3)
		# clf = KNeighborsClassifier(n_neighbors=10)
        H_o = cross_val_score(clf_o, X_train_o, y_train, cv=10)
        scores_o = round(H_o.mean(),3)
        std_o = round(np.std(H_o), 3)



		# Calculate Filter
		# reduct_f = reduct_f[-1]
		# X_test = self.data[:, reduct_f]
        X_train = self.data[:, reduct_f]

        clf = cf.fit(X_train, y_train)
        # scores_f = round(clf.score(X_test, y_train),3)
        H_f = cross_val_score(clf, X_train, y_train, cv=10)
        scores_f = round(H_f.mean(),3)
        std_f = round(np.std(H_f), 3)

        return (file_name, len(self.attributes)-1, len(reduct_f),
                scores_o, std_o, scores_f, std_f, round(time_f,3), self.chose_idx)


# Example usage
# con_attr = [
#     [0.2, 0.6, 1.0, 0.1, 0.0],
#     [0.9, 0.5, 0.3, 0.8, 2.0],
#     [0.6, 0.8, 0.6, 0.4, 1.0],
#     [0.3, 0.6, 0.9, 1.0, 0.0]
# ]

def main():
    file_path = '/Users/phunghongquan/Documents/NCS-VietAnh/algorithm_custom/data/libras_movement/movement_libras.data'
    data = pd.read_csv(file_path, header=None, delimiter=',')
    data_array = data.to_numpy()

    algo = IntuitiveFuzzy(data_array)
    
    result, elapsed_time = algo.filter()
    print(result, elapsed_time)

if __name__ == "__main__":
    main()
    