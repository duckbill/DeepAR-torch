from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import numpy as np
from numpy import concatenate

import matplotlib.ticker as ticker
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from _definition import pivot_k_window, MDPP
from data_process._data_process import create_dataset, scale_raw
from typing import List

import os

def remove_former(input_list, n):
    data_list = input_list[n:]
    return data_list

def create_varset(dataset,look_back=1):
    varset = []
    for i in range(len(dataset)-look_back):
        a=dataset[0:(i+look_back)+1]
        varset.append(a)
    return varset


if __name__ == '__main__':
    # load dataset
    dirs = "./Data/Crude_Oil_Price/WTI.npz"
    # dirs = "./Data/Residential_Load/Residential_Load_hour.npz"
    temp = np.load(dirs)
    load_data = temp["arr_0"].tolist()
    # np.savez(dirs,load_data)
    ts_array = np.array(load_data)
    # scaler, ts_array = scale_raw(ts_array)
    set_length = len(ts_array)
    idx_array = np.arange(set_length)
    ts_idx = list(range(set_length))

    k_windows = 4
    # peak_dic, trough_dic=pivot_k_window(load_data, k_windows)
    marks_dic = MDPP(load_data, 12, 0.20)

    # peak_range: List[int] = []
    # peak_value: List[float] = []
    # trough_range: List[int] = []
    # trough_value: List[float] = []
    marks_range: List[int] = []
    marks_value: List[float] = []

    # for idx in peak_dic:
    #     peak_range.append(idx)
    #     peak_value.append(peak_dic[idx])
    # for idx in trough_dic:
    #     trough_range.append(idx)
    #     trough_value.append(trough_dic[idx])
    for idx in marks_dic:
        marks_range.append(idx)
        marks_value.append(marks_dic[idx])

    #remove the former 4 turning points of the series
    marks_range = remove_former(marks_range, 4)
    marks_value = remove_former(marks_value, 4)

# # create fix idx set 
#     for point_idx, i in zip(marks_range, range(len(marks_range))):
#         print(point_idx, i+1)
#         dirs = "./Data/Crude_Oil_Price/ED_12/WTI_"+str(i+1)+"_"+str(point_idx)
#         if not os.path.exists(dirs):
#             os.mkdir(dirs)
#         training_idx = idx_array[:point_idx+1]
#         # shape:(N,T) s.t. N+T-1= length
#         training_idx_set = create_dataset(training_idx, look_back=12)
#         # training_idx = ts_idx[:point_idx+1]
#         test_idx=idx_array[point_idx+1-12:point_idx+1+12]
#         test_idx_set=create_dataset(test_idx, look_back=12)
#         np.savez(dirs+"/idxSet.npz",training_idx_set,test_idx_set)
#         # data_samples=ts_array[:point_idx+1+12]
#         # np.savez(dirs+"/dataSet.npz",data_samples)
#         # test_idx=ts_idx[point_idx+1-12:point_idx+1+12]
#     print("Done!")

# create var idx set 
    for point_idx, i in zip(marks_range, range(len(marks_range))):
        print(point_idx, i+1)
        dirs = "./Data/Crude_Oil_Price/ED_Var/WTI_"+str(i+1)+"_"+str(point_idx)
        if not os.path.exists(dirs):
            os.mkdir(dirs)
        training_idx = idx_array[:point_idx+1]
        # shape:(N,T) s.t. N+T-1= length
        training_idx_set = create_dataset(training_idx, look_back=12)
        # training_idx = ts_idx[:point_idx+1]
        test_idx=idx_array[point_idx+1-12:point_idx+1+12]
        test_idx_set=create_dataset(test_idx, look_back=12)
        np.savez(dirs+"/idxSet.npz",training_idx_set,test_idx_set)
    print("Done!")

# # create fix set
#     for point_idx, i in zip(marks_range, range(len(marks_range))):
#         print(point_idx, i+1)
#         dirs = "./Data/Crude_Oil_Price/ED_12/WTI_"+str(i+1)+"_"+str(point_idx)
#         if not os.path.exists(dirs):
#             os.mkdir(dirs)
#         data_array=ts_array[:point_idx+1+12]
#         scaler, data_array = scale_raw(data_array)
#         training_samples = data_array[:point_idx+1]
#         # shape:(N,T) s.t. N+T-1= length
#         training_set = create_dataset(training_samples, look_back=12)
#         # np.savez(dirs+"/trainSet.npz",training_set)
#         # training_idx = ts_idx[:point_idx+1]
#         test_samples=data_array[point_idx+1-12:point_idx+1+12]
#         test_set=create_dataset(test_samples, look_back=12)
#         np.savez(dirs+"/dataSet.npz",training_set,test_set)
#         data_samples=data_array[:point_idx+1+12]
#         np.savez(dirs+"/rawSet.npz",data_samples)
  
#     print("Done!")

# # create var set
#     for point_idx, i in zip(marks_range, range(len(marks_range))):
#         print(point_idx, i+1)
#         dirs = "./Data/Crude_Oil_Price/ED_Var/WTI_"+str(i+1)+"_"+str(point_idx)
#         if not os.path.exists(dirs):
#             os.mkdir(dirs)
#         data_array=ts_array[:point_idx+1+12]
#         scaler, data_array = scale_raw(data_array)
#         training_samples = data_array[:point_idx+1]
#         # shape:(N,T) s.t. N+T-1= length
#         training_set = create_varset(training_samples, look_back=12)
#         # np.savez(dirs+"/trainSet.npz",training_set)
#         # training_idx = ts_idx[:point_idx+1]
#         test_samples=data_array[point_idx+1-12:point_idx+1+12]
#         test_set=create_dataset(test_samples, look_back=12)
#         np.savez(dirs+"/dataSet.npz",training_set,test_set)
#         data_samples=data_array[:point_idx+1+12]
#         np.savez(dirs+"/rawSet.npz",data_samples)
#     print("Done!")