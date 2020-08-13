from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
from tqdm import trange

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
from data_process._data_process import create_dataset, scale_raw
from numpy import concatenate, atleast_2d
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(
    description='PyTorch SCCNN Time Series Forecasting after Landmarks')
parser.add_argument('-step', type=int, default=6, metavar='N',
                    help='steps for prediction (default: 1)')
parser.add_argument('-dataset', type=str, default='Brent')
parser.add_argument('-dim', type=int, default=24)

if __name__ == "__main__":
    args = parser.parse_args()

    data = args.dataset
    dim = args.dim
    ts = np.load('./Data/paper/'+data+'.npz')
    ts = ts['arr_0'].reshape(-1)
    # set_length = len(ts)
    segmentation = int(len(ts)*2/3)
    # np.savez('ts.npz',ts)
    # ts = ts.reshape(-1,1)

    # scaler, ts_scaled = scale_raw(ts)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    h = args.step
    dataset = create_dataset(ts, look_back=dim + h - 1)
    dataset = scaler.fit_transform(dataset)
    X, Y = dataset[:, :(0 - h)], dataset[:, (0-h):]

    train_data = dataset[:segmentation,:]
    test_data = dataset[:segmentation,:]

    train_input = X[:segmentation, :]
    train_target = Y[:segmentation]
    test_input = X[segmentation:, :]
    test_target = Y[segmentation:]
  
    train_target = train_target.reshape(-1, h)
    test_target = test_target.reshape(-1, h)

    train_rmse_batch = np.empty(10)
    test_rmse_batch = np.empty(10)
    train_pred_Batchs = np.empty((train_input.shape[0],h,10))
    test_pred_Batchs = np.empty((test_input.shape[0],h,10))

    train_rmse_loss_batch = np.empty((100,10)) 
    test_rmse_loss_batch = np.empty((100,10))

    def prep_data(data, train=True):
        