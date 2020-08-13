import torch 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Sampler

import logging

logger = logging.getLogger('DeepAR.Data')


def scale_raw(raw):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # raw = raw.reshape(raw.shape[0], 1)
    scaler = scaler.fit(raw)
    # transform train
    norm_raw = scaler.transform(raw)
    norm_raw = norm_raw[:,0]
    return scaler, norm_raw

def toTorch(train_input, train_target, test_input, test_target):
    train_input = torch.from_numpy(
        train_input).float()
    train_target = torch.from_numpy(
        train_target).float()
    # --
    test_input = torch.from_numpy(
        test_input).float()
    test_target = torch.from_numpy(
        test_target).float()
    return train_input, train_target, test_input, test_target


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    # dataset = np.insert(dataset, [0] * look_back, 0)
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    dataY = np.array(dataY)
    dataY = np.reshape(dataY, (dataY.shape[0], 1))
    dataset = np.concatenate((dataX, dataY), axis=1)
    return dataset

def unpadding(y):
    a = y.copy()
    h = y.shape[1]
    s = np.empty(y.shape[0] + y.shape[1] -1)

    for i in range(s.shape[0]):
        s[i]=np.diagonal(np.flip(a,1), offset= -i + h-1,axis1=0,axis2=1).copy().mean()
    
    return s

class TrainDataset(Dataset):
    def __init__(self, x_data, label_data):
        self.data = x_data.copy()
        self.label = label_data.copy()
        self.train_len = self.data.shape[0]
        logger.info(f'train_len: {self.train_len}')

    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]), self.label[index])

class TestDataset(Dataset):
    def __init__(self, x_data, label_data, v_data):
        self.data = x_data.copy()
        self.label = label_data.copy()
        self.v = v_data
        self.test_len = self.data.shape[0]
        logger.info(f'test_len: {self.test_len}')

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]),self.v[index],self.label[index])


def prep_data(data, train = True, h=None,dim=None):
    assert h != None and dim != None
    raw_data = unpadding(data)
    time_len = raw_data.shape[0]
    input_size = d
    window_size = h + d
    stride_size = h
    num_series = 1
    windows_per_series = np.full((num_series), (time_len-input_size) // stride_size)
    total_windows = np.sum(windows_per_series)


    x_input = np.zero((total_windows, window_size, 1+1),dtype = 'float32')
    label = np.zeros((samples,sample_len),dtype = 'float32')
    v_input= np.zeros((total_windows, 2),dtype = 'float32')

    count=0
    for i in range(windows_per_series[series]):
        if train:
            window_start = stride_size*i+data_start[series]
        else:
            window_start = stride_size*i
        window_end = window_start+window_size
        '''
        print("x: ", x_input[count, 1:, 0].shape)
        print("window start: ", window_start)
        print("window end: ", window_end)
        print("data: ", data.shape)
        print("d: ", data[window_start:window_end-1, series].shape)
        '''
        x_input[count, 1:, 0] = data[window_start:window_end-1, series]
        x_input[count, :, -1] = series
        label[count, :] = data[window_start:window_end, series]
        nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()
        if nonzero_sum == 0:
            v_input[count, 0] = 0
        else:
            v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(),nonzero_sum)+1
            x_input[count, :, 0] = x_input[count, :, 0]/v_input[count, 0]
            if train:
                label[count, :] = label[count, :]/v_input[count, 0]
        count += 1
    
    dataset_torch = None
    if train:
        dataset_torch=TrainDataset(x_data=x_input,label_data= label)
    else:
        dataset_torch=TestDataset(x_data=x_input,label_data= label, v_data= v_input)
    return dataset_torch