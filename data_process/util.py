import json
import logging
import os
import shutil

import torch
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams['savefig.dpi'] = 300 #Uncomment for higher plot resolutions
import matplotlib.pyplot as plt

import os

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, Sampler


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
        self.v = v_data.copy()
        self.test_len = self.data.shape[0]
        logger.info(f'test_len: {self.test_len}')

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return (self.data[index,:,:-1],int(self.data[index,0,-1]),self.v[index],self.label[index])


def prep_data(data, train = True, h=None,dim=None,sample_dense=True):
    assert h != None and dim != None
    raw_data = unpadding(data).reshape(-1,1)
    time_len = raw_data.shape[0]
    input_size = dim
    window_size = h + dim
    stride_size = h
    num_series = 1
    if not sample_dense:
        windows_per_series = np.full((num_series), (time_len-input_size) // stride_size)
    else:
        windows_per_series = np.full((num_series), 1+ time_len-window_size)
    total_windows = np.sum(windows_per_series)


    x_input = np.zeros((total_windows, window_size, 1+1),dtype = 'float32')
    label = np.zeros((total_windows,window_size),dtype = 'float32')
    v_input= np.zeros((total_windows, 2),dtype = 'float32')

    count=0
    for series in range(num_series):
        for i in range(windows_per_series[series]):
            # get the sample with minimal time period, in this case. which is 24 points (24h, 1 day)
            stride=1
            if not sample_dense:
                stride=stride_size

            window_start = stride*i
            window_end = window_start+window_size
            '''
            print("x: ", x_input[count, 1:, 0].shape)
            print("window start: ", window_start)
            print("window end: ", window_end)
            print("data: ", data.shape)
            print("d: ", data[window_start:window_end-1, series].shape)
            '''
            # using the observed value in the t-1 step to forecast the t step, thus the first observed value in the input should be t0 step and is 0, as well as the first value in the labels should be t1 step.
            x_input[count, 1:, 0] = raw_data[window_start:window_end-1, series]
            x_input[count, :, -1] = series
            label[count, :] = raw_data[window_start:window_end, series]
            # get the nonzero number of the input observed values
            nonzero_sum = (x_input[count, 1:input_size, 0]!=0).sum()
            if nonzero_sum == 0:
                v_input[count, 0] = 0
            else:
                # get the average values of the input observed values ( +1 means smoothing?)
                v_input[count, 0] = np.true_divide(x_input[count, 1:input_size, 0].sum(),nonzero_sum)+1
                # sample standardization
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

class Params:
    '''Class that loads hyperparameters from a json file.
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    '''

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path):
        '''Loads parameters from json file'''
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        '''Gives dict-like access to Params instance by params.dict['learning_rate']'''
        return self.__dict__

def set_logger(log_path):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    _logger = logging.getLogger('DeepAR')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))

def save_checkpoint(state, is_best, epoch, checkpoint, ins_name=-1):
    '''Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ins_name: (int) instance index
    '''
    if ins_name == -1:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}.pth.tar')
    else:
        filepath = os.path.join(checkpoint, f'epoch_{epoch}_ins_{ins_name}.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info(f'Checkpoint Directory does not exist! Making directory {checkpoint}')
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    logger.info(f'Checkpoint saved to {filepath}')
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))
        logger.info('Best checkpoint copied to best.pth.tar')


def load_checkpoint(checkpoint, model, optimizer=None):
    '''Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        gpu: which gpu to use
    '''
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"File doesn't exist {checkpoint}")
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda')
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def save_dict_to_json(d, json_path):
    '''Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    '''
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)

def plot_all_epoch(variable, save_name, location='./figures/'):
    num_samples = variable.shape[0]
    x = np.arange(start=1, stop=num_samples + 1)
    f = plt.figure()
    plt.plot(x, variable[:num_samples])
    f.savefig(os.path.join(location, save_name + '_summary.png'))
    plt.close()