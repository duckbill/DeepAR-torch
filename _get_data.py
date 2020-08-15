import os
import pandas as pd
import numpy as np

if __name__ == "__main__":
    
    data_frame = pd.read_csv('data/elect/LD2011_2014.txt',sep=";", index_col=0, parse_dates=True, decimal=',')

    train_start = '2011-01-01 00:00:00'
    train_end = '2014-08-31 23:00:00'
    test_start = '2014-08-25 00:00:00' #need additional 7 days as given info
    test_end = '2014-09-07 23:00:00'

    data_frame = data_frame.resample('1H',label = 'left',closed = 'right').sum()[train_start:test_end]

    data_frame.fillna(0, inplace=True)

    data = data_frame[train_start:test_end].values

    data_start = (data!=0).argmax(axis=0)

    get = data[data_start[0]:,0]

    np.save('data/paper/MT_001', get)
    
    get_test = np.load('data/paper/MT_001.npy')
    
    print(get_test.shape)