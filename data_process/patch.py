import numpy as np
import os


dirs = "./Results/COP/ED_12/"

sh = './data_process/'+'patch.sh'

dir_list = os.listdir(dirs)
dir_list.sort()

bench_list = [256, 512, 768, 1024]


def sift(ori, part):
    bench = ori.copy()
    for unit in part:
        bench.remove(unit)
    return(bench)


for folder in dir_list:
    x = folder.count('WTI')
    print('\n------------------------------------------------')
    print('Loading Data: '+folder)

    if x == 1:
        folder_dir = dirs+folder

        figs = os.listdir(folder_dir)

        if 'Fig_SVR_RBF.png' not in figs:
            with open(sh, 'a+') as f:
                print('python3 svr_train.py' + ' --dir ' + folder , file=f)
            print('python3 svr_train.py' + ' --dir ' + folder )

        gru_lists = []
        linear_lists = []
        lstm_lists = []
        rnn_lists = []
        for fig in figs:
            x = fig.count('Fig_')
            y = fig.count('H')
            if x == 1 and y == 1:
                fig = fig.replace('.png', '')
                fig = fig.replace('Fig_', '')
                fig = fig.replace('L1_H', '')
                fig = fig.replace('_E10000_Adam', '')
                parm_list = fig.split("_")
                if parm_list[0] == 'GRU':
                    gru_lists.append(int(parm_list[1]))
                if parm_list[0] == 'Linear':
                    linear_lists.append(int(parm_list[1]))
                if parm_list[0] == 'LSTM':
                    lstm_lists.append(int(parm_list[1]))
                if parm_list[0] == 'RNN':
                    rnn_lists.append(int(parm_list[1]))
            

        gru_t = sift(bench_list, gru_lists)
        linear_t = sift(bench_list, linear_lists)
        lstm_t = sift(bench_list, lstm_lists)
        rnn_t = sift(bench_list, rnn_lists)

        print('Linear: ', linear_t)
        print('RNN: ', rnn_t)
        print('GRU: ', gru_t)
        print('LSTM: ', lstm_t)

        if len(linear_t) > 0:
            for size in linear_t:
                with open(sh, 'a+') as f:
                    print('python3 mlp_train.py --cell Linear --hidden_size ' +
                          str(size) + ' --dir ' + folder + ' --num_iters 5000', file=f)
                print('python3 mlp_train.py --cell Linear --hidden_size ' +
                      str(size) + ' --dir ' + folder + ' --num_iters 5000')

        if len(rnn_t) > 0:
            for size in rnn_t:
                with open(sh, 'a+') as f:
                    print('python3 rnn_train.py --cell RNN --hidden_size ' +
                          str(size) + ' --dir ' + folder + ' --num_iters 5000', file=f)
                print('python3 rnn_train.py --cell RNN --hidden_size ' +
                      str(size) + ' --dir ' + folder + ' --num_iters 5000')                      
        if len(gru_t) > 0:
            for size in gru_t:
                with open(sh, 'a+') as f:
                    print('python3 rnn_train.py --cell GRU --hidden_size ' +
                          str(size) + ' --dir ' + folder + ' --num_iters 5000', file=f)
                print('python3 rnn_train.py --cell GRU --hidden_size ' +
                      str(size) + ' --dir ' + folder + ' --num_iters 5000')

        if len(lstm_t) > 0:
            for size in lstm_t:
                with open(sh, 'a+') as f:
                    print('python3 rnn_train.py --cell LSTM --hidden_size ' +
                          str(size) + ' --dir ' + folder + ' --num_iters 5000', file=f)
                print('python3 rnn_train.py --cell LSTM --hidden_size ' +
                      str(size) + ' --dir ' + folder + ' --num_iters 5000')

        print('')
