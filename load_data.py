import os

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import h5py
import random
import warnings
warnings.filterwarnings("ignore")


# 需要h5py读取
ALL_data = dict(
    Caltech101_7= {1: 'Caltech101_7', 'N': 1400, 'K': 7, 'V': 5, 'n_input': [1984, 512, 928, 254, 40],'para_loss': [1e-4, 1e-2]},
    HandWritten = {1: 'handwritten1031_v73', 'N': 2000, 'K': 10, 'V': 6, 'n_input': [240, 76, 216, 47, 64, 6],'para_loss': [1e-4, 1e-2]},
    ALOI_100 = {1: 'ALOI_100_7', 'N': 10800, 'K': 100, 'V': 4, 'n_input': [77, 13, 64, 125],'para_loss': [1e-3, 1e-3]},
    YouTubeFace10_4Views={1: 'YTF10_4', 'N': 38654, 'K': 10, 'V': 4, 'n_input': [944, 576, 512, 640],'para_loss': [1e-4, 1e-2]}
)


path = 'Datasets/'


def get_mask(view_num, alldata_len, missing_rate):

    missindex = np.ones((alldata_len, view_num))
    b=((10 - 10*missing_rate)/10) * alldata_len
    miss_begin = int(b)
    for i in range(miss_begin, alldata_len):
        missdata = np.random.randint(0, high=view_num,
                                     size=view_num - 1)

        missindex[i, missdata] = 0

    return missindex


def Form_Incomplete_Data(missrate=0.5, X = [], Y = []):
    np.random.seed(1)

    size = len(Y[0])
    view_num = len(X)
    index = [i for i in range(size)]
    np.random.shuffle(index)
    for v in range(view_num):
        X[v] = X[v][index]
        Y[v] = Y[v][index]


    missindex = get_mask(view_num, size, missrate)

    index_complete = []
    index_partial = []
    for i in range(view_num):
        index_complete.append([])
        index_partial.append([])
    for i in range(missindex.shape[0]):
        for j in range(view_num):
            if missindex[i, j] == 1:
                index_complete[j].append(i)
            else:
                index_partial[j].append(i)

    filled_index_com = []
    for i in range(view_num):
        filled_index_com.append([])
    max_len = 0
    for v in range(view_num):
        if max_len < len(index_complete[v]):
            max_len = len(index_complete[v])
    for v in range(view_num):
        if len(index_complete[v]) < max_len:
            diff_len = max_len - len(index_complete[v])

            diff_value = random.sample(index_complete[v], diff_len)
            filled_index_com[v] = index_complete[v] + diff_value
        elif len(index_complete[v]) == max_len:
            filled_index_com[v] = index_complete[v]

    filled_X_complete = []
    filled_Y_complete = []
    for i in range(view_num):
        filled_X_complete.append([])
        filled_Y_complete.append([])
        filled_X_complete[i] = X[i][filled_index_com[i]]
        filled_Y_complete[i] = Y[i][filled_index_com[i]]
    for v in range(view_num):

        X[v] = torch.from_numpy(X[v])
        filled_X_complete[v] = torch.from_numpy(filled_X_complete[v])

    return X, Y, missindex, filled_X_complete, filled_Y_complete, index_complete, index_partial

def load_data(dataset, missrate):

    data = h5py.File(path + dataset[1] + ".mat")
    X = []
    Y = []
    Label = np.array(data['Y']).T

    Label = Label.reshape(Label.shape[0])
    mm = MinMaxScaler()
    for i in range(data['X'].shape[1]):
        diff_view = data[data['X'][0, i]]
        diff_view = np.array(diff_view, dtype=np.float32).T
        std_view = mm.fit_transform(diff_view)
        X.append(std_view)
        Y.append(Label)
    X, Y, missindex, X_com, Y_com, index_com, index_incom = Form_Incomplete_Data(missrate=missrate, X=X, Y=Y)

    return X, Y, missindex, X_com, Y_com, index_com, index_incom



