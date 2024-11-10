from __future__ import print_function

import numpy as np
import pandas as pd
import torch as th
import random

PATH_TO_DATA= './movie/'


def load_adj(data_path):
    df_adj=pd.read_csv(data_path+'adj.tsv', sep='\t', dtype= {0:np.int32, 1:np.int32})
    adj_tensor=th.tensor(df_adj.values, dtype=th.int32)
    return adj_tensor

def load_latest_session(data_path):
    ret=[]
    for line in open(data_path+'latest_session.txt'):
        chunks= line.strip().split(',')
        ret.append(chunks)
    return ret

def load_map(data_path, name='user'):
    if name== 'user':
        file_path= data_path + 'user_id_map.tsv'
    elif name== 'item':
        file_path= data_path + 'item_id_map.tsv'
    else:
        raise NotImplementedError
    
    id_map = {}
    for line in open(file_path):
        k, v = line.strip().split('\t')
        id_map[k]=str(k)
    return id_map

def load_data(data_path):
    adj= load_adj(data_path)

    latest_sessions= load_latest_session(data_path)

    user_id_map= load_map(data_path, 'user')
    item_id_map= load_map(data_path, 'item')

    train_df= pd.read_csv(data_path+'train.tsv', sep='\t', dtype= {0:np.int32,1:np.int32})
    valid_df= pd.read_csv(data_path+'valid.tsv', sep='\t', dtype= {0:np.int32,1:np.int32})
    test_df= pd.read_csv(data_path+'test.tsv', sep='\t', dtype= {0:np.int32,1:np.int32})

    train_tensor= th.tensor(train_df.values, dtype=th.float32)
    valid_tensor= th.tensor(valid_df.values, dtype=th.float32)
    test_tensor= th.tensor(test_df.values, dtype=th.float32)

    return [adj, latest_sessions, user_id_map, item_id_map, train_tensor, valid_tensor, test_tensor]

if __name__== '__main__':
    
    data_path= PATH_TO_DATA
    data= load_data(data_path)