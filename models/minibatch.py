import numpy as np
import pandas as pd
import torch as th
from neigh_samplers import UniformNeighborSampler
from ..utils.dataloader import load_data
PATH_TO_DATA= './movie/'
np.random.seed(123)

class MinibatchIterator:
    
    def __init__(self, 
                 adj_info, 
                 latest_sessions,
                 data, 
                 batch_size,
                 max_degree,
                 num_nodes,
                 max_length=20,
                 samples_1_2=[10, 5],
                 training=True):
        
        self.num_layers= 2
        self.adj_info= adj_info
        self.latest_sessions= latest_sessions
        self.training= training
        self.train_df, self.valid_df, self.test_df= data
        self.all_data= pd.concat(data)
        self.batch_size= batch_size
        self.max_degree= max_degree
        self.num_nodes= num_nodes
        self.max_length= max_length
        self.samples_1_2= samples_1_2
        self.sizes= [1, samples_1_2[1], samples_1_2[1] * samples_1_2[0]]
        self.visible_time= self.user_visible_time()
        
        self.test_adj, self.test_deg= self.construct_test_adj()
        if self.training:
            self.adj, self.deg= self.construct_adj()
            self.train_session_ids= self._remove_infoless(self.train_df, self.adj, self.deg)
            self.valid_session_ids= self._remove_infoless(self.valid_df, self.test_adj, self.test_deg)
            self.sampler= UniformNeighborSampler(self.adj, self.visible_time, self.deg)
        
        self.test_session_ids= self._remove_infoless(self.test_df, self.test_adj, self.test_deg)
        self.padded_data, self.mask= self._padding_sessions(self.all_data)
        self.test_sampler= UniformNeighborSampler(self.test_adj, self.visible_time, self.test_deg)
        
        self.batch_num= 0
        self.batch_num_val= 0
        self.batch_num_test= 0

    def user_visible_time(self):
        visible_time= []
        for l in self.latest_sessions:
            timeid= max(loc for loc, val in enumerate(l) if val== 'NULL') + 1
            visible_time.append(timeid)
            assert timeid > 0 and timeid < len(l), 'Error in visible time creation: {}'.format(timeid)
        return visible_time

    def _remove_infoless(self, data, adj, deg):
        data= data.loc[deg[data['UserId']] != 0]
        reserved_session_ids= []
        for sessid in data.SessionId.unique():
            userid, timeid= sessid.split('_')
            userid, timeid= int(userid), int(timeid)
            cn_1= 0
            for neighbor in adj[userid, :]:
                if self.visible_time[neighbor] <= timeid and deg[neighbor] > 0:
                    cn_2= 0
                    for second_neighbor in adj[neighbor, :]:
                        if self.visible_time[second_neighbor] <= timeid:
                            break
                        cn_2 += 1
                    if cn_2 < self.max_degree:
                        break
                cn_1 += 1
            if cn_1 < self.max_degree:
                reserved_session_ids.append(sessid)
        return reserved_session_ids

    def _padding_sessions(self, data):
        data= data.sort_values(by=['TimeId']).groupby('SessionId')['ItemId'].apply(list).to_dict()
        new_data= {}
        data_mask= {}
        for k, v in data.items():
            mask= np.ones(self.max_length, dtype=np.float32)
            x= v[:-1]
            y= v[1:]
            padded_len= self.max_length - len(x)
            if padded_len > 0:
                x.extend([0] * padded_len)
                y.extend([0] * padded_len)
                mask[-padded_len:]= 0.
            new_data[k]= [np.array(x, dtype=np.int32), np.array(y, dtype=np.int32), np.array(v[:self.max_length], dtype=np.int32)]
            data_mask[k]= np.array(mask, dtype=bool)
        return new_data, data_mask

    def sample(self, nodeids, timeids, sampler):
        samples= [nodeids]
        support_size= 1
        support_sizes= [support_size]
        first_or_second= ['second', 'first']
        for k in range(self.num_layers):
            t= self.num_layers - k - 1
            node= sampler([samples[k], self.samples_1_2[t], timeids, first_or_second[t], support_size])
            support_size *= self.samples_1_2[t]
            samples.append(node.reshape(support_size * self.batch_size,))
            support_sizes.append(support_size)
        return samples, support_sizes

    def next_val_minibatch_feed_dict(self, val_or_test='val'):
        data= self.valid_session_ids if val_or_test== 'val' else self.test_session_ids
        batch_num= self.batch_num_val if val_or_test== 'val' else self.batch_num_test
        start= batch_num * self.batch_size
        current_batch_sessions= data[start: start + self.batch_size]
        nodes= [int(sess.split('_')[0]) for sess in current_batch_sessions]
        timeids= [int(sess.split('_')[1]) for sess in current_batch_sessions]
        samples, support_sizes= self.sample(nodes, timeids, self.test_sampler)
        return self._batch_feed_dict([current_batch_sessions, samples, support_sizes])

    def next_train_minibatch_feed_dict(self):
        start= self.batch_num * self.batch_size
        current_batch_sessions= self.train_session_ids[start: start + self.batch_size]
        nodes= [int(sess.split('_')[0]) for sess in current_batch_sessions]
        timeids= [int(sess.split('_')[1]) for sess in current_batch_sessions]
        samples, support_sizes= self.sample(nodes, timeids, self.sampler)
        return self._batch_feed_dict([current_batch_sessions, samples, support_sizes])

    def construct_adj(self):
        adj= np.full((self.num_nodes + 1, self.max_degree), self.num_nodes, dtype=np.int32)
        deg= np.zeros((self.num_nodes,))
        for nodeid in self.train_df.UserId.unique():
            neighbors= self.adj_info.loc[self.adj_info['Follower']== nodeid].Followee.unique().astype(np.int32)
            deg[nodeid]= len(neighbors)
            neighbors= np.random.choice(neighbors, self.max_degree, replace=len(neighbors) < self.max_degree)
            adj[nodeid, :]= neighbors
        return adj, deg

    def construct_test_adj(self):
        adj= np.full((self.num_nodes + 1, self.max_degree), self.num_nodes, dtype=np.int32)
        deg= np.zeros((self.num_nodes,))
        for nodeid in self.all_data.UserId.unique():
            neighbors= self.adj_info.loc[self.adj_info['Follower']== nodeid].Followee.unique().astype(np.int32)
            deg[nodeid]= len(neighbors)
            neighbors= np.random.choice(neighbors, self.max_degree, replace=len(neighbors) < self.max_degree)
            adj[nodeid, :]= neighbors
        return adj, deg

    def end(self):
        return self.batch_num * self.batch_size > len(self.train_session_ids) - self.batch_size

    def end_val(self, val_or_test='val'):
        data= self.valid_session_ids if val_or_test== 'val' else self.test_session_ids
        end= self.batch_num_val * self.batch_size > len(data) - self.batch_size if val_or_test== 'val' else self.batch_num_test * self.batch_size > len(data) - self.batch_size
        if end:
            self.batch_num_val= 0 if val_or_test== 'val' else self.batch_num_test= 0
        return end

    def shuffle(self):
        self.train_session_ids= np.random.permutation(self.train_session_ids)
        self.batch_num= 0

if __name__ == '__main__':
    data= load_data(PATH_TO_DATA)
    adj_info, latest_sessions, user_id_map, item_id_map, train_df, valid_df, test_df= data
    minibatch= MinibatchIterator(adj_info, latest_sessions, [train_df, valid_df, test_df], batch_size=1, max_degree=50, num_nodes=len(user_id_map), max_length=20, samples_1_2=[10, 5])
