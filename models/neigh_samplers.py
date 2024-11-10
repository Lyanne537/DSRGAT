from __future__ import division
from __future__ import print_function

import torch as th

class UniformNeighborSampler:

    def __init__(self, adj_info, visible_time, deg):
        self.adj_info= adj_info
        self.visible_time= visible_time
        self.deg= deg

    def __call__(self, inputs):
        nodeids, num_samples, timeids, first_or_second, support_size= inputs
        adj_lists= []

        for idx in range(len(nodeids)):
            node= nodeids[idx]
            timeid= timeids[idx//support_size]
            adj= self.adj_info[node,:]
            neighbors= []

        for neighbor in adj:
            if first_or_second == 'second':
                if self.visible_time[neighbor]<= timeid:
                    neighbors.append(neighbor)
            elif first_or_second == 'first':
                if self.visible_time[neighbor]<= timeid and self.deg[neighbor]> 0:
                    for second_neighbor in self.adj_info[neighbor]:
                        if self.visible_time[second_neighbor]<= self.adj_info[neighbor]:
                            neighbors.append(neighbor)
                        break
        
        assert len(neighbors) > 0,f"No neighbors found for node {node} at timeid {timeid}"


        neighbors= th.tensor(neighbors, dtype=th.int32)
        if len(neighbors)<  num_samples:
            neighbors= neighbors[th.randint(0, len(neighbors))]
        elif len(neighbors) > num_samples:
            neighbors= neighbors[th.randperm(len(neighbors))[:,num_samples]]

        adj_lists.append(neighbors)

        return th.stack(adj_lists)
                        


