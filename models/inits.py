import torch as th
import torch.nn as nn
import math

def uniform(shape, scale=0.05):
    initial= th.empty(shape).uniform_(-scale, scale)
    return nn.Parameter(initial)


def glorot(shape):
    init_range= math.sqrt(6.0/(shape[0]+shape[1]))
    initial= th.empty(shape).uniform_(-init_range, init_range)
    return nn.Parameter(initial)


def zeros(shape):
    initial= th.zeros(shape)
    return nn.Parameter(initial)