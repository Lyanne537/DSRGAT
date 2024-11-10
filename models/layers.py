import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from inits import zeros, glorot

_LAYER_UIDS= {}

def get_layer_uid(layer_name=''):

    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name]= 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer:

    def __init__(self, **kwargs):
        allowed_kwargs= {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name= kwargs.get('name')
        if not name:
            layer= self.__class__.__name__.lower()
            name= layer + '_' + str(get_layer_uid(layer))
        self.name= name
        self.vars= {}
        self.logging= kwargs.get('logging', False)
        self.sparse_inputs= False
        self.writer= SummaryWriter() if self.logging else None

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        outputs= self._call(inputs)
        if self.logging and self.writer:
            self.writer.add_histogram(f"{self.name}/inputs", inputs)
            self.writer.add_histogram(f"{self.name}/outputs", outputs)
        return outputs

    def _log_vars(self):
        if self.logging and self.writer:
            for var_name, var in self.vars.items():
                self.writer.add_histogram(f"{self.name}/vars/{var_name}", var)

class Dense(Layer):

    def __init__(self, input_dim, output_dim, dropout=0.0, weight_decay=0.0, 
                 act=F.relu, bias=True, featureless=False, sparse_inputs=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout= dropout
        self.weight_decay= weight_decay
        self.act= act
        self.featureless= featureless
        self.bias= bias
        self.input_dim= input_dim
        self.output_dim= output_dim
        self.sparse_inputs= sparse_inputs

        self.vars['weights']= glorot((input_dim, output_dim))
        if self.bias:
            self.vars['bias']= zeros((output_dim,))

        if self.logging:
            self._log_vars()

    def _call(self, inputs):

        x= F.dropout(inputs, self.dropout, training=True)
        output= th.matmul(x, self.vars['weights'])

        if self.bias:
            output += self.vars['bias']

        return self.act(output)
