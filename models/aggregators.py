import torch as th
import torch.nn as nn
import torch.nn.functional as F

from layers import Layer, Dense
from inits import glorot, zeros

class MeanAggregator(Layer):

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0.0, bias=False, act=F.relu, name=None, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout= dropout
        self.bias= bias
        self.act= act
        self.concat= concat

        if neigh_input_dim is None:
            neigh_input_dim= input_dim

        self.vars['neigh_weights']= glorot((neigh_input_dim, output_dim))
        self.vars['self_weights']= glorot((input_dim, output_dim))
        if self.bias:
            self.vars['bias']= zeros((output_dim,))

        if self.logging:
            self._log_vars()

        self.input_dim= input_dim
        self.output_dim= output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs= inputs

        neigh_vecs= F.dropout(neigh_vecs, self.dropout, training=True)
        self_vecs= F.dropout(self_vecs, self.dropout, training=True)

        neigh_means= th.mean(neigh_vecs, dim=1)

        from_neighs= th.matmul(neigh_means, self.vars['neigh_weights'])
        from_self= th.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output= from_self + from_neighs
        else:
            output= th.cat([from_self, from_neighs], dim=1)

        if self.bias:
            output += self.vars['bias']

        return self.act(output)

class GCNAggregator(Layer):

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0.0, bias=False, act=F.relu, name=None, concat=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout= dropout
        self.bias= bias
        self.act= act
        self.concat= concat

        if neigh_input_dim is None:
            neigh_input_dim= input_dim

        self.vars['weights']= glorot((neigh_input_dim, output_dim))
        if self.bias:
            self.vars['bias']= zeros((output_dim,))

        if self.logging:
            self._log_vars()

        self.input_dim= input_dim
        self.output_dim= output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs= inputs

        neigh_vecs= F.dropout(neigh_vecs, self.dropout, training=True)
        self_vecs= F.dropout(self_vecs, self.dropout, training=True)

        means= th.mean(th.cat([neigh_vecs, self_vecs.unsqueeze(1)], dim=1), dim=1)

        output= th.matmul(means, self.vars['weights'])

        if self.bias:
            output += self.vars['bias']

        return self.act(output)
    

class AttentionAggregator(Layer):

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0.0, bias=False, act=F.relu, name=None, concat=False, **kwargs):
        super(AttentionAggregator, self).__init__(**kwargs)

        self.dropout= dropout
        self.bias= bias
        self.act= act
        self.concat= concat

        if neigh_input_dim is None:
            neigh_input_dim= input_dim

        self.vars['weights']= glorot((neigh_input_dim, output_dim))
        if self.bias:
            self.vars['bias']= zeros((output_dim,))

        if self.logging:
            self._log_vars()

        self.input_dim= input_dim
        self.output_dim= output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs= inputs

        neigh_vecs= F.dropout(neigh_vecs, self.dropout, training=True)
        self_vecs= F.dropout(self_vecs, self.dropout, training=True)

        query= self_vecs.unsqueeze(1)  # [batch_size, 1, input_dim]
        neigh_self_vecs= th.cat([neigh_vecs, query], dim=1)  # [batch_size, num_neighbors + 1, input_dim]

        score= th.bmm(query, neigh_self_vecs.transpose(1, 2))  # [batch_size, 1, num_neighbors + 1]
        score= F.softmax(score, dim=-1)  # [batch_size, 1, num_neighbors + 1]

        context= th.bmm(score, neigh_self_vecs).squeeze(1)  # [batch_size, input_dim]

        output= th.matmul(context, self.vars['weights'])

        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GatAggregator(Layer):

    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 dropout=0.0, bias=False, act=F.elu, name=None, concat=False, **kwargs):
        super(GatAggregator, self).__init__(**kwargs)

        self.dropout= dropout
        self.bias= bias
        self.act= act
        self.concat= concat
        self.gat_out_dim= 64  
        self.num_heads= output_dim // self.gat_out_dim  

        if neigh_input_dim is None:
            neigh_input_dim= input_dim

        self.neigh_denses= nn.ModuleList([Dense(neigh_input_dim, self.gat_out_dim, dropout=dropout) for _ in range(self.num_heads)])
        self.self_denses= nn.ModuleList([Dense(input_dim, self.gat_out_dim, dropout=dropout) for _ in range(self.num_heads)])

        self.attn_denses= nn.ModuleList([Dense(self.gat_out_dim, 1, dropout=dropout) for _ in range(self.num_heads)])

        if self.bias:
            self.vars['bias']= zeros((output_dim,))

        if self.logging:
            self._log_vars()

        self.input_dim= input_dim
        self.output_dim= output_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs= inputs

        neigh_vecs= F.dropout(neigh_vecs, self.dropout, training=True)
        self_vecs= F.dropout(self_vecs, self.dropout, training=True)

        self_attens= []
        neigh_attens= []

        for i in range(self.num_heads):
            neigh_red_vecs= self.neigh_denses[i](neigh_vecs)
            self_red_vecs= self.self_denses[i](self_vecs.unsqueeze(1)).squeeze(1)

            self_f= self.attn_denses[i](self_red_vecs.unsqueeze(1))
            neigh_f= self.attn_denses[i](neigh_red_vecs)
            logits= self_f + neigh_f.transpose(1, 2)
            coefs= F.softmax(F.relu(logits), dim=1)

            neigh_atten= th.bmm(coefs.transpose(1, 2), neigh_red_vecs).squeeze(1)
            self_atten= self_red_vecs
            self_attens.append(self_atten)
            neigh_attens.append(neigh_atten)

        from_self= th.cat(self_attens, dim=-1)
        from_neighs= th.cat(neigh_attens, dim=-1)

        if not self.concat:
            output= from_self + from_neighs
        else:
            output= th.cat([from_self, from_neighs], dim=1)

        if self.bias:
            output += self.vars['bias']
        
        return self.act(output)

class MaxPoolingAggregator(Layer):

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0.0, bias=False, act=F.relu, name=None, concat=False, **kwargs):
        
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout= dropout
        self.bias= bias
        self.act= act
        self.concat= concat

        if neigh_input_dim is None:
            neigh_input_dim= input_dim

        hidden_dim= 512 if model_size== "small" else 1024

        self.mlp_layers= nn.ModuleList([Dense(input_dim=neigh_input_dim, output_dim=hidden_dim, 
                                               act=F.relu, dropout=dropout, logging=self.logging)])

        self.vars['neigh_weights']= glorot((hidden_dim, output_dim))
        self.vars['self_weights']= glorot((input_dim, output_dim))
        
        if self.bias:
            self.vars['bias']= zeros((output_dim,))

        if self.logging:
            self._log_vars()

        self.input_dim= input_dim
        self.output_dim= output_dim
        self.neigh_input_dim= neigh_input_dim
        self.hidden_dim= hidden_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs= inputs

        neigh_h= F.dropout(neigh_vecs, self.dropout, training=True)

        batch_size, num_neighbors, _= neigh_h.shape
        h_reshaped= neigh_h.view(batch_size * num_neighbors, self.neigh_input_dim)

 
        for layer in self.mlp_layers:
            h_reshaped= layer(h_reshaped)
 
        neigh_h= h_reshaped.view(batch_size, num_neighbors, self.hidden_dim)
        neigh_h= th.max(neigh_h, dim=1).values 


        from_neighs= th.matmul(neigh_h, self.vars['neigh_weights'])
        from_self= th.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output= from_self + from_neighs
        else:
            output= th.cat([from_self, from_neighs], dim=1)

        if self.bias:
            output += self.vars['bias']
        
        return self.act(output)


class MeanPoolingAggregator(Layer):

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0.0, bias=False, act=F.relu, name=None, concat=False, **kwargs):
        
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        self.dropout= dropout
        self.bias= bias
        self.act= act
        self.concat= concat

        if neigh_input_dim is None:
            neigh_input_dim= input_dim

        hidden_dim= 512 if model_size== "small" else 1024

        self.mlp_layers= nn.ModuleList([Dense(input_dim=neigh_input_dim, output_dim=hidden_dim, 
                                               act=F.relu, dropout=dropout, logging=self.logging)])

        self.vars['neigh_weights']= glorot((hidden_dim, output_dim))
        self.vars['self_weights']= glorot((input_dim, output_dim))
        
        if self.bias:
            self.vars['bias']= zeros((output_dim,))

        if self.logging:
            self._log_vars()

        self.input_dim= input_dim
        self.output_dim= output_dim
        self.neigh_input_dim= neigh_input_dim
        self.hidden_dim= hidden_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs= inputs

        neigh_h= F.dropout(neigh_vecs, self.dropout, training=True)

        batch_size, num_neighbors, _= neigh_h.shape
        h_reshaped= neigh_h.view(batch_size * num_neighbors, self.neigh_input_dim)

        for layer in self.mlp_layers:
            h_reshaped= layer(h_reshaped)

        neigh_h= h_reshaped.view(batch_size, num_neighbors, self.hidden_dim)
        neigh_h= th.mean(neigh_h, dim=1)  

        from_neighs= th.matmul(neigh_h, self.vars['neigh_weights'])
        from_self= th.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output= from_self + from_neighs
        else:
            output= th.cat([from_self, from_neighs], dim=1)

        if self.bias:
            output += self.vars['bias']
        
        return self.act(output)
    
class SeqAggregator(nn.Module):

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0., bias=False, act=F.relu, concat=False):
        super(SeqAggregator, self).__init__()

        self.dropout= dropout
        self.bias= bias
        self.act= act
        self.concat= concat

        if neigh_input_dim is None:
            neigh_input_dim= input_dim

        hidden_dim= 128 if model_size== "small" else 256

        self.lstm= nn.GRU(neigh_input_dim, hidden_dim, batch_first=True)

        self.neigh_weights= nn.Parameter(th.empty(hidden_dim, output_dim))
        self.self_weights= nn.Parameter(th.empty(input_dim, output_dim))
        nn.init.xavier_normal_(self.neigh_weights)
        nn.init.xavier_normal_(self.self_weights)

        if self.bias:
            self.bias_param= nn.Parameter(th.zeros(output_dim))

    def forward(self, self_vecs, neigh_vecs):

        lengths= (neigh_vecs.abs().sum(dim=2) != 0).sum(dim=1).cpu()
        packed_input= nn.utils.rnn.pack_padded_sequence(neigh_vecs, lengths, batch_first=True, enforce_sorted=False)

        packed_output, _= self.lstm(packed_input)
        lstm_output, _= nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        idx= (lengths- 1).view(-1, 1).expand(-1, lstm_output.size(2)).unsqueeze(1)
        neigh_h= lstm_output.gather(1, idx).squeeze(1)

        from_neighs= th.matmul(neigh_h, self.neigh_weights)
        from_self= th.matmul(self_vecs, self.self_weights)

        if self.concat:
            output= th.cat([from_self, from_neighs], dim=1)
        else:
            output= from_self + from_neighs
        if self.bias:
            output += self.bias_param

        return self.act(output)

class SeqAggregator(Layer):

    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
                 dropout=0.0, bias=False, act=F.relu, name=None, concat=False, **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout= dropout
        self.bias= bias
        self.act= act
        self.concat= concat

        if neigh_input_dim is None:
            neigh_input_dim= input_dim

        hidden_dim= 128 if model_size== "small" else 256
        self.hidden_dim= hidden_dim

        self.gru= nn.GRU(input_size=neigh_input_dim, hidden_size=hidden_dim, batch_first=True)

        self.vars['neigh_weights']= glorot((hidden_dim, output_dim))
        self.vars['self_weights']= glorot((input_dim, output_dim))
        
        if self.bias:
            self.vars['bias']= zeros((output_dim,))

        if self.logging:
            self._log_vars()

        self.input_dim= input_dim
        self.output_dim= output_dim
        self.neigh_input_dim= neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs= inputs

        batch_size= neigh_vecs.size(0)

        used= th.sign(th.max(th.abs(neigh_vecs), dim=2).values)
        length= used.sum(dim=1).int().clamp(min=1)

        packed_neigh_vecs= nn.utils.rnn.pack_padded_sequence(neigh_vecs, length.cpu(), batch_first=True, enforce_sorted=False)
        _, rnn_states= self.gru(packed_neigh_vecs)

        neigh_h= rnn_states[-1]

        from_neighs= th.matmul(neigh_h, self.vars['neigh_weights'])
        from_self= th.matmul(self_vecs, self.vars["self_weights"])

        if not self.concat:
            output= from_self + from_neighs
        else:
            output= th.cat([from_self, from_neighs], dim=1)

        if self.bias:
            output += self.vars['bias']
        
        return self.act(output)