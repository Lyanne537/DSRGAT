import torch as th
import torch.nn as nn
import torch.nn.functional as F
from aggregators import GatAggregator
from layers import Dense

class DSRGAT(nn.Module):
    def __init__(self, args, support_sizes):
        super(DSRGAT, self).__init__()
        self.args= args
        self.support_sizes= support_sizes

        # Embedding layers for items and users
        self.embedding= nn.Embedding(args.num_items, args.embedding_size)
        self.user_embedding= nn.Embedding(args.num_users, args.emb_user)

        # Transformer encoder for global features (long-term interest)
        encoder_layer= nn.TransformerEncoderLayer(d_model=args.emb_user, nhead=8, dim_feedforward=args.hidden_size)
        self.transformer_encoder= nn.TransformerEncoder(encoder_layer, num_layers=2)

        # GRU for local feature extraction (short-term interest)
        self.gru= nn.GRU(input_size=args.embedding_size, hidden_size=args.hidden_size, batch_first=True)

        self.gat_aggregator= GatAggregator(
            input_dim=args.hidden_size, output_dim=args.hidden_size, dropout=args.dropout
        )

        # Fully connected layer 
        self.fc= nn.Linear(args.dim2 + args.hidden_size, args.embedding_size)

    def global_features(self, support_nodes_layer1, support_nodes_layer2):
        # Compute long-term (global) interests using Transformer
        feature_layer1= self.user_embedding(support_nodes_layer1)
        feature_layer2= self.user_embedding(support_nodes_layer2)

        feature_layer1= self.transformer_encoder(feature_layer1.permute(1, 0, 2)).permute(1, 0, 2)
        feature_layer2= self.transformer_encoder(feature_layer2.permute(1, 0, 2)).permute(1, 0, 2)

        return feature_layer1, feature_layer2

    def local_features(self, support_sessions_layer1, support_sessions_layer2):
        # Compute short-term (local) interests using GRU
        inputs_1= self.embedding(support_sessions_layer1)
        inputs_2= self.embedding(support_sessions_layer2)

        _, states1= self.gru(inputs_1)
        _, states2= self.gru(inputs_2)

        return states1[-1], states2[-1]

    def global_and_local_features(self, support_nodes_layer1, support_nodes_layer2, support_sessions_layer1, support_sessions_layer2):
        # Combine both global and local features
        global_feature_layer2, global_feature_layer1= self.global_features(support_nodes_layer1, support_nodes_layer2)
        local_feature_layer2, local_feature_layer1= self.local_features(support_sessions_layer1, support_sessions_layer2)
        global_local_layer2= th.cat([global_feature_layer2, local_feature_layer2], dim=-1)
        global_local_layer1= th.cat([global_feature_layer1, local_feature_layer1], dim=-1)
        return global_local_layer2, global_local_layer1

    def step_by_step(self, features_0, features_1_2, dims, num_samples, support_sizes, concat=False):
        # Aggregation with GAT to combine global and local features
        outputs= []
        hidden= [features_0, features_1_2[0], features_1_2[1]]
        output, _= self.aggregate(hidden, dims, num_samples, support_sizes, concat=concat)
        outputs.append(output)
        return th.stack(outputs, axis=0)

    def aggregate(self, hidden, dims, num_samples, support_sizes, concat=False):
        """GAT-based aggregation to compute next-layer hidden representations."""
        outputs= []
        for layer in range(len(num_samples)):
            dim_mult= 2 if concat and layer != 0 else 1
            aggregator= self.gat_aggregator
            next_hidden= []
            for hop in range(len(num_samples) - layer):
                h= aggregator((hidden[hop], hidden[hop + 1].view(-1, num_samples[layer], dims[layer])))
                next_hidden.append(h)
            hidden= next_hidden
            outputs.append(hidden[0])
        return outputs[-1], None

    def _loss(self, input_y, mask_y):

        fc_output= self.fc(self.transposed_outputs.view(-1, self.args.dim2 + self.hidden_size))
        logits= th.matmul(fc_output, self.embedding.weight.t())
        reshaped_logits= logits.view(self.args.batch_size, self.args.max_length, self.args.num_items)
        xe_loss= F.cross_entropy(reshaped_logits.transpose(1, 2), input_y, reduction='none') * mask_y
        reg_loss= sum([param.norm(2) for param in self.parameters()]) * self.args.weight_decay
        return xe_loss.sum() / mask_y.float().sum() + reg_loss

    def _ndcg(self, predictions, targets, k=20):

        ranks= th.argsort(predictions, dim=-1, descending=True)
        gains= th.where(ranks== targets.unsqueeze(-1), 1 / th.log2(th.arange(2, k + 2, dtype=th.float32)), th.zeros_like(ranks, dtype=th.float32))
        ndcg= gains.sum(dim=-1)
        return ndcg.mean()

    def _recall(self, predictions, targets, k=20):

        top_k_preds= th.topk(predictions, k, dim=-1).indices
        recall_at_k= (top_k_preds== targets.unsqueeze(-1)).float().sum() / targets.numel()
        return recall_at_k

    def forward(self, input_x, support_nodes_layer1, support_nodes_layer2, support_sessions_layer1, support_sessions_layer2):

        if self.args.global_only:
            features_1_2= self.global_features(support_nodes_layer1, support_nodes_layer2)
        elif self.args.local_only:
            features_1_2= self.local_features(support_sessions_layer1, support_sessions_layer2)
        else:
            features_1_2= self.global_and_local_features(support_nodes_layer1, support_nodes_layer2, support_sessions_layer1, support_sessions_layer2)


        features_0= self.embedding(input_x)

        aggregated_features= self.step_by_step(features_0, features_1_2, self.args.dims, self.args.num_samples, self.support_sizes)
        logits= self.fc(th.cat([features_0, aggregated_features], dim=-1))
        return logits
