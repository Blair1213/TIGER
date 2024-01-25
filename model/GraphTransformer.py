# -*- coding: utf-8 -*-
# @Time    : 2023/5/30 下午4:14
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : GraphTransformer.py
# @Software : PyCharm

import os
import sys

BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
import torch
import math
from torch.nn import TransformerEncoderLayer, TransformerEncoder, BCEWithLogitsLoss
from torch_geometric.nn import GCNConv, SAGEConv, GCN2Conv, GATConv, ECConv, global_mean_pool, GINConv
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn.conv import MessagePassing


class GraphTransformerEncode(torch.nn.Module):
    def __init__(self, num_heads, in_dim, dim_forward, rel_encoder, spatial_encoder, dropout):
        super(GraphTransformerEncode, self).__init__()

        self.num_heads = num_heads
        self.in_dim = in_dim
        self.dim_forward = dim_forward

        self.ffn = Sequential(
            Linear(self.in_dim, self.dim_forward),
            ReLU(),
            Linear(self.dim_forward, self.in_dim)
        )

        self.multiHeadAttention = MultiheadAttention(dim_model = self.in_dim, num_heads = self.num_heads, rel_encoder=rel_encoder, spatial_encoder = spatial_encoder)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=in_dim, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.ffn[0].reset_parameters()
        self.ffn[2].reset_parameters()

        self.multiHeadAttention.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()

    def forward(self, feature, sp_edge_index, sp_value, edge_rel):

        x_norm = self.layernorm1(feature)
        attn_output, attn_weight = self.multiHeadAttention(x_norm, sp_edge_index, sp_value, edge_rel)
        attn_output = self.dropout1(attn_output)
        out1 = attn_output + feature

        residual = out1
        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)
        ffn_output = self.dropout2(ffn_output)
        out2 = residual + ffn_output

        return out2, attn_weight

class SpatialEncoding(torch.nn.Module):
    def __init__(self, dim_model):
        super(SpatialEncoding, self).__init__()

        self.dim = dim_model
        self.fnn = Sequential(
            Linear(1, dim_model),
            ReLU(),
            Linear(dim_model, 1),
            ReLU()
        )

    def reset_parameters(self):
        self.fnn[0].reset_parameters()
        self.fnn[2].reset_parameters()

    def forward(self, lap):
        lap_ = torch.unsqueeze(lap, dim=-1) ##[n_edges, 1]
        out = self.fnn(lap_)

        return out


class MultiheadAttention(MessagePassing):
    def __init__(self, dim_model, num_heads, rel_encoder, spatial_encoder, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.d_model = dim_model
        self.num_heads = num_heads

        self.rel_embedding = rel_encoder
        self.rel_encoding = Sequential(
            Linear(dim_model, 1),
            ReLU()
        )

        self.spatial_encoding = spatial_encoder


        assert dim_model % num_heads == 0
        self.depth = self.d_model // num_heads

        self.wq = Linear(dim_model, dim_model)
        self.wk = Linear(dim_model, dim_model)
        self.wv = Linear(dim_model, dim_model)

        self.dense = Linear(dim_model, dim_model)

    def reset_parameters(self):
        self.rel_embedding.reset_parameters()
        self.rel_encoding[0].reset_parameters()
        self.spatial_encoding.reset_parameters()

        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.dense.reset_parameters()

    def softmax_kernel_transformation(self, data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
        data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))
        data = data_normalizer * data
        ratio = data_normalizer
        data_dash = projection_matrix(data) ##[node_num, dim]
        diag_data = torch.square(data)
        diag_data = torch.sum(diag_data, dim=len(data.shape) - 1)
        diag_data = diag_data / 2.0
        diag_data = torch.unsqueeze(diag_data, dim=len(data.shape) - 1)
        last_dims_t = len(data_dash.shape) - 1
        attention_dims_t = len(data_dash.shape) - 3
        if is_query:
            data_dash = ratio * (
                    torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[
                        0]) + numerical_stabilizer
            )
        else:
            data_dash = ratio * (
                    torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                                                                dim=attention_dims_t, keepdim=True)[
                        0]) + numerical_stabilizer
            )
        return data_dash

    def denominator(self, qs, ks):
        ##qs [num_node, num_heads, depth]
        all_ones = torch.ones([ks.shape[0]]).to(qs.device)
        ks_sum = torch.einsum("nhm,n->hm", ks, all_ones)  # ks_sum refers to O_k in the paper
        return torch.einsum("nhm,hm->nh", qs, ks_sum)

    def forward(self, x, sp_edge_index, sp_value, edge_rel):

        rel_embedding = self.rel_embedding(edge_rel)
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x).view(x.shape[0],self.num_heads,self.depth) ##[nodes_num, num_heads, depth]

        row, col = sp_edge_index
        query_end, key_start = q[col], k[row] ##[edge_nums, num_heads, depths]
        query_end += rel_embedding
        key_start += rel_embedding

        query_end = query_end.view(sp_edge_index.shape[1],self.num_heads,self.depth)
        key_start = key_start.view(sp_edge_index.shape[1],self.num_heads,self.depth)

        edge_attn_num = torch.einsum("ehd,ehd->eh", query_end, key_start) ##[edge_nums, num_heads]
        data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(edge_attn_num.shape[-1], dtype=torch.float32)))
        edge_attn_num *= data_normalizer
        edge_attn_bias = self.spatial_encoding(sp_value)
        edge_attn_num += edge_attn_bias

        attn_normalizer = self.denominator(q.view(x.shape[0],self.num_heads,self.depth), k.view(x.shape[0],self.num_heads,self.depth))
        edge_attn_dem = attn_normalizer[col] ##[edge_nums, num_heads]
        attention_weight = edge_attn_num / edge_attn_dem ##[edge_nums, num_heads] ##scaled

        outputs = []
        for i in range(self.num_heads):
            output_per_head = self.propagate(edge_index=sp_edge_index, x = v[:,i,:], edge_weight = attention_weight[:, i], size=None)
            outputs.append(output_per_head)

        out = torch.cat(outputs,dim=-1)

        return self.dense(out), attention_weight


class GraphTransformer(torch.nn.Module):
    def __init__(self, layer_num = 3, embedding_dim = 64, num_heads = 4, num_rel = 10, dropout = 0.2, type = 'graph'): ##type指示的是graph还是node，也就是对应的是图级别的表示学习，还是节点级别的表示学习
        super(GraphTransformer, self).__init__()

        self.type = type
        self.rel_encoder = torch.nn.Embedding(num_rel, embedding_dim)  ##权重共享的
        self.spatial_encoder = SpatialEncoding(embedding_dim)  ##这两个是权重共享的

        self.encoder = torch.nn.ModuleList()
        for i in range(layer_num - 1):
            self.encoder.append(GraphTransformerEncode(num_heads = num_heads, in_dim = embedding_dim, dim_forward = embedding_dim*2,
                                                       rel_encoder = self.rel_encoder, spatial_encoder = self.spatial_encoder, dropout=dropout))

    def reset_parameters(self):
        for e in self.encoder:
            e.reset_parameters()


    def forward(self, feature, data):

        ##首先就是按照edge index计算attn_weight, 然后按照权重聚合就可以了！！
        x = feature
        graph_embedding_layer = []
        attn_layer = []
        for graphEncoder in self.encoder:
            x, attn = graphEncoder(x, data.sp_edge_index, data.sp_value, data.sp_edge_rel)
            graph_embedding_layer.append(x)
            attn_layer.append(attn)

        #all_out = torch.stack([x for x in graph_embedding_layer])


        if self.type == 'graph':
            ##pooling
            sub_representation = []
            for index, drug_mol_graph in enumerate(data.to_data_list()):
                sub_embedding = x[(data.batch == index).nonzero().flatten()]  ##第index个图中的各个节点的表示，[atom_number, emd_dim]
                sub_representation.append(sub_embedding)
            representation = global_mean_pool(x, batch=data.batch)  ##每个drug分子的图的表示
        else:
            ##只返回第一个
            sub_representation = []
            for index, drug_subgraph in enumerate(data.to_data_list()):
                sub_embedding = x[(data.batch == index).nonzero().flatten()]
                #print(sub_embedding.shape)
                sub_representation.append(sub_embedding) ##只取那个节点的embedding
            #print(x.shape)
            #print(data.id.shape)
            representation = x[data.id.nonzero().flatten()]

        return representation, sub_representation, attn_layer

        ##对于节点级别的表示，需要每一层的级联，然后做最后的互信息最大化，这个层级的优化可能要考虑一下，但是最终落到的还是节点和图
