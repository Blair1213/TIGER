# -*- coding: utf-8 -*-
# @Time    : 2023/5/31 下午3:34
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : utils.py
# @Software : PyCharm


import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch, Dataset
from torch_geometric import data as DATA
import torch
import numpy as np


# initialize the dataset
class DTADataset(InMemoryDataset):
    def __init__(self, x=None, y=None, sub_graph=None, smile_graph=None, ):
        super(DTADataset, self).__init__()

        self.labels = y
        self.drug_ID = x
        self.sub_graph = sub_graph
        self.smile_graph = smile_graph
        #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2 = self.process(x, y, sub_graph, smile_graph)

    def read_drug_info(self, drug_id, labels):

        c_size, features, edge_index, rel_index, sp_edge_index, sp_value, sp_rel, deg = self.smile_graph[str(drug_id)]  ##drug——id是str类型的，不是int型的，这点要注意
        subset, subgraph_edge_index, subgraph_rel, mapping_id, s_edge_index, s_value, s_rel, deg = self.sub_graph[str(drug_id)]

        data_mol = DATA.Data(x=torch.Tensor(np.array(features)),
                              edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                              y=torch.LongTensor([labels]),
                              rel_index=torch.Tensor(np.array(rel_index, dtype=int)),
                              sp_edge_index=torch.LongTensor(sp_edge_index).transpose(1, 0),
                              sp_value=torch.Tensor(np.array(sp_value, dtype=int)),
                              sp_edge_rel=torch.LongTensor(np.array(sp_rel, dtype=int))
                              )
        data_mol.__setitem__('c_size', torch.LongTensor([c_size]))

        data_graph = DATA.Data(x=torch.LongTensor(subset),
                                edge_index=torch.LongTensor(subgraph_edge_index).transpose(1,0),
                                y=torch.LongTensor([labels]),
                                id=torch.LongTensor(np.array(mapping_id, dtype=bool)),
                                rel_index=torch.Tensor(np.array(subgraph_rel, dtype=int)),
                                sp_edge_index=torch.LongTensor(s_edge_index).transpose(1, 0),
                                sp_value=torch.Tensor(np.array(s_value, dtype=int)),
                                sp_edge_rel=torch.LongTensor(np.array(s_rel, dtype=int))
                                )

        return data_mol, data_graph

    def __len__(self):
        #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2
        return len(self.drug_ID)

    def __getitem__(self, idx):
        drug1_id = self.drug_ID[idx, 0]
        drug2_id = self.drug_ID[idx, 1]
        labels = int(self.labels[idx])

        drug1_mol, drug1_subgraph = self.read_drug_info(drug1_id, labels)
        drug2_mol, drug2_subrgraph = self.read_drug_info(drug2_id, labels)

        return drug1_mol, drug1_subgraph, drug2_mol, drug2_subrgraph


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])

    return batchA, batchB, batchC, batchD
