# -*- coding: utf-8 -*-
# @Time    : 2023/5/28 下午6:28
# @Author  : xiaorui su
# @Email   :  suxiaorui19@mails.ucas.edu.cn
# @File    : main.py.py
# @Software : PyCharm

import os
import argparse
import gc
import torch
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import json
import copy
from utils import *
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from model.tiger import TIGER
from torch_geometric.utils import degree
from torch.utils.data.distributed import DistributedSampler
from data_process import smile_to_graph, read_smiles, read_interactions, generate_node_subgraphs, read_network
from sklearn.model_selection import StratifiedKFold, KFold
from train_eval import train, test, eval

import random

def init_args(user_args=None):

    parser = argparse.ArgumentParser(description='TIGER')

    parser.add_argument('--model_name', type=str, default='tiger')

    parser.add_argument('--dataset', type=str, default="drugbank")

    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--layer', type=int, default=2)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--model_episodes', type=int, default=50)
    parser.add_argument('--extractor', type=str, default="randomWalk") ##option [khop-subtree, randomWalk, probability]
    parser.add_argument('--graph_fixed_num', type=int, default=1)
    parser.add_argument('--khop', type=int, default=2)
    parser.add_argument('--fixed_num', type=int, default=32)

    # Graphormer
    parser.add_argument("--d_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--max_smiles_degree", type=int, default=300)
    parser.add_argument("--max_graph_degree", type=int, default=600)
    parser.add_argument("--dropout", type=float, default=0.2)


    # coeff
    parser.add_argument('--sub_coeff', type=float, default=0.1)
    parser.add_argument('--mi_coeff', type=float, default=0.1)

    parser.add_argument('--s_type', type=str, default='random')

    args = parser.parse_args()

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def k_fold(data, kf, folds, y):

    test_indices = []
    train_indices = []

    if len(y):
        for _, idx in kf.split(torch.zeros(len(data)), y):
            test_indices.append(idx)
    else:
        for _, idx in kf.split(data):
            test_indices.append(idx)

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(data), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

def split_fold(folds, dataset, labels, scenario_type='random'):

    test_indices, train_indices, val_indices = [], [], []

    if scenario_type == 'random':##这是根据interactions在划分的数据集，也就是根据interactions的label进行的数据集划分
        skf = StratifiedKFold(folds, shuffle=True, random_state=2023)
        train_indices, test_indices, val_indices = k_fold(dataset, skf, folds, labels)

    return train_indices, test_indices, val_indices

def load_data(args):

    ##这个逻辑是这样的:先读出ddi中所有drug的数量，并不一定是全部的drug

    dataset = args.dataset

    data_path = "dataset/" + dataset + "/"

    ligands = read_smiles(os.path.join(data_path, "drug_smiles.txt"))


    # smiles to graphs
    print("load drug smiles graphs!!")
    smile_graph, num_rel_mol_update, max_smiles_degree = smile_to_graph(data_path, ligands)

    print("load networks !!")
    num_node, network_edge_index, network_rel_index, num_rel = read_network(data_path + "networks.txt")

    print("load DDI samples!!")
    interactions_label, all_contained_drgus = read_interactions(os.path.join(data_path, "ddi.txt"), smile_graph)
    interactions = interactions_label[:, :2]
    labels = interactions_label[:, 3]


    print("generate subgraphs!!")
    drug_subgraphs, max_subgraph_degree, num_rel_update = generate_node_subgraphs(dataset, all_contained_drgus,
                                                                                  network_edge_index, network_rel_index,
                                                                                  num_rel, args)

    data_sta = {
        'num_nodes': num_node + 1,
        'num_rel_mol': num_rel_mol_update + 1,
        'num_rel_graph': num_rel_update + 1,
        'num_interactions': len(interactions),
        'num_drugs_DDI': len(all_contained_drgus),
        'max_degree_graph': max_smiles_degree + 1,
        'max_degree_node': int(max_subgraph_degree)+1
    }

    print(data_sta)

    return interactions, labels, smile_graph, drug_subgraphs, data_sta

def save(save_dir, args, train_log, test_log):
    args.device = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + "/args.json", 'w') as f:
        json.dump(args.__dict__, f)
    with open(save_dir + '/test_results.json', 'w') as f:
        json.dump(test_log, f)
    with open(save_dir + '/train_log.json', 'w') as f:
        json.dump(train_log, f)

def save_results(save_dir, args, results_list):
    acc = []
    auc = []
    aupr = []
    f1 = []

    for r in results_list:
        acc.append(r['acc'])
        auc.append(r['auc'])
        aupr.append(r['aupr'])
        f1.append(r['f1'])

    acc = np.array(acc)
    auc = np.array(auc)
    aupr = np.array(aupr)
    f1 = np.array(f1)

    results = {
        'acc':[np.mean(acc),np.std(acc)],
        'auc':[np.mean(auc),np.std(auc)],
        'aupr': [np.mean(aupr), np.std(aupr)],
        'f1': [np.mean(f1), np.std(f1)],
    }

    args = vars(args)
    args.update(results)

    with open(save_dir + args['extractor'] + '_all_results.json', 'a+') as f:
        json.dump(args, f)


def init_model(args, dataset_statistics):
    if args.model_name == 'tiger':
        model = TIGER(max_layer=args.layer,
                      num_features_drug = 67,
                      num_nodes=dataset_statistics['num_nodes'],
                      num_relations_mol=dataset_statistics['num_rel_mol'],
                      num_relations_graph=dataset_statistics['num_rel_graph'],
                      output_dim=args.d_dim,
                      max_degree_graph=dataset_statistics['max_degree_graph'],
                      max_degree_node = dataset_statistics['max_degree_node'],
                      sub_coeff=args.sub_coeff,
                      mi_coeff=args.mi_coeff,
                      dropout=args.dropout,
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    return model, optimizer

def main(args = None, k_fold = 5):

    if args is None:
        args = init_args()

    results_of_each_fold = []

    ##加载interactions的data
    data, labels, smile_graph, node_graph, dataset_statistics = load_data(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    setup_seed(42)
    ##split datasets
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*split_fold(k_fold, data, labels, args.s_type))):
        print(f"============================{fold+1}/{k_fold}==================================")
        print("loading data!!")
        ##load_data
        train_data = DTADataset(x=data[train_idx], y=labels[train_idx], sub_graph=node_graph, smile_graph=smile_graph)
        test_data = DTADataset(x=data[test_idx], y=labels[test_idx], sub_graph=node_graph, smile_graph=smile_graph)
        eval_data = DTADataset(x=data[val_idx], y=labels[val_idx], sub_graph=node_graph, smile_graph=smile_graph)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)  ##用DataLoader加载的数据，index是会自动增加的！！
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)  ##用DataLoader加载的数据，index是会自动增加的！！
        eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)  ##用DataLoader加载的数据，index是会自动增加的！！

        if args.model_name:
            model, optimizer = init_model(args, dataset_statistics)
            model.to(device)
            model.reset_parameters()

            ##train_model
            #trange = tqdm(range(1, args.model_episodes + 1))
            best_auc = 0.0
            early_stop_num = 0

            train_log = {'train_acc':[], 'train_auc':[], 'train_aupr':[], 'train_loss':[],
                         'eval_acc':[], 'eval_auc':[], 'eval_aupr':[], 'eval_loss':[]}

            for i_episode in range(args.model_episodes):
                loop = tqdm(train_loader, ncols=80)
                loop.set_description(f'Epoch[{i_episode}/{args.model_episodes}]')
                train_acc, train_f1, train_auc, train_aupr, train_loss = train(loop, model, optimizer)
                eval_acc, eval_f1, eval_auc, eval_aupr, eval_loss = eval(eval_loader, model)
                print(f"train_auc:{train_auc} train_aupr:{train_aupr} eval_auc:{eval_auc} eval_aupr:{eval_aupr}")

                train_log['train_acc'].append(train_acc)
                train_log['train_auc'].append(train_auc)
                train_log['train_aupr'].append(train_aupr)
                train_log['train_loss'].append(train_loss)

                train_log['eval_acc'].append(eval_acc)
                train_log['eval_auc'].append(eval_auc)
                train_log['eval_aupr'].append(eval_aupr)
                train_log['eval_loss'].append(eval_loss)

                if eval_auc > best_auc:
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_auc = eval_auc
                    early_stop_num = 0
                else:
                    early_stop_num += 1
                    if early_stop_num > 10:
                        print("early stop!")
                        break

            model.load_state_dict(best_model_state)
            model.to(device)
            test_log = test(test_loader, model) ##test_log是一个字典，里面存储着metrics

            save_dir = os.path.join('./best_save/', args.model_name, args.dataset, args.extractor,
                                    "fold_{}".format(fold), "{:.5f}".format(test_log['auc']))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model.save(save_dir)  ##保存当前最好的模型
            save(save_dir, args, train_log, test_log)
            print(f"save to {save_dir}")
            results_of_each_fold.append(test_log)


    save_results(os.path.join('./best_save/', args.model_name, args.dataset), args, results_of_each_fold)

    return ;


if __name__ == "__main__":
    main()
