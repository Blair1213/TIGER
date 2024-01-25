import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict
from rdkit import Chem
import torch
from rdkit.Chem import MolFromSmiles
from multiprocessing import Pool
import networkx as nx
from randomWalk import Node2vec
from torch_geometric.utils import subgraph, degree, get_laplacian
from utils import *
from torch import Tensor
import numpy as np


e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()]), atom.GetDegree()


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(datapath, ligands):

    smile_graph = {}

    paths = datapath + "/mol_sp.json"

    if os.path.exists(paths):
        with open(paths, 'r') as f:
            smile_graph = json.load(f)
        max_rel = 0
        max_degree = 0
        for s in smile_graph.keys():
            max_rel = max(smile_graph[s][6]) if max(smile_graph[s][6]) > max_rel else max_rel
            max_degree = smile_graph[s][7] if smile_graph[s][7] > max_degree else max_degree

        return smile_graph, max_rel, max_degree

    smiles_max_node_degree = []
    num_rel_mol_update = 0
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]))  ##还是smiles序列
        c_size, features, edge_index, rel_index, s_edge_index, s_value, s_rel, deg = single_smile_to_graph(lg)
        if c_size == 0: ##证明这个药物只由一个atom组成，这种的不考虑
            continue
        if max(s_value) > num_rel_mol_update:
            num_rel_mol_update = max(s_value)
        smile_graph[d] = c_size, features, edge_index, rel_index, s_edge_index, s_value, s_rel, deg
        smiles_max_node_degree.append(deg)

    with open(paths, 'w') as f:
        json.dump(smile_graph, f)

    return smile_graph, num_rel_mol_update, max(smiles_max_node_degree)


# mol smile to mol graph edge index
def single_smile_to_graph(smile):

    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    degrees = []
    for atom in mol.GetAtoms():
        feature, degree = atom_features(atom)
        features.append((feature / sum(feature)).tolist())
        degrees.append(degree)

    mol_index = []  ##begin, end, rel
    for bond in mol.GetBonds():
        mol_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])
        mol_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])

    if len(mol_index) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0

    mol_index = np.array(sorted(mol_index))
    mol_edge_index = mol_index[:,:2]
    mol_rel_index = mol_index[:,2]

    ##在这个位置应该计算的是最短路径
    s_edge_index_value = calculate_shortest_path(mol_edge_index)
    s_edge_index = s_edge_index_value[:, :2]
    s_value = s_edge_index_value[:, 2]
    s_rel = s_value
    s_rel[np.where(s_value == 1)] = mol_rel_index  ##将直接相连的关
    s_rel[np.where(s_value != 1)] += 23

    assert len(s_edge_index) == len(s_value)
    assert len(s_edge_index) == len(s_rel)

    ##c_size:原子的个数
    ##features:每个原子的特征 c_size * 67
    ##edge_index:边 n_edges * 2
    return c_size, features, mol_edge_index.tolist(), mol_rel_index.tolist(), s_edge_index.tolist(), s_value.tolist(), s_rel.tolist(), max(degrees)

def calculate_shortest_path(edge_index):

    s_edge_index_value = []

    g = nx.DiGraph()
    g.add_edges_from(edge_index.tolist())

    paths = nx.all_pairs_shortest_path_length(g)
    for node_i, node_ij in paths:
        for node_j, length_ij in node_ij.items():
            s_edge_index_value.append([node_i, node_j, length_ij])

    s_edge_index_value.sort()

    return np.array(s_edge_index_value)


def read_interactions(path, drug_dict):
    interactions = []
    all_drug_in_ddi = []
    positive_drug_inter_dict = {}
    positive_num = 0
    negative_num = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            drug1_id, drug2_id, rel, label = line.strip().split(" ")[:4]
            if drug1_id in drug_dict and drug2_id in drug_dict:
                all_drug_in_ddi.append(drug1_id)
                all_drug_in_ddi.append(drug2_id)
                if float(label) > 0:
                    positive_num += 1
                else:
                    negative_num += 1
                if drug1_id in positive_drug_inter_dict:
                    if drug2_id not in positive_drug_inter_dict[drug1_id]:
                        positive_drug_inter_dict[drug1_id].append(drug2_id)
                        interactions.append([int(drug1_id), int(drug2_id), int(rel), int(label)])
                else:
                    positive_drug_inter_dict[drug1_id] = [drug2_id]
                    interactions.append([int(drug1_id), int(drug2_id), int(rel), int(label)])
        f.close()

    print(positive_num)
    print(negative_num)

    assert negative_num == positive_num

    return np.array(interactions, dtype=int), set(all_drug_in_ddi)

def read_network(path):

    edge_index = []
    rel_index = []

    flag = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            if flag == 0:
                flag = 1
                continue
            else:
                flag += 1
                head, tail, rel = line.strip().split(" ")[:3]
                edge_index.append([int(head), int(tail)])
                rel_index.append(int(rel))

        f.close()
    num_node = np.max((np.array(edge_index)))
    num_rel = max(rel_index) + 1
    print(len(list(set(rel_index))))

    return num_node, edge_index, rel_index, num_rel

def read_smiles(path):
    print("Read " + path + "!")
    flag = 0
    out = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            if flag == 0:
                flag += 1
                continue
            else:
                id, sequence = line.strip().split("\t")
                if id not in out:
                    out[id] = sequence  ##这里面的id是str类型
        f.close()

    return out

def generate_node_subgraphs(dataset, drug_id, network_edge_index, network_rel_index, num_rel, args):

    method = args.extractor
    edge_index = torch.from_numpy(np.array(network_edge_index).T) ##[2, num_edges]
    rel_index = torch.from_numpy(np.array(network_rel_index))

    row, col = edge_index
    reverse_edge_index = torch.stack((col, row),0)
    undirected_edge_index = torch.cat((edge_index, reverse_edge_index),1)

    paths = "data/" + str(dataset) + "/" + str(method) + "/"

    if not os.path.exists(paths):
        os.mkdir(paths)

    if method == "khop-subtree":
        subgraphs, max_degree, max_rel_num = subtreeExtractor(drug_id, undirected_edge_index, rel_index, paths, num_rel,
                                                              fixed_num=args.fixed_num, khop=args.khop)
    elif method == "probability":
        pagerank_paths = "data/" + str(dataset) + "/" + str(method) + "/pageRank.json"
        subgraphs, max_degree, max_rel_num = probExtractor(drug_id, undirected_edge_index, rel_index, paths, num_rel,
                                                           fixed_num=args.fixed_num, pagerank_path = pagerank_paths)
    elif method == "randomWalk":
        subgraphs, max_degree, max_rel_num = rwExtractor(drug_id, undirected_edge_index, rel_index, paths, num_rel,
                                                         sub_num=args.graph_fixed_num, length=args.fixed_num)

    return subgraphs, max_degree, max_rel_num

def subtreeExtractor(drug_id, edge_index, rel_index, shortest_paths, num_rel, fixed_num, khop):

    all_degree = []
    num_rel_update = []
    subgraphs = {}

    json_path = shortest_paths + "subtree_fixed_" + str(fixed_num) + "_hop_" + str(khop) + "sp.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            subgraphs = json.load(f)
            max_rel = 0
            max_degree = 0
            for s in subgraphs.keys():
                max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
                max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree

        return subgraphs, max_degree, max_rel;

    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    for d in drug_id:
        subset, sub_edge_index, sub_rel_index, mapping_list = k_hop_subgraph(int(d), khop, edge_index, undirected_rel_index, fixed_num, relabel_nodes=True)  ##subset是所有集合的节点，mapping指示的是center node是哪个
        row, col = sub_edge_index
        all_degree.append(torch.max(degree(col)).item())

        ##因为这里面会涉及到multi-relation，所以在添加子图的时候，要把多条边都添加进去
        new_s_edge_index = sub_edge_index.transpose(1,0).numpy().tolist()
        new_s_value = [1 for _ in range(len(new_s_edge_index))]
        new_s_rel = sub_rel_index.numpy().tolist()
        node_idx = subset.numpy().tolist()

        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:  ##也是保证多关系的边全部在数据里
                continue
            else:
                s_edge_index.append(sp_edge_index[i].tolist())
                s_value.append(sp_value[i])
                s_rel.append(sp_value[i] + num_rel)

        assert len(s_edge_index) == len(s_value)
        assert len(s_edge_index) == len(s_rel)

        num_rel_update.append(np.max(s_rel))

        subgraphs[d] = node_idx, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel, torch.max(degree(col)).item()

    with open(json_path, 'w') as f:
        json.dump(subgraphs, f, default=convert)

    ## subset: LongTensor
    ## edge_index: LongTensor
    ## subgraph_rel: Tensor
    return subgraphs, max(all_degree), max(num_rel_update)

def probExtractor(drug_id, edge_index, rel_index, shortest_paths, num_rel, fixed_num, pagerank_path):

    json_path = shortest_paths + "prob_fix_" + str(fixed_num) + "_sp.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            subgraphs = json.load(f)
            max_rel = 0
            max_degree = 0
            for s in subgraphs.keys():
                max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
                max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree

        return subgraphs, max_degree, max_rel;

    g = nx.DiGraph()
    g.add_edges_from(edge_index.transpose(1, 0).tolist())

    if not os.path.exists(pagerank_path):
        pagerank = np.array(google_matrix(g), dtype='float32')
        page_dict = {}
        for d in drug_id:
            page_dict[d] = list(pagerank[list(g.nodes()).index(int(d))])
        with open(pagerank_path, 'w') as f:
            json.dump(page_dict, f)
    else:
        with open(pagerank_path, 'r') as f:
            page_dict = json.load(f)
        f.close()


    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    num_rel_update = []
    max_degree = []
    subgraphs = {}
    for d in drug_id:
        subsets = [int(d)]

        neighbors = np.random.choice(
            a=list(g.nodes()),
            size=fixed_num,
            replace=False,
            p=page_dict[d])

        subsets.extend(neighbors)
        subsets = list(set(subsets))

        print(subsets)

        mapping_list = [False for _ in subsets]
        mapping_idx = subsets.index(int(d))
        mapping_list[mapping_idx] = True

        sub_edge_index, sub_rel_index =  subgraph(subsets, edge_index, undirected_rel_index, relabel_nodes=True)
        row_sub, col_sub = sub_edge_index
        ##因为这里面会涉及到multi-relation，所以在添加子图的时候，要把多条边都添加进去
        new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
        new_s_value = [1 for _ in range(len(new_s_edge_index))]
        new_s_rel = sub_rel_index.numpy().tolist()


        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:  ##也是保证多关系的边全部在数据里
                continue
            else:
                s_edge_index.append(sp_edge_index[i].tolist())
                s_value.append(sp_value[i])
                s_rel.append(sp_value[i] + num_rel)

        assert len(s_edge_index) == len(s_value)
        assert len(s_edge_index) == len(s_rel)

        num_rel_update.append(int(np.max(s_rel)))
        max_degree.append(torch.max(degree(col_sub)).item())

        subgraphs[d] = subsets, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel, torch.max(degree(col_sub)).item()

    with open(json_path, 'w') as f:
        json.dump(subgraphs, f, default=convert)

    return subgraphs, max(max_degree), max(num_rel_update)

def rwExtractor(drug_id, edge_index, rel_index, shortest_paths, num_rel, sub_num, length):

    json_path = shortest_paths + "rw_num_" + str(sub_num) + "_length_" + str(length) + "sp.json"
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            subgraphs = json.load(f)
            max_rel = 0
            max_degree = 0
            for s in subgraphs.keys():
                max_rel = max(subgraphs[s][6]) if max(subgraphs[s][6]) > max_rel else max_rel
                max_degree = subgraphs[s][7] if subgraphs[s][7] > max_degree else max_degree
        return subgraphs, max_degree, max_rel;

    my_graph = nx.Graph()
    my_graph.add_edges_from(edge_index.transpose(1,0).numpy().tolist())
    undirected_rel_index = torch.cat((rel_index, rel_index), 0)

    num_rel_update = []
    max_degree = []
    subgraphs = {}
    for d in drug_id:
        subsets = Node2vec(start_nodes=[int(d)], graph=my_graph, path_length=length, num_paths=sub_num, workers=6, dw=True).get_walks() ##返回一个list
        mapping_id = subsets.index(int(d))
        mapping_list = [False for _ in range(len((subsets)))]
        mapping_list[mapping_id] = True

        sub_edge_index, sub_rel_index = subgraph(subsets, edge_index, undirected_rel_index, relabel_nodes=True)
        row_sub, col_sub = sub_edge_index
        ##因为这里面会涉及到multi-relation，所以在添加子图的时候，要把多条边都添加进去
        new_s_edge_index = sub_edge_index.transpose(1, 0).numpy().tolist()
        new_s_value = [1 for _ in range(len(new_s_edge_index))]
        new_s_rel = sub_rel_index.numpy().tolist()

        s_edge_index = new_s_edge_index.copy()
        s_value = new_s_value.copy()
        s_rel = new_s_rel.copy()

        edge_index_value = calculate_shortest_path(sub_edge_index.transpose(1, 0).numpy())
        sp_edge_index = edge_index_value[:, :2]
        sp_value = edge_index_value[:, 2]

        for i in range(len(sp_edge_index)):
            if sp_value[i] == 1:  ##也是保证多关系的边全部在数据里
                continue
            else:
                s_edge_index.append(sp_edge_index[i].tolist())
                s_value.append(sp_value[i])
                s_rel.append(sp_value[i] + num_rel)

        assert len(s_edge_index) == len(s_value)
        assert len(s_edge_index) == len(s_rel)

        num_rel_update.append(int(np.max(s_rel)))
        max_degree.append(torch.max(degree(col_sub)).item())

        subgraphs[d] = subsets, new_s_edge_index, new_s_rel, mapping_list, s_edge_index, s_value, s_rel, torch.max(degree(col_sub)).item()

    with open(json_path, 'w') as f:
        json.dump(subgraphs, f, default=convert)

    return subgraphs, max(max_degree), max(num_rel_update)


def k_hop_subgraph(node_idx, num_hops, edge_index, rel_index, fixed_num, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):

    np.random.seed(42)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        #print(col[edge_mask].shape)
        if fixed_num == None:
            subsets.append(col[edge_mask])
        elif col[edge_mask].size(0) > fixed_num:
            neighbors = np.random.choice(a=col[edge_mask].numpy(), size=fixed_num, replace=False)
            subsets.append(torch.LongTensor(neighbors))
        else:
            subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]
    #print(subset)

    rel_index = rel_index[edge_mask] if rel_index is not None else None


    mapping_mask = [False for _ in range(len(subset))]
    mapping_mask[inv] = True


    return subset, edge_index, rel_index, mapping_mask


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def convert(o):
    if isinstance(o, np.int64): return int(o)
    raise TypeError

def min_max(data:list):
    min_value = min(data)
    max_value = max(data)

    norm_data = []
    for d in data:
        norm_data.append((d-min_value+0.00001)/(max_value-min_value))

    return [d/sum(norm_data) for d in norm_data]


def google_matrix(
    G, alpha=0.85, personalization=None, nodelist=None, weight="weight", dangling=None
):
    import numpy as np

    if nodelist is None:
        nodelist = list(G)

    M = np.asmatrix(nx.to_numpy_array(G, nodelist=nodelist, weight=weight), dtype='float32')
    N = len(G)
    if N == 0:
        return M

    # Personalization vector
    if personalization is None:
        p = np.repeat(1.0 / N, N).astype('float32')
    else:
        p = np.array([personalization.get(n, 0) for n in nodelist], dtype="float32")
        if p.sum() == 0:
            raise ZeroDivisionError
        p /= p.sum()


    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = np.array([dangling.get(n, 0) for n in nodelist], dtype='float32')
        dangling_weights /= dangling_weights.sum()
    dangling_nodes = np.where(M.sum(axis=1) == 0)[0]

    # Assign dangling_weights to any dangling nodes (nodes with no out links)
    for node in dangling_nodes:
        M[node] = dangling_weights

    M /= M.sum(axis=1).astype('float32')  # Normalize rows to sum to 1

    return np.multiply(alpha, M, dtype='float32') + np.multiply(1 - alpha, p, dtype='float32')