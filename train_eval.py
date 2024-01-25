import time
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score, auc
from tqdm import *

# training function at each epoch

def train(loop, model, optimizer):
    correct, total_loss = 0, 0
    model.train()

    prob_all = []
    label_all = []

    for data in loop:
        if torch.cuda.is_available():
            data_mol1 = data[0].cuda()
            data_drug1 = data[1].cuda()
            data_mol2 = data[2].cuda()
            data_drug2 = data[3].cuda()
        else:
            data_mol1 = data[0].cpu()
            data_drug1 = data[1].cpu()
            data_mol2 = data[2].cpu()
            data_drug2 = data[3].cpu()

        optimizer.zero_grad(set_to_none=True)

        predicts, loss = model(data_mol1, data_drug1, data_mol2, data_drug2)
        loss.backward()

        prob_all.append(predicts)
        label_all.append(data_mol1.y)

        total_loss += loss.item() * num_graphs(data_mol1)

        optimizer.step()

    train_loss = total_loss / len(loop)
    label_all = torch.concat(label_all).cpu().detach().numpy()
    prob_all = torch.concat(prob_all).cpu().detach().numpy()
    train_acc, train_f1, train_auc, train_aupr = get_score(label_all, prob_all)

    return train_acc, train_f1, train_auc, train_aupr, train_loss

def eval(loader, model):
    correct, total_loss = 0, 0
    model.eval()

    prob_all = []
    label_all = []

    with torch.no_grad():
        for idx, data in enumerate(loader):
            if torch.cuda.is_available():
                data_mol1 = data[0].cuda()
                data_drug1 = data[1].cuda()
                data_mol2 = data[2].cuda()
                data_drug2 = data[3].cuda()
            else:
                data_mol1 = data[0].cpu()
                data_drug1 = data[1].cpu()
                data_mol2 = data[2].cpu()
                data_drug2 = data[3].cpu()


            predicts, loss = model(data_mol1, data_drug1, data_mol2, data_drug2)

            ##获取指标
            prob_all.append(predicts)
            label_all.append(data_mol1.y)
            total_loss += loss.item() * num_graphs(data_mol1)

    eval_loss = total_loss / len(loader.dataset)
    label_all = torch.concat(label_all).cpu().detach().numpy()
    prob_all = torch.concat(prob_all).cpu().detach().numpy()
    eval_acc, eval_f1, eval_auc, eval_aupr = get_score(label_all, prob_all)

    return eval_acc, eval_f1, eval_auc, eval_aupr, eval_loss

def test(loader, model):
    correct, total_loss = 0, 0
    model.eval()

    prob_all = []
    label_all = []

    with torch.no_grad():
        for idx, data in enumerate(loader):
            if torch.cuda.is_available():
                data_mol1 = data[0].cuda()
                data_drug1 = data[1].cuda()
                data_mol2 = data[2].cuda()
                data_drug2 = data[3].cuda()
            else:
                data_mol1 = data[0].cpu()
                data_drug1 = data[1].cpu()
                data_mol2 = data[2].cpu()
                data_drug2 = data[3].cpu()

            predicts, loss = model(data_mol1, data_drug1, data_mol2, data_drug2)

            ##获取指标
            prob_all.append(predicts)
            label_all.append(data_mol1.y)
            total_loss += loss.item() * num_graphs(data_mol1)


    test_loss = total_loss / len(loader.dataset)
    label_all = torch.concat(label_all).cpu().detach().numpy()
    prob_all = torch.concat(prob_all).cpu().detach().numpy()
    test_acc, test_f1, test_auc, test_aupr = get_score(label_all, prob_all)

    return {"acc":test_acc,
            "f1":test_f1,
            "auc":test_auc,
            "aupr":test_aupr,
            "loss":test_loss}

def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.c_size

def get_score(label_all, prob_all):

    predicts_label = [1 if prob >= 0.5 else 0 for prob in prob_all]

    acc = accuracy_score(label_all, predicts_label)
    f1 = f1_score(label_all, predicts_label)
    auroc = roc_auc_score(label_all, prob_all)
    p, r, t = precision_recall_curve(label_all, prob_all)
    auprc = auc(r, p)

    return acc, f1, auroc, auprc
