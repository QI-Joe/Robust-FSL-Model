from doctest import FAIL_FAST
from numpy import around
from sympy import Union
from typing import Union as union2
import torch
import torch.nn as nn
import random
from torch_geometric.datasets import Planetoid
from typing import Optional
import torch.nn.functional as F
from MVGRL.MVGRL_node import MVGEncoder

import torch as th
from torch.optim import Adam
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, roc_auc_score, recall_score, f1_score
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from typing import *


from utils.my_dataloader import Temporal_Dataloader
"""
OK, so this is evaluation file for node classification task, which include model of GCA, MVGRL and CLDG
Class may not involved, in first stage only MVGRL evaluated method will be invovled. 
structure may refer to TGB_baseline: https://github.com/fpour/TGB_Baselines/tree/main/evaluation
"""

def get_split(num_samples: int, emb:Union[torch.Tensor | tuple[torch.Tensor]], data: Union[Data | tuple[Data, Data]], \
              Nontemproal: bool=False, \
              train_ratio: float = 0.1, test_ratio: float = 0.8):
    random.seed(2024)
    torch.manual_seed(2024)
    if Nontemproal:
        assert train_ratio + test_ratio < 1
        train_size = int(num_samples * train_ratio)
        val_size = int(num_samples * train_ratio* 2)
        indices = torch.randperm(num_samples)

        assert not isinstance(data, list), f"expected to be static graph, but got dynamic-fit graph with {len(data)} graphs"
        assert not isinstance(emb, list), f"expected to be static graph, but got dynamic-fit graph with {len(emb)} embeddings"

        graph, emb = data, emb
        train = {
            "emb": emb[indices[:train_size]],
            "label": graph.y[indices[:train_size]],
        }
        valid = {
            "emb": emb[indices[train_size: val_size]],
            "label": graph.y[indices[train_size: val_size]],
        }
        test = {
            "emb": emb[indices[val_size:]],
            "label": graph.y[indices[val_size:]],
        }
    else:
        train_size = int(num_samples * train_ratio)
        indices = torch.randperm(num_samples)

        graph1, graph2 = data
        emb1, emb2 = emb
        train = {
            "emb": emb1[indices[:train_size]],
            "label": graph1.y[indices[:train_size]],
        }
        valid = {
            "emb": emb1[indices[train_size: ]],
            "label": graph1.y[indices[train_size:]],
        }
        test = {
            "emb": emb2,
            "label": graph2.y,
        }
    return (train, valid, test)


class LogRegression(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LogRegression, self).__init__()
        self.lin = torch.nn.Linear(in_channels, num_classes)
        nn.init.xavier_uniform_(self.lin.weight.data)
        # torch.nn.init.xavier_uniform_(self.lin.weight.data)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        ret = self.lin(x)
        return ret

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = nn.Linear(in_channels, in_channels)
        self.lin_dst = nn.Linear(in_channels, in_channels)
        self.lin_final = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        # h = self.lin_src(z_src) + self.lin_dst(z_dst)
        # h = h.relu()
        h = F.cosine_similarity(self.lin_src(z_src), self.lin_dst(z_dst))
        return self.lin_final(h)

def eval_MVGRL(encoder_model: MVGEncoder, data: tuple[Data, Data], device) -> dict[str, float]:
    encoder_model.eval()
    num_epoch = 1000
    if not isinstance(data, (Data, Temporal_Dataloader)):
        trian_val, test = data
        z1, z2, _, _, _, _, n_id, _ = encoder_model(trian_val.x, trian_val.edge_index)
        t1 = (z1 + z2).detach()
        encoder_model.switch_mode(in_eval=True)
        tz1, tz2, _, _, _, _, n_id, _ = encoder_model(test.x, test.edge_index)
        t2 = (tz1 + tz2).detach()

        trian_test = get_split(num_samples=t1.size()[0], emb=(t1,t2), data=data,)

    else:
        z1, z2, _, _, _, _, n_id, _ = encoder_model(data.x, data.edge_index)
        t1 = (z1 + z2).detach()
        data.y = data.y[n_id]

        trian_test = get_split(num_samples=t1.size(0), emb = t1, data = data, Nontemproal=True)

    result = log_regression(dataset=trian_test, evaluator = MulticlassEvaluator(), \
                            model_name="MVGRL", num_classes=None, device=device, num_epochs=num_epoch)
    return result

def eval_CLDG(embedding_model: LogRegression,
              embeddings: tuple[th.Tensor],  
              DATASET: str, 
              trainTnum: int,
              in_label: pd.DataFrame = None,
              testIdx: Optional[th.Tensor] = None,
              device_id="cuda:0", 
              idxloader: object=None,
              *args) -> dict[str, float]:
    
    ''' Linear Evaluation '''
    # if DATASET == "dblp":
    #     labels, train_idx, val_idx, test_labels, n_classes = CLDG_testdataloader(DATASET, testIdx, idxloader=idxloader)
    if DATASET == "mathoverflow" or DATASET == "dblp":
        train_nodes, test_nodes = testIdx
        labels, test_label = idxloader.node.label[train_nodes].values, idxloader.node.label[test_nodes].values
        labels, test_labels = torch.tensor(labels), torch.tensor(test_label).to(device_id)
        lenth = len(train_nodes)
        label_train, label_val = list(range(int(0.4*lenth))), list(range(int(0.4*lenth),lenth))
        train_idx, val_idx = train_nodes[:int(0.4*lenth)], train_nodes[int(0.4*lenth):]
    else: 
        raise NotImplementedError("This kind of dataset import is not supported....")

    train_val_emb, test_emb = embeddings

    train_embs = train_val_emb[train_idx] # .to(device_id)
    val_embs = train_val_emb[val_idx]
    test_embs = test_emb[test_labels]

    n_classes = torch.unique(labels)[-1].item()+1
    label = labels.to(device_id)

    train_labels = label[label_train].clone().detach()
    val_labels = label[label_val].clone().detach()

    train = {"emb": train_embs, "label": train_labels}
    val = {"emb": val_embs, "label": val_labels}
    test = {"emb": test_embs, "label": test_labels}

    data = (train, val, test)

    return log_regression(dataset=data, evaluator=MulticlassEvaluator(), model_name="CLDG", device = device_id, num_classes=n_classes, num_epochs=100)


def get_idx_split(dataset, split, preload_split):
    if split[:4] == 'rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = dataset[0].x.size(0)
        train_size = int(num_nodes * train_ratio)
        indices = torch.randperm(num_nodes)
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }
    elif split == 'ogb':
        return dataset.get_idx_split()
    elif split.startswith('wikics'):
        split_idx = int(split.split(':')[1])
        return {
            'train': dataset[0].train_mask[:, split_idx],
            'test': dataset[0].test_mask,
            'val': dataset[0].val_mask[:, split_idx]
        }
    elif split == 'preloaded':
        assert preload_split is not None, 'use preloaded split, but preloaded_split is None'
        train_mask, test_mask, val_mask = preload_split
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }
    else:
        raise RuntimeError(f'Unknown split type {split}')

def Simple_Regression(embedding: torch.Tensor, label: union2[torch.Tensor | np.ndarray], num_classes: int, \
                      num_epochs: int = 1500,  project_model=None, return_model: bool = False) -> tuple[float, float, float, float]:
    device = embedding.device
    if not isinstance(label, torch.Tensor):
        label = torch.LongTensor(label).to(device)
    linear_regression = LogRegression(embedding.size(1), num_classes).to(device) if project_model==None else project_model
    f = nn.LogSoftmax(dim=-1)
    optimizer = Adam(linear_regression.parameters(), lr=0.01, weight_decay=1e-4)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        linear_regression.train()
        optimizer.zero_grad()
        output = linear_regression(embedding)
        loss = loss_fn(f(output), label)

        loss.backward(retain_graph=False)
        optimizer.step()

        if (epoch+1) % 500 == 0:
            print(f'LogRegression | Epoch {epoch}: loss {loss.item():.4f}')

    with torch.no_grad():
        projection = linear_regression(embedding)
        y_true, y_hat = label.cpu().numpy(), projection.argmax(-1).cpu().numpy()
        accuracy, precision, recall, f1 = accuracy_score(y_true, y_hat), \
                                        precision_score(y_true, y_hat, average='macro', zero_division=0), \
                                        recall_score(y_true, y_hat, average='macro'),\
                                        f1_score(y_true, y_hat, average='macro')
        prec_micro, recall_micro, f1_micro = precision_score(y_true, y_hat, average='micro', zero_division=0), \
                                            recall_score(y_true, y_hat, average='micro'),\
                                            f1_score(y_true, y_hat, average='micro')
    if return_model:
        return {"test_acc": accuracy, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, \
            "micro_prec": prec_micro, "micro_recall": recall_micro, "micro_f1": f1_micro}, linear_regression
    
    return {"test_acc": accuracy, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, \
            "micro_prec": prec_micro, "micro_recall": recall_micro, "micro_f1": f1_micro}, None


def log_regression(dataset: Tuple[dict[str, torch.Tensor]], # (T(trian, valid), T+1(test))
                   evaluator: nn.modules,
                   model_name: str,
                   num_classes: int, 
                   device: object, 
                   num_epochs: int = 500,
                   verbose: bool = True,
                   preload_split=None):
    """
    z should be embedding result
    evaluator is embeeding_model/decoder
    data is Data object
    split should store the train/valid/test mask/splitation/indices, better in dict format
    """

    model_input = None
    train, val, test = dataset[0], dataset[1], dataset[2]
    model_input = train["emb"].size(1)

    f = nn.LogSoftmax(dim=-1)

    loss_fn = nn.CrossEntropyLoss()

    num_classes = test["label"].unique().size(0)
    classifier = LogRegression(model_input, num_classes)
    classifier = classifier.to(device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)

    best_train_acc = 0
    best_val_acc = 0

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()

        output = classifier(train["emb"])
        loss = loss_fn(f(output), train["label"])

        loss.backward(retain_graph=False)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            # val split is available
            val_acc = evaluator.eval({
                'y_true': val["label"].view(-1, 1),
                'y_pred': classifier(val["emb"]).argmax(-1).view(-1, 1)
            })['acc']
            train_acc = evaluator.eval({
                'y_true': train["label"].view(-1, 1),
                'y_pred': classifier(train["emb"]).argmax(-1).view(-1, 1)
            })['acc']
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc
            if verbose:
                print(f'(PE)|logreg epoch {epoch}: best test acc {best_val_acc:02f}, current acc {around(val_acc, 4)}, loss {loss.item():03f}')
    
    y_true_GPU, y_hat_GPU = test["label"].view(-1,1), classifier(test["emb"]).argmax(-1)            
    test_acc = evaluator.eval({
        'y_true': y_true_GPU,
        'y_pred': y_hat_GPU
        })['acc']
    y_true, y_hat = y_true_GPU.cpu().numpy(), y_hat_GPU.cpu().numpy()
    accuracy, precision, recall, f1 = accuracy_score(y_true, y_hat), \
                                    precision_score(y_true, y_hat, average='macro', zero_division=0), \
                                    recall_score(y_true, y_hat, average='macro'),\
                                    f1_score(y_true, y_hat, average='macro')
    micro_prec, micro_recall, micro_f1 = precision_score(y_true, y_hat, average='micro', zero_division=0), \
                                        recall_score(y_true, y_hat, average='micro'),\
                                        f1_score(y_true, y_hat, average='micro')
    return {'train_acc': best_train_acc, "val_acc": best_val_acc, "test_acc": test_acc, \
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, \
            "micro_prec": micro_prec, "micro_recall": micro_recall, "micro_f1": micro_f1}


class MulticlassEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        return (correct / total).item()

    def eval(self, res):
        return {'acc': self._eval(**res)}

def eval_GCA(model: torch.nn.Module, data: union2[Data | tuple[Data]], \
            num_classes: int, split, device) -> dict[str, float]:
    model.eval()
    if isinstance(data, (Data, Temporal_Dataloader)):
        t_data = data
        length = t_data.x.size(0)
        with torch.no_grad():
            z = model(t_data.x, t_data.edge_index)
        dataset = get_split(num_samples=length, emb = z, data = t_data, Nontemproal=True)
    else:
        t_data, t1_data = data
        length = t_data.x.size(0)
        with torch.no_grad():
            z = model(t_data.x, t_data.edge_index)
            z1 = model(t1_data.x, t1_data.edge_index)
        dataset = get_split(num_samples=length, emb = (z, z1), data = (t_data, t1_data), train_ratio=0.8)

    evaluator = MulticlassEvaluator()
    macro = log_regression(dataset=dataset, evaluator=evaluator, model_name="GCA", num_epochs=1000,\
                            num_classes=num_classes, device=device, preload_split=split)

    return macro

def eval_GAT(emb: tuple[torch.Tensor], data: tuple[Data, Data], num_classes: int, device: str, split_ratio: float = 0.4, *args) -> dict[str, float]:
    """
    representation of data here is supposed to be a intermeidate calling method of label, which
    specified be defiened as a list, or a numpy array; but torch.Tensor is not recommended since is over-captability
    """
    
    trian_emb, test_emb = emb
    train_label, test_label = data[0].y, data[1].y
    
    train_idx, val_idx = list(range(int(0.4*trian_emb.size(0)))), list(range(int(0.4*trian_emb.size(0)), trian_emb.size(0)))
    train_emb_gen, val_emb_gen = trian_emb[train_idx], trian_emb[val_idx]

    train_label, val_label = train_label[train_idx], train_label[val_idx]

    train = {"emb": train_emb_gen, "label": train_label}
    val = {"emb": val_emb_gen, "label": val_label}
    test = {"emb": test_emb, "label": test_label}

    dataset = (train, val, test)
    return log_regression(dataset=dataset, evaluator=MulticlassEvaluator(), model_name="GAT", num_classes=num_classes, device=device, num_epochs=100)


def eval_GAT_SL(emb: torch.Tensor, data: Temporal_Dataloader, num_classes: int, models:nn.Linear, \
                   is_val: bool, is_test: bool, \
                   device: str="cuda:0", split_ratio: float=0.4):
    """
    in SL trianing that the validation and text is sperated not doing it together, thus the same learning MLP should be used
    data: needed due to we need correct label
    """
    if is_val and not is_test:
        emb = emb[data.val_mask].detach()
        truth = data.y[data.val_mask].detach()
        return Simple_Regression(emb, truth, num_classes=num_classes, project_model=models, return_model=True)
    elif is_test and not is_val:
        emb = emb[data.test_mask].detach()
        truth = data.y[data.test_mask].detach()
        return Simple_Regression(emb, truth, num_classes=num_classes, project_model=models, return_model=False)
    raise ValueError(f"is_val, is_test should not be the same. is_val: {is_val}, is_test: {is_test}")

def eval_Graphsage_SL(emb: torch.Tensor, labels: np.ndarray, num_classes: int, models:nn.Linear, \
                   is_val: bool, is_test: bool, \
                   device: str="cuda:0", split_ratio: float=0.1):
    labels = torch.LongTensor(labels)
    if is_val and not is_test:
        train_acc = accuracy_score(labels, emb.argmax(-1))
        print(f"Train Accuracy: {train_acc:.4f}")
        return Simple_Regression(emb, labels, num_classes=num_classes, project_model=models, return_model=True)
    elif is_test and not is_val:
        return Simple_Regression(emb, labels, num_classes=num_classes, project_model=models, return_model=False)
    raise ValueError(f"is_val, is_test should not be the same. is_val: {is_val}, is_test: {is_test}")

def eval_Roland_CL(embs: tuple[torch.Tensor], data: tuple[Data], num_classes: int, device: str, split_ratio: float = 0.4, *args) -> dict[str, float]:
    trian_emb, test_emb = embs
    train_label, test_label = data[0].y, data[1].y
    
    train_node, test_node = data[0].idx2node.index_Temporal.values, data[1].idx2node.index_Temporal.values
    train_idx, val_idx = list(range(int(0.4*len(train_node)))), list(range(int(0.4*len(train_node)), len(train_node)))
    train_emb_gen, val_emb_gen = trian_emb[train_node[train_idx]], trian_emb[train_node[val_idx]]

    train_label, val_label = train_label[train_idx], train_label[val_idx]

    train = {"emb": train_emb_gen, "label": train_label}
    val = {"emb": val_emb_gen, "label": val_label}
    test = {"emb": test_emb[test_node], "label": test_label}

    dataset = (train, val, test)
    return log_regression(dataset=dataset, evaluator=MulticlassEvaluator(), model_name="RoLAND", num_classes=num_classes, device=device, num_epochs=100)

def eval_Roland_SL(emb: torch.Tensor, data: Temporal_Dataloader, num_classes: int, models:nn.Linear, \
                   is_val: bool, is_test: bool, \
                   device: str="cuda:0", split_ratio: float=0.1):
    """
    in SL trianing that the validation and text is sperated not doing it together, thus the same learning MLP should be used
    data: needed due to we need correct label
    """
    if is_val and not is_test:
        emb = emb[data.val_mask].detach()
        truth = data.y[data.val_mask].detach()
        return Simple_Regression(emb, truth, num_classes=num_classes, project_model=models, return_model=True, num_epochs=2000)
    elif is_test and not is_val:
        ground_node_mask = data.layer2_n_id.index_Temporal.values
        test_indices = ground_node_mask[data.test_mask]
        emb = emb[test_indices].detach()
        truth = data.y[test_indices].detach()
        return Simple_Regression(emb, truth, num_classes=num_classes, project_model=models, return_model=False, num_epochs=2000)
    raise ValueError(f"is_val, is_test should not be the same. is_val: {is_val}, is_test: {is_test}")

def eval_APPNP_st(emb: torch.Tensor, data: Temporal_Dataloader, num_classes: int, models:nn.Linear, \
                   is_val: bool, is_test: bool, \
                   device: str="cuda:0", split_ratio: float=0.4):
    if is_val and not is_test:
        # emb = emb.detach()
        # a, pred = emb[data.val_mask].max(dim=1)
        # correct = pred.eq(data.y[data.val_mask]).sum().item()
        # acc = correct / pred.size(0)
        # print(f"Validation Accuracy: {acc:.4f}")
        emb = emb[data.val_mask].detach()
        truth = data.y[data.val_mask].detach()
        return Simple_Regression(emb, truth, num_classes=num_classes, project_model=models, return_model=True)

def eval_model_Dy(emb: torch.Tensor, data: Temporal_Dataloader, num_classes: int, device: str="cuda:0"):
    t1_nodes = torch.tensor(data.my_n_id.node.node.values).to(device)
    emb = emb.detach()[t1_nodes]
    truth = data.y.detach()
    return Simple_Regression(emb, truth, num_classes=num_classes, project_model=None, return_model=False)

def eval_GCONV(emb: torch.Tensor, data: Temporal_Dataloader, num_classes: int, models: union2[nn.Linear | None], \
                   is_val: bool, is_test: bool, \
                   device: str="cuda:0", split_ratio: float=0.4):
    if models == None:
        models = nn.Linear(emb.size(1), num_classes).to("cuda:0")
    if models != None:
        models = models.to("cuda:0")
    if is_val and not is_test:
        emb = emb[data.val_mask].detach()
        truth = data.y[data.val_mask].detach()
    elif is_test and not is_val:
        emb = emb[data.test_mask].detach()
        truth = data.y[data.test_mask].detach()
    return Simple_Regression(emb, truth, num_classes=num_classes, project_model=models, return_model=True)

def pre_eval_Roland(model, test_data, device):
    model.eval()
    test_data = test_data.to(device)

    h, _ = model(test_data.x, test_data.edge_index, test_data.edge_label_index)
    pred_cont = torch.sigmoid(h).cpu().detach().numpy()
    label = test_data.edge_label.cpu().detach().numpy()
    avgpr_score = average_precision_score(label, pred_cont)
    
    return avgpr_score
