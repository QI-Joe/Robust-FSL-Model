import random
import torch
import sys
sys.path.append("/home/24052653g/Robust-FSL-Model/")
from utils.my_dataloader import Temporal_Splitting, Dynamic_Dataloader, data_load, to_cuda
from utils.time_evaluation import TimeRecord
from GCA_node import Encoder, GRACE, GCA_config
from GCA_utils import get_base_model, GCA_Augmentation
from utils.evaluate_nodeclassification import eval_GCA
from GCA_functional import *
from torch_geometric.utils import dropout_adj
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import copy
import argparse
print(sys.path)

def train_GCA(model: GRACE, optimizer, param, data: Data, feature_weights, drop_weights, args):
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])
    
    if(args.graph_size == 0):
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2, batch_size=1024 if args.dataset == 'Coauthor-Phy' else None)
        loss.backward()
        optimizer.step()
        return loss.item(), z1.shape[0]
    else:
        total_loss = 0
        batch_size = 2000
        dz1, dz2 = Data(x = x_1, edge_index = edge_index_1), Data(x = x_2, edge_index = edge_index_2)
        neighbor1 = NeighborLoader(dz1, batch_size=batch_size, num_neighbors=[-1], shuffle=False)
        for idx, batch in enumerate(neighbor1):
            inter_batch_size = batch.batch_size
            seed_node = batch.n_id[:inter_batch_size]
            neighbor2 = NeighborLoader(dz2, batch_size=inter_batch_size, num_neighbors=[-1], input_nodes=seed_node, shuffle=False)
            batch2 = next(iter(neighbor2))
            assert batch.n_id[:inter_batch_size].tolist() == batch2.n_id[:inter_batch_size].tolist(), "sorry, this method seems a little problem."
            z1 = model(batch.x, batch.edge_index)[: inter_batch_size]
            z2 = model(batch2.x, batch2.edge_index)[:inter_batch_size]
            loss = model.loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss/idx, batch_size
        # get neighborloader for z1 and z2, and constrcut the exact loss function


def main_GCA(outside_args, default_param, time_: TimeRecord):
    args = outside_args
    time_.get_dataset(args.dataset)
    time_.set_up_logger(name="time_logger")
    time_.set_up_logger()
    time_.record_start()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in default_param.keys():
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    non_split = True
    random.seed(2024)
    torch.manual_seed(2024)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graph, idxloader = data_load(args.dataset)
    snapshot = 3
    num_classes = graph.y.max().item() + 1
    # graph_list = Temporal_Splitting(graph).temporal_splitting(time_mode="view", \
    #                 snapshot=snapshot, views=snapshot-2, strategy="sequential", non_split=non_split)
    # temporaLoader = Dynamic_Dataloader(graph_list, graph=graph)

    encoder = Encoder(graph.x.size(1), param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau']).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    datawarehouse: list[tuple] = []

    time_.temporal_record()
    data = copy.deepcopy(graph)
    data = to_cuda(data)
    
    drop_weights, feature_weights = GCA_Augmentation(data=data, param=param, args=args, device=device)
    # feature_weights somehow will return Nan value, currently will detect and switch it to 0
    # also, it will be print out
    if torch.isnan(feature_weights).any():
        val, freq = torch.unique(feature_weights, return_counts=True)
        val, freq = number_calculate(val, freq)
        print("#"+"-"*30+\
                "\n"+\
            f'Feature weights has NaN value, will be replaced by 0')
        print(f"torch unique shows: {val}, {freq}")
        print("#"+"-"*30)
        feature_weights[torch.isnan(feature_weights)] = 0

    log = args.verbose.split(',')

    microList, epoch_time = [], []
    spliter = generate_split(data.num_nodes, train_ratio=0.8, val_ratio=0.2)
    for epoch in range(1, args.num_epoches + 1):
        time_.epoch_record()

        loss, batch_size = train_GCA(model, optimizer, param, data, feature_weights, drop_weights, args)
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, node Num={data.num_nodes}')
        
        time_.epoch_end(batch_size)
        if (epoch+1) % 200 == 0:
            all_data = data
            micro = eval_GCA(model=model, data=all_data, device=device, split=spliter, num_classes=num_classes)

            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {micro["test_acc"]:03f}')
            microList.append(micro)
    train_acc, val_acc, test_acc, accuracy, precision, recall, f1, \
        micro_prec, micro_recall, micro_f1\
            = zip(*[list(data.values()) for data in microList])
    time_.score_record(microList, data.num_nodes, 1)
    # train_acc, val_acc .etc. shape:
    # based on how many times test, if every 50 epoch test once, then it will be epoch/50 length
    # thus, last acc should be focused since it will be the highest one.

    final_micro = eval_GCA(model=model, data=all_data, device=device, split=spliter, num_classes=num_classes)
    datawarehouse.append([final_micro["test_acc"], train_acc, val_acc, test_acc, accuracy, precision, recall, f1])
    if 'final' in log:
        print(f'{final_micro}')
    
    time_.temporal_end(data.num_nodes)

    time_.record_end()
    time_.to_log()
    for i in range(len(datawarehouse)):
        final_micro, train_acc, val_acc, test_acc, accuracy, precision, recall, f1 = datawarehouse[i]
        print(f'View {i}, \n \
            Final Test Acc {final_micro:04f}, \n \
            Train Acc {train_acc[-1]:03f}, \n \
            Test Acc {test_acc[-1]:03f}, \n \
            Val Acc {val_acc[-1]:03f} \n\n \
            Avg accuracy {sum(accuracy)/len(accuracy):05f}, \n \
            Avg precision {sum(precision)/len(precision):05f}, \n \
            Avg recall {sum(recall)/len(recall):05f}, \n \
            Avg f1 {sum(f1)/len(f1):05f}')
        
if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='appnp')
    parser.add_argument('--device', type=str, default='cuda:0')
    default_param = {

    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args, model_detail = parser.parse_known_args()

    print("Expected model is:", args.model.upper())
    
    args.model = args.model.lower()
    time_rec = TimeRecord(args.model)

    gca_config, default_param = GCA_config(model_detail)
    main_GCA(gca_config, default_param, time_rec)