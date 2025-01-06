import argparse
import torch
import random
import torch
from utils.my_dataloader import Temporal_Splitting, Dynamic_Dataloader, data_load, to_cuda, Temporal_Dataloader
from utils.time_evaluation import TimeRecord
from GCL.models import DualBranchContrast
import GCL.losses as L
import GCL.augmentors as A
from utils.evaluate_nodeclassification import eval_MVGRL


from MVGRL_node import MVGEncoder, Encoder_Neighborlaoder, GConv
import torch.nn as nn
import torch.optim as optim
import torch_geometric.transforms as T
from torch.optim import Adam

def train_MVGRL(encoder_model: nn.modules, contrast_model, data: Temporal_Dataloader, optimizer: Adam):
    encoder_model.train()
    optimizer.zero_grad()
    batch_size = data.x.shape[0]
    z1, z2, g1, g2, z1n, z2n, _, batch_size = encoder_model(data.x, data.edge_index)
    loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item(), batch_size

def main_MVGRL(unknow_parms, time_: TimeRecord):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--num_epoches', type=int, default=400)
    parser.add_argument('--snapshots', type=int, default=3)
    parser.add_argument('--extra_abondan', type=int, default=0)
    parser.add_argument("--graph_size", type=int, default=0)
    args = parser.parse_args(unknow_parms)

    time_.get_dataset(args.dataset)
    time_.set_up_logger(name="time_logger")
    time_.set_up_logger()
    time_.record_start()

    snapshot = args.snapshots

    running_graph = 1

    graph, idxloader = data_load(args.dataset)
    graph_list = Temporal_Splitting(graph, snapshot).temporal_splitting()
    dataneighbor = Dynamic_Dataloader(graph_list, graph=graph)


    aug1 = A.Identity()
    gconv1 = GConv(input_dim=graph.pos.size(1), hidden_dim=512, num_layers=2).to(device)
    gconv2 = GConv(input_dim=graph.pos.size(1), hidden_dim=512, num_layers=2).to(device)
    contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device)


    datacollector: list[list] = []

    for t in range(running_graph):
        time_.temporal_record()
        sum_loss: list = []
        batch = dataneighbor.get_temporal()
        aug2 = A.PPRDiffusion(alpha=0.2)
        batch = to_cuda(batch)

        # batch, _ = data_load(args.dataset, True)
        # batch.to(device=device)
        if args.graph_size == 0:
            encoder_model = MVGEncoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=512).to(device)
        else: 
            encoder_model = Encoder_Neighborlaoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=512).to(device)
        optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.001)

        for epoch in range(1, args.num_epoches): # 2000 / 1500+
            time_.epoch_record()
            loss, bc_size = train_MVGRL(encoder_model, contrast_model, batch, optimizer)
            sum_loss.append(loss)
            print(f'(T): Epoch={epoch:03d}, loss={loss:.4f}, node Num={batch.x.shape[0]}')

            time_.epoch_end(bc_size)
            if (epoch+1) % 50 == 0:
                test_result = eval_MVGRL(encoder_model, batch, device=device)

                train_acc, val_acc, test_acc = test_result["train_acc"], test_result["val_acc"], test_result["test_acc"]
                print(f'(E): Epoch={epoch:03d}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}')
                datacollector.append(test_result)

        dataset = (batch, to_cuda(dataneighbor.get_T1graph(t)))
        test_result = eval_MVGRL(encoder_model, batch, device=device)

        test_result["min_loss"] = min(sum_loss)
        time_.score_record(datacollector, batch.x.shape[0], t)

        time_.temporal_end(batch.x.shape[0])
        print(f'(E): Best test F1Mi={test_result["test_acc"]:.4f}, F1Ma={test_result["precision"]:.4f}')
        dataneighbor.update_event(t)
    
    time_.record_end()
    time_.to_log()

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='mvgrl')
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


    main_MVGRL(model_detail, time_rec)