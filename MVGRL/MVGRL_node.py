import torch

from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import random
import GCL.augmentors as A # pip install PyGCL
# from utils.my_dataloader import to_cuda 


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class MVGEncoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim):
        super(MVGEncoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)
        self.switch = False

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def switch_mode(self, in_eval: bool):
        self.switch = in_eval
        return self.switch

    def forward(self, x, edge_index, edge_weight=None):
        aug1, aug2 = self.augmentor
        batch_size = x.size(0)
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)

        if self.switch:
            aug2 = A.PPRDiffusion(alpha=0.2)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)

        z1 = self.encoder1(x1, edge_index1, edge_weight1)
        z2 = self.encoder2(x2, edge_index2, edge_weight2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1(*self.corruption(x1, edge_index1, edge_weight1))
        z2n = self.encoder2(*self.corruption(x2, edge_index2, edge_weight2))
        return z1, z2, g1, g2, z1n, z2n, torch.arange(z1.size(0)), batch_size

class Encoder_Neighborlaoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim):
        super(Encoder_Neighborlaoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)
        self.switch = False

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def switch_mode(self, in_eval: bool):
        self.switch = in_eval
        return self.switch

    def forward(self, x, edge_index, edge_weight=None):
        # x = x.cpu()
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        if self.switch:
            aug2 = A.PPRDiffusion(alpha=0.2)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        # perhaps we can treat edge_weight as an edge_attr when building Data() object

        torch.manual_seed(2024)
        random.seed(2024)

        batch_size = 2000
        n_id_recorder = torch.LongTensor([])
        dx1, dx2 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_weight1), Data(x=x2, edge_index=edge_index2, edge_attr=edge_weight2)
        neighbor1 = NeighborLoader(data=dx1, batch_size=batch_size, num_neighbors=[-1], shuffle=False)
        device = dx1.x.device
        z1, z2, g1, g2, z1n, z2n = torch.FloatTensor([]).to(device), torch.FloatTensor([]).to(device), torch.FloatTensor([]).to(device), \
            torch.FloatTensor([]).to(device), torch.FloatTensor([]).to(device), torch.FloatTensor([]).to(device)
        for batch in neighbor1:
            inter_batch1_size = batch.batch_size
            seed_node = batch.n_id[:inter_batch1_size]
            n_id_recorder = torch.cat((n_id_recorder, seed_node), dim=0)
            neighbor2 = NeighborLoader(data=dx2, batch_size=batch_size, num_neighbors=[-1], input_nodes=seed_node, shuffle=False)
            batch2 = next(iter(neighbor2))
            inter_batch2_size = batch2.batch_size

            z1a = self.encoder1(batch.x, batch.edge_index, batch.edge_attr)[:inter_batch1_size]
            z2a = self.encoder2(batch2.x, batch2.edge_index, batch2.edge_attr)[:inter_batch2_size]
            z1 = torch.cat((z1, z1a), dim=0)
            z2 = torch.cat((z2, z2a), dim=0)

            g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
            g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))

            z1n = torch.cat((z1n, self.encoder1(*self.corruption(batch.x, batch.edge_index, batch.edge_attr))[:inter_batch1_size]), dim=0)
            z2n = torch.cat((z2n, self.encoder2(*self.corruption(batch2.x, batch2.edge_index, batch2.edge_attr))[:inter_batch2_size]), dim=0)

        return z1, z2, g1, g2, z1n, z2n, n_id_recorder, batch_size


