from torch_geometric.nn import AttentiveFP, global_mean_pool, GAT
from torch_geometric.nn.aggr import SetTransformerAggregation
import torch.nn as nn
from egnn_clean.egnn_clean import EGNN

class AttentiveFPGraphRegressor(nn.Module):
    def __init__(self, node_feat_dim=3, edge_feat_dim=3, hidden_dim=128, out_dim=174,num_layers=3, num_timesteps=2):
        super().__init__()
        self.gnn = AttentiveFP(
            in_channels=node_feat_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=edge_feat_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=0.2
        )
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        # Get node embeddings
        x = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)

        # Map to graph-level output
        out = self.lin(x)# [batch_size x out_dim]
        return out

class BaselineGAT(nn.Module):
    def __init__(self, node_feat_dim=3, edge_feat_dim=3, hidden_dim=128, out_dim=174,num_layers=3,dropout=0.):
        super().__init__()

        self.gnn = GAT(
            in_channels=node_feat_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            v2=True,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )

        self.readout = SetTransformerAggregation(channels=hidden_dim, heads=8)

        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = self.gnn(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        x_read = self.readout(x,index=data.batch)
        out = self.lin(x_read)
        return out


class EGNN_predictor(nn.Module):

    def __init__(self, node_feat_dim=3, edge_feat_dim=3, hidden_dim=128, out_dim=174,num_layers=3):
        self.gnn = EGNN(
            in_node_nf=node_feat_dim,
            in_edge_nf=edge_feat_dim,
            hidden_dim=hidden_dim,
            n_layers=num_layers,
        )

        self.readout = SetTransformerAggregation(channels=hidden_dim, heads=8)

        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, _ = self.gnn(h=data.x, x=data.pos, edges=data.edge_index, edge_attr=data.edge_attr)
        x = self.gnn(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        x_read = self.readout(x,index=data.batch)
        out = self.lin(x_read)
        return out


class Hierachical_GAT(nn.Module):
    pass
    # forward : 1) get node embedding trougth GAT
    # 2) perform edges prediction (only for aa-aa edges)
    # 3) concatenate prediction (6 x nb edges aa-aa)
    # add 0 to fill up to 174 dim
