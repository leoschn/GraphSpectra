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
        super().__init__()

        self.gnn = EGNN(
            in_node_nf=node_feat_dim,
            in_edge_nf=edge_feat_dim,
            hidden_nf=hidden_dim,
            n_layers=num_layers,
            out_node_nf=hidden_dim,
        )

        self.readout = SetTransformerAggregation(channels=hidden_dim, heads=8)

        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, _ = self.gnn(h=data.x, x=data.pos, edges=data.edge_index, edge_attr=data.edge_attr)
        x_read = self.readout(x,index=data.batch)
        out = self.lin(x_read)
        return out


class EdgeHead(nn.Module):

    def __init__(self, emb_dim):

        super().__init__()

        self.mlp = nn.Sequential(

            nn.Linear(2*emb_dim, 256),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(256, 6)
        )

    def forward(self, h, edge_index):

        src, dst = edge_index

        h1 = h[src]

        h2 = h[dst]

        edge_feat = torch.cat([
            h1,
            h2,
        ], dim=1)

        return self.mlp(edge_feat).squeeze(-1)

class BondBreakPredictor(nn.Module):

    def __init__(self, node_feat_dim=3, edge_feat_dim=3, hidden_dim=128, out_dim=174, num_layers=3, dropout=0.):
        super().__init__()

        self.gnn = GAT(
            in_channels=node_feat_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            v2=True,
            edge_dim=edge_feat_dim,
            dropout=dropout,
        )


        self.edge_head = EdgeHead(hidden_dim)

    def forward(self, data):

        h = self.gnn(
            data.x,
            data.edge_index,
            data.edge_attr
        )

        aa_edges = data.edge_index[
            :,
            data.aa_edge_mask
        ]

        pred = self.edge_head(
            h,
            aa_edges
        )

        return pred