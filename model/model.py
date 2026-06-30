from torch_geometric.nn import AttentiveFP, global_mean_pool, GAT
from torch_geometric.nn.aggr import SetTransformerAggregation
import torch.nn as nn
import torch
from egnn_clean.egnn_clean import EGNN
from data.graph_creation_utils import get_edge_dim


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

    def forward(self, h):

        return self.mlp(h).squeeze(-1)

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

        self.bond_dim = get_edge_dim()

        self.edge_head = EdgeHead(hidden_dim)

    def forward(self, data):
        h = self.gnn(
            data.x,
            data.edge_index,
            data.edge_attr
        )



        src_all, dst_all = data.edge_index

        is_aa = (
                data.edge_attr[:, self.bond_dim + 2]
                == 1
        )

        # only 1 edge per pair (from 2 => undirected)
        aa_mask = is_aa & (src_all < dst_all)

        #aa-aa edges scr/dst
        src = src_all[aa_mask]
        dst = dst_all[aa_mask]

        h1 = h[src]
        h2 = h[dst]

        edge_feat = torch.cat(
            [h1, h2],
            dim=1
        )

        #apply prediction head to each dim (each aa pairs)
        pred_valid = self.edge_head(edge_feat).squeeze(-1).reshape(-1)

        #pex graph : 4 AA bonds graph1 : 3 AA bonds graph2 : 5 AA bonds => PyG batch fuse it in a single graph
        # pred size 12

        edge_batch = data.batch[src]
        # tensor([
        # 0,0,0,0,
        # 1,1,1,
        # 2,2,2,2,2
        # ])

        values, counts_rep = torch.unique(edge_batch, return_counts=True)

        full_frag_batch = values.repeat_interleave(counts_rep * 6)
        # x6 for 3 (charge) x 2 (type) of frags

        counts = torch.bincount(full_frag_batch)
        # [24,18,30]

        starts = torch.cumsum(
            counts,
            dim=0
        ) - counts
        # [0,24,42]


        slot = (
                torch.arange(
                    full_frag_batch.size(0),
                    device=edge_batch.device
                )
                - starts[full_frag_batch]
        )

        #tensor([
        # 0,1,2,3,...23
        # 0,1,2,...17
        # 0,1,2,3,4...29
        # ])

        pred = pred_valid.new_zeros(
            max(edge_batch)+1,
            174
        )
        #full zero tensor

        pred[
            full_frag_batch,
            slot
        ] = pred_valid
        #copy pred int in relevant spot

        return pred

