from torch_geometric.nn import AttentiveFP, global_mean_pool
import torch.nn as nn

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