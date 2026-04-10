from torch_geometric.nn import AttentiveFP
from data.dataset import *
from torch_geometric.loader import DataLoader
from model.model import AttentiveFPGraphRegressor
from model.losses import masked_spectral_distance

def train():
    model.train()
    total_loss = 0

    for data in train_loader:

        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)
        loss = masked_spectral_distance(out, data.y.view(BATCH_SIZE, -1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        out = model(data)

        loss = masked_spectral_distance(out, data.y)
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


if __name__ =='__main__':

    train_dataset = SpectraGraphDataset(data_source='dataset/traintest_hcd.hdf5',label_type='scarce')
    val_dataset = SpectraGraphDataset(data_source='dataset/holdout_hcd.hdf5',label_type='scarce')
    test_dataset = SpectraGraphDataset(data_source='dataset/holdout_hcd.hdf5',label_type='scarce')
    BATCH_SIZE = 32

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)



    model = AttentiveFPGraphRegressor(node_feat_dim=train_dataset.node_dim, edge_feat_dim=train_dataset.edge_dim,
                                      hidden_dim=128, out_dim=174)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



    for epoch in range(1, 101):
        train_loss = train()
        val_loss = evaluate(val_loader)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    test_loss = evaluate(test_loader)
    print("Test Loss:", test_loss)