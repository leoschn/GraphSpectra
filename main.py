import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
import os

from data.streaming_dataset import StreamingSpectraDataset
from model.model import AttentiveFPGraphRegressor
from model.losses import masked_spectral_distance
from config import load_args

def train(epoch,train_loader):
    model.train()
    total_loss = 0
    #manual chunk level shuffle
    train_loader.dataset.chunk_shuffle()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]")

    for data in pbar:
        data = data.to(device)

        optimizer.zero_grad()
        out = model(data)

        loss = masked_spectral_distance(out, data.y.view(data.num_graphs, -1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        # Update progress bar
        pbar.set_postfix(loss=loss.item())

        # Log per batch (optional, can comment if too verbose)
        wandb.log({"train_batch_loss": loss.item()})

    epoch_loss = total_loss / len(train_loader.dataset)

    return epoch_loss


@torch.no_grad()
def evaluate(loader, split="val"):
    model.eval()
    total_loss = 0

    pbar = tqdm(loader, desc=f"[{split.upper()}]")

    for data in pbar:
        data = data.to(device)
        out = model(data)

        loss = masked_spectral_distance(out, data.y.view(data.num_graphs, -1))
        total_loss += loss.item() * data.num_graphs

        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader.dataset)


if __name__ == '__main__':

    args = load_args()
    # -----------------------
    # WandB init
    # -----------------------
    os.environ["WANDB_API_KEY"] = 'b4a27ac6b6145e1a5d0ee7f9e2e8c20bd101dccd'
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = os.path.abspath("./wandb_run")

    wandb.init(
        project="attentivefp-spectra",
        config={
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_timesteps": args.num_timesteps,

        }
    )

    config = wandb.config

    # -----------------------
    # Data
    # -----------------------
    train_dataset = StreamingSpectraDataset(root=args.root_train)
    val_dataset = StreamingSpectraDataset(root=args.root_val)
    test_dataset = StreamingSpectraDataset(root=args.root_test)
    print('Data loaded.')
    print('Data dim -- node : ', train_dataset[0].x.shape ,' edge : ',train_dataset[0].edge_attr.shape, ' y : ', train_dataset[0].y.shape )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False,num_workers=6,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,num_workers=6,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,num_workers=6,pin_memory=True)

    # -----------------------
    # Model
    # -----------------------
    model = AttentiveFPGraphRegressor(
        node_feat_dim=train_dataset[0].x.shape[1],
        edge_feat_dim=train_dataset[0].edge_attr.shape[1],
        hidden_dim=config.hidden_dim,
        hidden_layers=args.hidden_layers,
        num_layers=args.num_layers,
        num_timesteps=args.num_timesteps,
        out_dim=174
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device used:', device)
    model = model.to(device)
    print('Model loaded.')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    print('Optimizer loaded.')
    # Optional: track gradients & model
    #wandb.watch(model, log="all", log_freq=100)

    # -----------------------
    # Training loop
    # -----------------------
    print('Starting Training...')
    for epoch in range(1, config.epochs + 1):
        train_loss = train(epoch,train_loader)
        val_loss = evaluate(val_loader, split="val")

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Log per epoch
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })


    # -----------------------
    # Test
    # -----------------------
    test_loss = evaluate(test_loader, split="test")
    print("Test Loss:", test_loss)

    wandb.log({"test_loss": test_loss})

    wandb.finish()