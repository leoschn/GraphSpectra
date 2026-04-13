import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
import os

from data.dataset import *
from model.model import AttentiveFPGraphRegressor
from model.losses import masked_spectral_distance
from config import load_args

def train(epoch):
    model.train()
    total_loss = 0

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
    # Data
    # -----------------------
    train_dataset = SpectraGraphDatasetPrec(
        data_source=args.dataset_train, label_type='scarce',root=args.root_train,
    )
    val_dataset = SpectraGraphDatasetPrec(
        data_source=args.dataset_val, label_type='scarce',root=args.root_val,
    )
    test_dataset = SpectraGraphDatasetPrec(
        data_source=args.dataset_test, label_type='scarce',root=args.root_test,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,num_workers=6,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,num_workers=6,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,num_workers=6,pin_memory=True)

