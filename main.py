import torch
from torch.nn.functional import dropout
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
import os
import itertools
import numpy as np
from data.streaming_dataset import StreamingSpectraDataset
from data.hierarchical_streaming_dataset import HierarchicalStreamingSpectraDataset
from model.model import AttentiveFPGraphRegressor, BaselineGAT, EGNN_predictor, BondBreakPredictor
from model.losses import masked_spectral_distance
from config import load_args
import pandas as pd

def infinite_loader(loader):
    """Creates an infinite dataloader iterator."""
    while True:
        loader.dataset.chunk_shuffle()  # reshuffle chunks every pass
        for batch in loader:
            yield batch


def train_step(data):
    model.train()

    data = data.to(device)

    optimizer.zero_grad()

    out = model(data)

    loss = masked_spectral_distance(
        out,
        data.y.view(data.num_graphs, -1)
    )

    loss = loss.mean()

    loss.backward()
    optimizer.step()

    return loss.item(), data.num_graphs


@torch.inference_mode()
def evaluate(loader, split="val", save_predictions=False):

    model.eval()

    sample_losses = []

    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"[{split.upper()}]")

    for data in pbar:
        data = data.to(device)

        out = model(data)

        # -------- batch loss (for logging) --------
        batch_loss = masked_spectral_distance(
            out,
            data.y.view(data.num_graphs, -1)
        )
        mean_loss = batch_loss.mean()

        pbar.set_postfix(loss=mean_loss.item())

        sample_losses.extend(batch_loss.cpu().numpy())

        if save_predictions:
            all_preds.append(out.cpu())
            all_targets.append(data.y.view(data.num_graphs, -1).cpu())

    mean_loss = np.mean(sample_losses)
    median_loss = np.median(sample_losses)

    results = {
        "mean": mean_loss,
        "median": median_loss,
    }

    if save_predictions:
        results["predictions"] = torch.cat(all_preds, dim=0)
        results["targets"] = torch.cat(all_targets, dim=0)

    return results


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
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_timesteps": args.num_timesteps,
            "model_type": args.model_type,
            "dropout":args.dropout,
            "dataset_train": args.root_train,
            "scheduler":args.scheduler,
        }
    )

    config = wandb.config

    # -----------------------
    # Data
    # -----------------------
    train_dataset = HierarchicalStreamingSpectraDataset(root=args.root_train)
    val_dataset = HierarchicalStreamingSpectraDataset(root=args.root_val)
    test_dataset = HierarchicalStreamingSpectraDataset(root=args.root_test)

    print('Data loaded.')
    print(
        'Data dim -- node : ',
        train_dataset[0].x.shape,
        ' edge : ',
        train_dataset[0].edge_attr.shape,
        ' y : ',
        train_dataset[0].y.shape
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=6,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=6,
        pin_memory=True
    )

    train_iter = infinite_loader(train_loader)

    if config.model_type == "GAT":
        model = BaselineGAT(
            node_feat_dim=train_dataset[0].x.shape[1],
            edge_feat_dim=train_dataset[0].edge_attr.shape[1],
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            out_dim=174,
            dropout=config.dropout
        )
    elif config.model_type == "EGNN":
        model = EGNN_predictor(
            node_feat_dim=train_dataset[0].x.shape[1],
            edge_feat_dim=train_dataset[0].edge_attr.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            out_dim=174
        )

    elif config.model_type == "local_GAT":
        model = BondBreakPredictor(
            node_feat_dim=train_dataset[0].x.shape[1],
            edge_feat_dim=train_dataset[0].edge_attr.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            out_dim=174)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device used:', device)

    model = model.to(device)

    print('Model loaded.')

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,  # validation evaluations, not training steps
            min_lr=1e-6,
            threshold=1e-4,
            threshold_mode="rel"
        )
    current_lr = config.lr


    print('Optimizer loaded.')

    # save path dir creation
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # -----------------------
    # Step-based training loop
    # -----------------------
    best_loss = float('inf')

    print('Starting Training...')

    pbar = tqdm(range(1, config.max_steps + 1), desc="Training")

    for step in pbar:

        data = next(train_iter)

        train_loss, batch_size = train_step(data)

        pbar.set_postfix(train_loss=train_loss)

        # -----------------------
        # Logging
        # -----------------------
        wandb.log({
            "step": step,
            "train_loss": train_loss
        })

        # -----------------------
        # Validation
        # -----------------------
        if step % config.eval_every == 0:

            val_results = evaluate(val_loader, split="val")

            val_loss = val_results["mean"]
            val_median = val_results["median"]

            if config.scheduler == 'plateau':
                scheduler.step(val_loss)

                current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"\nStep {step:06d}"
                f" | Train: {train_loss:.4f}"
                f" | Val Mean: {val_loss:.4f}"
                f" | Val Median: {val_median:.4f}"
                f" | LR: {current_lr:.2e}"
            )

            wandb.log({
                "step": step,
                "val_mean": val_loss,
                "val_median": val_median,
                "lr": current_lr,
            })

            # -----------------------
            # Save best model
            # -----------------------
            if val_loss < best_loss:
                best_loss = val_loss

                torch.save(model.state_dict(), args.save_path)

                print(f"Best model saved at step {step}")

    # -----------------------
    # Test
    # -----------------------
    print("Loading best model...")

    model.load_state_dict(torch.load(args.save_path))

    test_results = evaluate(
        test_loader,
        split="test",
        save_predictions=True
    )

    print("Test Mean:", test_results["mean"])
    print("Test Median:", test_results["median"])

    wandb.log({
        "test_loss": test_results["mean"],
        "test_median": test_results["median"],
    })

    wandb.finish()

    pred = test_results["predictions"].numpy()
    target = test_results["targets"].numpy()

    df_pred = pd.DataFrame(pred)
    df_true = pd.DataFrame(target)

    df_pred.to_csv(os.path.splitext(args.save_path)[0]+"predictions.csv", index=False)
    df_true.to_csv(os.path.splitext(args.save_path)[0]+"predictions.csv", index=False)