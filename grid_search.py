import itertools
import subprocess

hidden_dims = [128]

lrs = [1e-4,1e-5,1e-6]

num_layers = [5]

replicate = [0,1,2]

dropouts = [0]

configs = list(
    itertools.product(
        hidden_dims,
        lrs,
        num_layers,
        dropouts,
        replicate,
    )
)

for i, (hidden_dim, lr, num_layers, dropout, rep) in enumerate(configs):

    print(
        f"Run {i+1}/{len(configs)} | "
        f"hd={hidden_dim} "
        f"lr={lr} "
        f"layers={num_layers} "
        f"dropout={dropout} "
    )

    cmd = [
        "python",
        "main.py",
        "--hidden_dim", str(hidden_dim),
        "--lr", str(lr),
        "--num_layers", str(num_layers),
        "--save_path", f'saved_model/EGNN_lr_{lr}_{rep}.pt',
        "--dropout", str(dropout),
        "--max_steps", str(10000),
        "--model_type", "EGNN",
        "--scheduler", "plateau",
    ]

    subprocess.run(cmd)