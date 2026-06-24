import itertools
import subprocess

hidden_dims = [128]

lrs = [1e-4]

num_layers = [5]

replicate = [0,1,2,3,4]

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
        "--save_path", f'saved_model/baselineGAT_long_train_{rep}.pt',
        "--dropout", str(dropout)
    ]

    subprocess.run(cmd)