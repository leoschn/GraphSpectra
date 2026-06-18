import itertools
import subprocess

hidden_dims = [64, 128, 256]

lrs = [1e-4, 1e-3, 1e5]

num_layers = [3, 5, 7]


configs = list(
    itertools.product(
        hidden_dims,
        lrs,
        num_layers,
        seeds
    )
)

for i, (hidden_dim, lr, num_layers, seed) in enumerate(configs):

    print(
        f"Run {i+1}/{len(configs)} | "
        f"hd={hidden_dim} "
        f"lr={lr} "
        f"layers={num_layers} "
    )

    cmd = [
        "python",
        "main.py",
        "--hidden_dim", str(hidden_dim),
        "--lr", str(lr),
        "--num_layers", str(num_layers),
    ]

    subprocess.run(cmd)