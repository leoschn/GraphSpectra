import itertools
import subprocess

hidden_dims = [64, 128, 256]

lrs = [1e-1,1e-2,1e-3]

num_layers = [3, 5, 7]


configs = list(
    itertools.product(
        hidden_dims,
        lrs,
        num_layers,
    )
)

for i, (hidden_dim, lr, num_layers) in enumerate(configs):

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
        "--save_path", f'saved_model_2/GAT_layer_{num_layers}_dim_{hidden_dim}_lr_{lr}.pt',
        "--max_steps", str(2500),
        "--model_type", "GAT",
    ]

    subprocess.run(cmd)