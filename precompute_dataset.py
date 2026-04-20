import os
import torch
import h5py
from multiprocessing import Pool, cpu_count
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from data.streaming_dataset import process_batch

# ===== IMPORT YOUR EXISTING FUNCTIONS =====
# (keep all your RDKit + feature code exactly as is)
# - process_batch
# - seq_to_mol_with_ox
# - get_node_features
# etc.

# =========================================


# =========================
# CONFIG
# =========================
CHUNK_SIZE = 10240
BATCH_SIZE = 1024
N_WORKERS = min(cpu_count(), 8)


# =========================
# SAVE CHUNK + METADATA
# =========================
def save_chunk(buffer, out_dir, chunk_id):
    path = os.path.join(out_dir, f"chunk_{chunk_id}.pt")
    torch.save(buffer, path)

    meta_path = os.path.join(out_dir, "meta.txt")
    with open(meta_path, "a") as f:
        f.write(f"{path},{len(buffer)}\n")


# =========================
# MAIN PREPROCESS FUNCTION
# =========================
def preprocess_to_chunks(data_source, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # reset metadata
    open(os.path.join(out_dir, "meta.txt"), "w").close()

    buffer = []
    chunk_id = 0

    with h5py.File(data_source, "r") as f:
        intensity = f["intensities_raw"]
        sequence = f["sequence_integer"]
        precursor_charge_onehot = f["precursor_charge_onehot"]
        energy_list = f["collision_energy_aligned"]

        length = intensity.shape[0]

        for start in range(0, length, BATCH_SIZE):
            end = min(start + BATCH_SIZE, length)
            print(f"Processing {start}-{end}/{length}")

            batch_data = process_batch(
                start, end,
                sequence,
                intensity,
                precursor_charge_onehot,
                energy_list
            )

            for data in batch_data:
                buffer.append(data)

                if len(buffer) >= CHUNK_SIZE:
                    save_chunk(buffer, out_dir, chunk_id)
                    buffer = []
                    chunk_id += 1

        if buffer:
            save_chunk(buffer, out_dir, chunk_id)

    print("Preprocessing DONE ✅")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    preprocess_to_chunks(
        data_source="/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/train_hcd_reduce.hdf5",
        out_dir="processed_graphs_train_hcd_reduced"
    )
    preprocess_to_chunks(
        data_source="/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/val_hcd_reduce.hdf5",
        out_dir="processed_graphs_val_hcd_reduced"
    )