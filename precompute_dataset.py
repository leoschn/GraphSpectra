import os
import torch
import h5py
from multiprocessing import Pool, cpu_count
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from data.streaming_dataset import process_batch
from data.hierarchical_streaming_dataset import process_batch_hierarchical

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
CHUNK_SIZE = 2048
BATCH_SIZE = 2048
N_WORKERS = min(cpu_count(), 8)
print('CHUNK_SIZE:', CHUNK_SIZE)
print('BATCH_SIZE:', BATCH_SIZE)
print('N_WORKERS:', cpu_count())


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
def preprocess_to_chunks(data_source, out_dir, hierarchical=False):
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
            if hierarchical:
                batch_data = process_batch_hierarchical(
                    start, end,
                    sequence,
                    intensity,
                    precursor_charge_onehot,
                    energy_list
                )
            else : batch_data = process_batch(
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
    # preprocess_to_chunks(data_source='dataset_dummy/val_hcd_dummy.hdf5', out_dir='dataset_dummy/test',hierarchical=True)

    # print('Processing val hierachical dataset')
    # preprocess_to_chunks(
    #     data_source="/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/raw_dataset/val_hcd_dummy.hdf5",
    #     out_dir="dataset/processed_dataset/hierarchical_dataset/processed_graphs_val_hcd_hierarchical_dummy",hierarchical=True
    # )
    #
    # print('Processing train hierachical dataset')
    # preprocess_to_chunks(
    #     data_source="/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/raw_dataset/train_hcd_dummy.hdf5",
    #     out_dir="dataset/processed_dataset/hierarchical_dataset/processed_graphs_train_hcd_hierarchical_dummy",hierarchical=True
    # )
    #
    #
    #
    # print('Processing test hierachical dataset')
    # preprocess_to_chunks(
    #     data_source="/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/raw_dataset/holdout_hcd_dummy.hdf5",
    #     out_dir="dataset/processed_dataset/hierarchical_dataset/processed_graphs_holdout_hcd_hierarchical_dummy",hierarchical=True
    # )

    print('Processing train baseline dataset')
    preprocess_to_chunks(
        data_source="/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/raw_dataset/train_hcd_dummy.hdf5",
        out_dir="dataset/processed_dataset/baseline_dataset/processed_graphs_train_hcd_baseline_dummy",hierarchical=False
    )

    print('Processing val baseline dataset')
    preprocess_to_chunks(
        data_source="/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/raw_dataset/raw_dataset/val_hcd_dummy.hdf5",
        out_dir="dataset/processed_dataset/baseline_dataset/processed_graphs_val_hcd_baseline_dummy",hierarchical=False
    )

    print('Processing test baseline dataset')
    preprocess_to_chunks(
        data_source="/lustre/fswork/projects/rech/bun/ucg81ws/these/GraphSpectra/dataset/raw_dataset/holdout_hcd_dummy.hdf5",
        out_dir="dataset/processed_dataset/baseline_dataset/processed_graphs_holdout_hcd_baseline_dummy",hierarchical=False
    )