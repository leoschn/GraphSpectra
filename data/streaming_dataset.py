import numpy as np
from typing import Union, List
import pandas as pd
import torch
import ast
import h5py
import os
import bisect
import random

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.data import Dataset

from multiprocessing import Pool, cpu_count
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

from data.graph_creation_utils import *


# =========================
# GLOBALS for multiprocessing
# =========================
SEQ = None
INTY = None
CHARGE = None
ENERGY = None


# =========================
# Worker initializer
# =========================
def init_worker(seq, inty, charge, energy):
    global SEQ, INTY, CHARGE, ENERGY
    SEQ = seq
    INTY = inty
    CHARGE = charge
    ENERGY = energy


# =========================
# Worker function
# =========================

def process_one(i):
    try :
        seq = ''.join(int_to_aa_dict[n] for n in SEQ[i].tolist())
        inty = INTY[i]
        charge_ohe = CHARGE[i]
        energy = ENERGY[i]

        charge = np.argmax(charge_ohe)

        # ---- molecule ----
        if '(ox)' in seq:
            mol = seq_to_mol_with_ox(seq)
        else:
            mol = Chem.MolFromSequence(seq)

        if mol is None:
            return None

        mol_feats = precompute_mol_features(mol)

        # ---- node features ----
        x_local = get_node_features(
            mol,
            mol_feats=mol_feats,
        )
        x_global = get_global_feature(mol, charge_ohe, energy)
        x = np.concatenate([x_local, x_global], axis=1)
        pos = get_atom_positions(mol, mol_feats=mol_feats)

        # ---- edges ----
        edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
        if len(edges) == 0:
            return None

        edge_index = np.array(edges).T
        edge_attr = get_edge_features(mol)


        # ---- labels ----
        y = np.array(inty, dtype=np.float32)

        return {
            "x": x,
            "pos": pos,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "y": y
        }

    except Exception as e:
        print(f"[Worker ERROR] index {i}: {e}")
        return None


# =========================
# Batch processing with multiprocessing
# =========================
def process_batch(start, end, sequence, intensity, charge, energy):
    try:
        seq_batch = sequence[start:end]
        inty_batch = intensity[start:end]
        charge_batch = charge[start:end]
        energy_batch = energy[start:end]

        n_workers = min(cpu_count(), 8)

        with Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(seq_batch, inty_batch, charge_batch, energy_batch)
        ) as pool:
            results = pool.map(process_one, range(end - start))

        data_list = []
        for r in results:
            if r is None:
                continue

            try:
                edge_index = torch.from_numpy(r["edge_index"]).long()
                edge_attr = torch.from_numpy(r["edge_attr"]).float()

                edge_index, edge_attr = to_undirected(edge_index, edge_attr)

                data = Data(
                    x=torch.from_numpy(r["x"]).float(),
                    pos=torch.from_numpy(r["pos"]).float(),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.from_numpy(r["y"]).float()
                )

                data_list.append(data)

            except Exception as e:
                print(f"[Post-process ERROR]: {e}")
                continue

        return data_list

    except Exception as e:
        print(f"[Batch ERROR] {start}-{end}: {e}")
        return []



class StreamingSpectraDataset(Dataset):
    def __init__(self, root):
        super().__init__(root)

        self.root = root
        meta_file = os.path.join(root, "meta.txt")

        self.chunk_files = []
        self.cumulative_sizes = []

        total = 0

        print("Loading metadata...")

        with open(meta_file, "r") as f:
            for line in f:
                path, size = line.strip().split(",")
                size = int(size)

                self.chunk_files.append(path)
                total += size
                self.cumulative_sizes.append(total)

        self.total_len = total

        self.cache = None
        self.current_chunk_idx = -1

        print(f"Total graphs: {self.total_len}")

    def chunk_shuffle(self): #exclude last chunk since it is incomplete
        rest = self.chunk_files[:-1]
        random.shuffle(rest)
        self.chunk_files[:-1] = rest

    def len(self):
        return self.total_len

    def get(self, idx):
        # fast binary search
        chunk_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        start = 0 if chunk_idx == 0 else self.cumulative_sizes[chunk_idx - 1]
        local_idx = idx - start

        # load chunk if needed
        if chunk_idx != self.current_chunk_idx:
            self.cache = torch.load(self.chunk_files[chunk_idx],weights_only=False)
            self.current_chunk_idx = chunk_idx

        return self.cache[local_idx]
