import os
import torch
import h5py
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_undirected

from rdkit import Chem

from dataset import (
    int_to_aa_dict,
    seq_to_mol_with_ox,
    get_node_features,
    get_edge_features,
)


class SpectraPyGDataset(InMemoryDataset):
    def __init__(self, root, split="train", transform=None, pre_transform=None):
        self.split = split  # "train", "val", "test"
        super().__init__(root, transform, pre_transform)

        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[self.split_idx])

    # Map split to index
    @property
    def split_idx(self):
        return {"train": 0, "val": 1, "test": 2}[self.split]

    @property
    def raw_file_names(self):
        return ["train.h5", "val.h5", "test.h5"]

    @property
    def processed_file_names(self):
        return ["train_data.pt", "val_data.pt", "test_data.pt"]

    def process(self):
        for split_idx, raw_path in enumerate(self.raw_paths):
            print(f"Processing {raw_path}...")

            data_list = []

            with h5py.File(raw_path, "r") as f:
                intensity = f["intensities_raw"]
                sequence = f["sequence_integer"]

                for i in tqdm(range(len(intensity))):
                    seq = ''.join(int_to_aa_dict[n] for n in sequence[i].tolist())
                    inty = intensity[i]

                    try:
                        data = self._process_one(seq, inty)
                        data_list.append(data)
                    except Exception as e:
                        print(f"Skipping {i}: {e}")

            # Optional transforms
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            data, slices = self.collate(data_list)

            torch.save((data, slices), self.processed_paths[split_idx])
            print(f"Saved → {self.processed_paths[split_idx]}")

    def _process_one(self, seq, inty):
        # Build molecule
        if "(ox)" in seq:
            mol = seq_to_mol_with_ox(seq)
        else:
            mol = Chem.MolFromSequence(seq)

        # Node features
        x = torch.tensor(get_node_features(mol), dtype=torch.float)

        # Edges
        edge_index = []
        for bond in mol.GetBonds():
            edge_index.append([
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx()
            ])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(get_edge_features(mol), dtype=torch.float)

        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

        # Labels
        y = torch.tensor(inty, dtype=torch.float32).flatten()

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)