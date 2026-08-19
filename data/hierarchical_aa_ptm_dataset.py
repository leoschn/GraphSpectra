"""
Alternative hierarchical graph creation pipeline for amino-acid sequences with PTMs.

Pipeline
--------
AA/PTM sequence -> RDKit molecule/SMILES -> PyG hierarchical graph

This module is designed as an alternative to hierarchical_streaming_dataset.py.
It reuses graph_creation_utils.py for atom/bond/AA features, while using the
residue library in mol_builder.py so modified residues can be represented by
explicit SMILES templates.

Supported residue notation
---------------------------
Examples:
    "ACDEFG"
    "ACDM(ox)"
    "ACK(cr)GY(ph)"

The tokenizer accepts compact residue-attached PTM codes only: one amino-acid
letter, optionally followed by a parenthesized PTM code. Each resulting token
must be present in mol_builder.RESIDUES.
"""

from __future__ import annotations

import bisect
import os
import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_undirected

import graph_creation_utils as gcu
from mol_builder import RESIDUES, _connect

# ---------------------------------------------------------------------------
# Sequence parsing
# ---------------------------------------------------------------------------

_STANDARD_AA_TOKENS = set("ACDEFGHIKLMNPQRSTVWY")
_PTM_TOKENS = sorted(
    [token for token in RESIDUES if len(token) > 1],
)


def tokenize_aa_ptm_sequence(
    sequence: str,
) -> list[str]:
    """
    Convert an AA/PTM sequence into residue tokens.

    Accepted form:
      - compact residue-attached notation:
            "ACDM(ox)"
            "ACK(cr)GY(ph)"

    Lists, separator-delimited tokens, and prefix/suffix PTM aliases are not
    accepted. PTM tokens must match mol_builder.RESIDUES exactly.
    """
    if not isinstance(sequence, str):
        raise TypeError(
            "AA/PTM sequence must be a compact string such as 'ACDM(ox)'."
        )

    text = sequence.strip().strip("_")
    if not text:
        raise ValueError("Empty peptide sequence.")

    if any(sep in text for sep in [",", ";", "-", " ", "\t", "\n"]):
        raise ValueError(
            "AA/PTM sequence must be compact, for example 'ACDM(ox)'."
        )

    tokens = []
    i = 0
    while i < len(text):
        residue = text[i]
        if residue not in _STANDARD_AA_TOKENS:
            raise ValueError(
                f"Expected an amino-acid letter at character {i}: "
                f"{text[i:i + 20]!r}."
            )

        i += 1
        token = residue

        if i < len(text) and text[i] == "(":
            end = text.find(")", i + 1)
            if end == -1:
                raise ValueError(
                    f"Unclosed PTM code after residue {residue!r} at "
                    f"character {i - 1}."
                )
            token = f"{residue}{text[i:end + 1]}"
            i = end + 1

        if token not in RESIDUES:
            raise KeyError(
                f"Unknown residue/PTM token: {token!r}. "
                f"Available PTM tokens include: {_PTM_TOKENS[:10]}."
            )

        tokens.append(token)

    return tokens


# ---------------------------------------------------------------------------
# Molecule construction with residue metadata
# ---------------------------------------------------------------------------

def _add_residue_metadata(mol: Chem.Mol, residue_number: int, residue_name: str) -> None:
    """
    Attach PDB-style residue metadata to every atom in a residue template.

    graph_creation_utils.split_peptide_by_residue() uses residue numbers to
    construct AA-level nodes. PTM atoms therefore receive the same residue
    number as the modified amino acid.
    """
    for atom in mol.GetAtoms():
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueNumber(int(residue_number))
        info.SetResidueName(str(residue_name)[:3].ljust(3))
        info.SetName(f"{atom.GetSymbol():>4}"[:4])
        atom.SetMonomerInfo(info)


def _build_ptm_aware_molecule(tokens: Sequence[str]) -> Chem.Mol:
    """
    Build a peptide from mol_builder.RESIDUES while preserving residue IDs.

    Unlike mol_builder.build_peptide(), this version annotates each residue
    before residues are connected. RDKit's CombineMols therefore retains the
    metadata required for AA-level graph nodes.
    """
    if not tokens:
        raise ValueError("Cannot build an empty peptide.")

    if all(token in _STANDARD_AA_TOKENS for token in tokens):
        mol = Chem.MolFromSequence("".join(tokens))
        if mol is None:
            raise ValueError(f"Failed to build peptide from sequence: {tokens!r}.")
        return mol

    residue_mols = []
    for residue_number, token in enumerate(tokens, start=1):
        residue = Chem.MolFromSmiles(RESIDUES[token])
        if residue is None:
            raise ValueError(f"Invalid residue template for {token!r}.")
        _add_residue_metadata(residue, residue_number, token)
        residue_mols.append(residue)

    mol = residue_mols[0]
    for residue in residue_mols[1:]:
        mol = _connect(mol, residue)

    # Convert the terminal C-terminal dummy into OH, retaining the residue
    # metadata of the terminal carbonyl carbon.
    rw = Chem.RWMol(mol)

    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue

        amap = atom.GetAtomMapNum()
        if amap != 2:
            continue

        neighbor_idx = atom.GetNeighbors()[0].GetIdx()
        neighbor = rw.GetAtomWithIdx(neighbor_idx)

        oxygen = Chem.Atom("O")
        info = neighbor.GetMonomerInfo()
        if info is not None:
            oxygen.SetMonomerInfo(info)
        oxygen_idx = rw.AddAtom(oxygen)
        rw.AddBond(neighbor_idx, oxygen_idx, Chem.BondType.SINGLE)

    # Remove remaining N/C terminal dummy atoms.
    dummy_indices = [
        atom.GetIdx()
        for atom in rw.GetAtoms()
        if atom.GetAtomicNum() == 0
    ]
    for idx in sorted(dummy_indices, reverse=True):
        rw.RemoveAtom(idx)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def aa_ptm_to_smiles(
    sequence: str,
    canonical: bool = True,
) -> tuple[list[str], str]:
    """
    AA/PTM sequence -> residue tokens + RDKit SMILES.
    """
    tokens = tokenize_aa_ptm_sequence(sequence)
    mol = _build_ptm_aware_molecule(tokens)
    smiles = Chem.MolToSmiles(
        mol,
        canonical=canonical,
        isomericSmiles=True,
    )
    return tokens, smiles


def aa_ptm_to_mol(
    sequence: str,
) -> tuple[list[str], Chem.Mol]:
    """AA/PTM sequence -> residue tokens + RDKit molecule."""
    tokens = tokenize_aa_ptm_sequence(sequence)
    return tokens, _build_ptm_aware_molecule(tokens)


# ---------------------------------------------------------------------------
# Hierarchical graph construction
# ---------------------------------------------------------------------------

EDGE_TYPE_DIM = 4
NODE_TYPE_DIM = 3


def _pad_features(features: np.ndarray, before_dim: int, after_dim: int) -> np.ndarray:
    return np.concatenate(
        [
            np.zeros((features.shape[0], before_dim), dtype=np.float32),
            features.astype(np.float32),
            np.zeros((features.shape[0], after_dim), dtype=np.float32),
        ],
        axis=1,
    )


def _repeat_global_features(global_features: np.ndarray, num_nodes: int) -> np.ndarray:
    if num_nodes == 0:
        return np.empty((0, global_features.shape[0]), dtype=np.float32)
    return np.tile(global_features.reshape(1, -1), (num_nodes, 1)).astype(np.float32)


def _edge_array(edges: list[tuple[int, int]]) -> np.ndarray:
    if not edges:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64)


def _build_hierarchical_arrays(
    mol: Chem.Mol,
    charge_ohe: Optional[np.ndarray] = None,
    energy: Optional[np.ndarray] = None,
    with_position: bool = True,
    y: Optional[Union[np.ndarray, Sequence[float], float]] = None,
    exclude_feature=None,
) -> dict:
    """
    Convert an RDKit peptide molecule into the same 3-level hierarchy as the
    original dataset:

        atom -> atom
        atom -> AA
        AA -> AA
        AA -> global

    Node types are:
        [1,0,0] atom
        [0,1,0] AA
        [0,0,1] global

    The output is numpy arrays, matching the intermediate representation used
    by hierarchical_streaming_dataset.py.
    """
    if charge_ohe is None:
        charge_ohe = np.empty(0, dtype=np.float32)
    else:
        charge_ohe = np.asarray(charge_ohe, dtype=np.float32).reshape(-1)

    if energy is None:
        energy = np.empty(0, dtype=np.float32)
    else:
        energy = np.asarray(energy, dtype=np.float32).reshape(-1)

    global_features = np.concatenate([charge_ohe, energy]).astype(np.float32)

    if with_position:
        feature_exclusion = exclude_feature
    else:
        feature_exclusion = (
            ["conformation"]
            if exclude_feature is None
            else list(set(exclude_feature) | {"conformation"})
        )

    mol_feats = gcu.precompute_mol_features(
        mol,
        exclude_feature=feature_exclusion,
    )

    x_atom = gcu.get_node_features(
        mol,
        mol_feats=mol_feats,
        exclude_feature=feature_exclusion,
    )
    x_aa = gcu.get_aa_node_features(mol, exclude_feature=feature_exclusion)

    # Global node has no intrinsic feature vector; it only receives the
    # experiment-level/global features.
    x_global = np.zeros((1, 0), dtype=np.float32)

    atom_dim = x_atom.shape[1]
    aa_dim = x_aa.shape[1]

    x_atom = _pad_features(x_atom, 0, aa_dim)
    x_aa = _pad_features(x_aa, atom_dim, 0)
    x_global = _pad_features(x_global, atom_dim + aa_dim, 0)

    x_atom = np.concatenate(
        [x_atom, _repeat_global_features(global_features, x_atom.shape[0])],
        axis=1,
    )
    x_aa = np.concatenate(
        [x_aa, _repeat_global_features(global_features, x_aa.shape[0])],
        axis=1,
    )
    x_global = np.concatenate(
        [x_global, _repeat_global_features(global_features, 1)],
        axis=1,
    )

    node_type = np.concatenate(
        [
            np.tile([1, 0, 0], (x_atom.shape[0], 1)),
            np.tile([0, 1, 0], (x_aa.shape[0], 1)),
            np.tile([0, 0, 1], (x_global.shape[0], 1)),
        ],
        axis=0,
    ).astype(np.float32)

    x = np.concatenate(
        [np.concatenate([x_atom, x_aa, x_global], axis=0), node_type],
        axis=1,
    )

    # -------------------- hierarchy edges --------------------
    total_atom = mol.GetNumAtoms()
    total_aa = x_aa.shape[0]
    global_idx = total_atom + total_aa

    edges_atom_atom = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
    ]

    # Residue IDs come from the metadata added during molecule construction.
    residue_numbers = sorted(
        {
            atom.GetMonomerInfo().GetResidueNumber()
            for atom in mol.GetAtoms()
            if atom.GetMonomerInfo() is not None
        }
    )

    expected_residue_numbers = list(range(1, len(residue_numbers) + 1))
    if residue_numbers != expected_residue_numbers:
        raise ValueError(
            f"Residue metadata is not contiguous: {residue_numbers}"
        )

    residue_to_node = {
        residue_number: total_atom + idx
        for idx, residue_number in enumerate(residue_numbers)
    }

    edges_atom_aa = []
    for atom in mol.GetAtoms():
        info = atom.GetMonomerInfo()
        if info is None:
            raise ValueError(
                f"Atom {atom.GetIdx()} has no residue metadata."
            )
        residue_number = info.GetResidueNumber()
        edges_atom_aa.append(
            (atom.GetIdx(), residue_to_node[residue_number])
        )

    edges_aa_aa = [
        (total_atom + idx, total_atom + idx + 1)
        for idx in range(total_aa - 1)
    ]

    edges_aa_global = [
        (total_atom + idx, global_idx)
        for idx in range(total_aa)
    ]

    edges = np.concatenate(
        [
            _edge_array(edges_atom_atom),
            _edge_array(edges_atom_aa),
            _edge_array(edges_aa_aa),
            _edge_array(edges_aa_global),
        ],
        axis=0,
    )
    edge_index = edges.T

    bond_dim = gcu.get_edge_dim(exclude_feature=feature_exclusion)

    edge_attr_atom_atom = np.concatenate(
        [
            gcu.get_edge_features(mol, exclude_feature=feature_exclusion),
            np.tile([1, 0, 0, 0], (len(edges_atom_atom), 1)),
        ],
        axis=1,
    )

    edge_attr_atom_aa = np.concatenate(
        [
            np.zeros((len(edges_atom_aa), bond_dim), dtype=np.float32),
            np.tile([0, 1, 0, 0], (len(edges_atom_aa), 1)),
        ],
        axis=1,
    )

    edge_attr_aa_aa = np.concatenate(
        [
            np.zeros((len(edges_aa_aa), bond_dim), dtype=np.float32),
            np.tile([0, 0, 1, 0], (len(edges_aa_aa), 1)),
        ],
        axis=1,
    )

    edge_attr_aa_global = np.concatenate(
        [
            np.zeros((len(edges_aa_global), bond_dim), dtype=np.float32),
            np.tile([0, 0, 0, 1], (len(edges_aa_global), 1)),
        ],
        axis=1,
    )

    edge_attr = np.concatenate(
        [
            edge_attr_atom_atom,
            edge_attr_atom_aa,
            edge_attr_aa_aa,
            edge_attr_aa_global,
        ],
        axis=0,
    ).astype(np.float32)

    if with_position:
        pos = np.concatenate(
            [
                gcu.get_atom_positions(mol, mol_feats=mol_feats),
                gcu.get_aa_node_positions(mol, mol_feats=mol_feats),
                np.zeros((1, 3), dtype=np.float32),
            ],
            axis=0,
        )
    else:
        pos = None

    if y is None:
        y_array = np.empty(0, dtype=np.float32)
    else:
        y_array = np.asarray(y, dtype=np.float32).reshape(-1)

    result = {
        "x": x.astype(np.float32),
        "edge_index": edge_index.astype(np.int64),
        "edge_attr": edge_attr,
        "y": y_array,
        "residue_numbers": np.asarray(residue_numbers, dtype=np.int64),
    }
    if pos is not None:
        result["pos"] = pos.astype(np.float32)

    return result


def aa_ptm_to_pyg(
    sequence: str,
    charge_ohe: Optional[np.ndarray] = None,
    energy: Optional[np.ndarray] = None,
    y: Optional[Union[np.ndarray, Sequence[float], float]] = None,
    with_position: bool = True,
    exclude_feature=None,
    return_smiles: bool = False,
) -> Union[Data, tuple[Data, str]]:
    """
    Main user-facing function:

        AA/PTM sequence -> RDKit molecule -> SMILES -> PyG Data

    The graph has atom, AA/residue, and global nodes and four edge types:
        atom-atom, atom-AA, AA-AA, AA-global.

    `charge_ohe`, `energy`, and `y` are optional so the function can be used
    independently of the spectra dataset.
    """
    tokens, mol = aa_ptm_to_mol(sequence)
    arrays = _build_hierarchical_arrays(
        mol,
        charge_ohe=charge_ohe,
        energy=energy,
        with_position=with_position,
        y=y,
        exclude_feature=exclude_feature,
    )

    edge_index = torch.from_numpy(arrays["edge_index"]).long()
    edge_attr = torch.from_numpy(arrays["edge_attr"]).float()

    # Match the original dataset: hierarchy edges are created once and made
    # undirected during PyG conversion.
    edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    kwargs = {
        "x": torch.from_numpy(arrays["x"]).float(),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": torch.from_numpy(arrays["y"]).float(),
    }
    if with_position:
        kwargs["pos"] = torch.from_numpy(arrays["pos"]).float()

    data = Data(**kwargs)
    data.sequence = tokens
    data.smiles = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )
    data.num_atom_nodes = mol.GetNumAtoms()
    data.num_aa_nodes = len(tokens)
    data.global_node_index = mol.GetNumAtoms() + len(tokens)

    if return_smiles:
        return data, data.smiles
    return data


# ---------------------------------------------------------------------------
# Dataset/chunk processing
# ---------------------------------------------------------------------------

def process_sequence_batch(
    sequences: Sequence[str],
    labels: Optional[Sequence] = None,
    charge: Optional[Sequence] = None,
    energy: Optional[Sequence] = None,
    with_position: bool = True,
    exclude_feature=None,
) -> list[Data]:
    """
    Convert a batch of AA/PTM sequences into PyG Data objects.

    Unlike the original multiprocessing implementation, this function does
    not require HDF5 integer-encoded sequences. It accepts the actual
    sequence/PTM notation directly.
    """
    data_list = []

    for i, sequence in enumerate(sequences):
        try:
            label_i = None if labels is None else labels[i]
            charge_i = None if charge is None else charge[i]
            energy_i = None if energy is None else energy[i]

            data = aa_ptm_to_pyg(
                sequence,
                charge_ohe=charge_i,
                energy=energy_i,
                y=label_i,
                with_position=with_position,
                exclude_feature=exclude_feature,
            )
            data_list.append(data)

        except Exception as exc:
            print(f"[Sequence ERROR] index {i}: {exc}")

    return data_list


class HierarchicalStreamingAAPTMDataset(Dataset):
    """
    Streaming PyG dataset compatible with chunk files generated by this module.

    Expected:
        root/
            meta.txt
            chunk_000000.pt
            chunk_000001.pt
            ...

    Each chunk file is a torch-saved list[torch_geometric.data.Data].
    meta.txt contains:
        /path/to/chunk_000000.pt,<number_of_graphs>
        /path/to/chunk_000001.pt,<number_of_graphs>
    """

    def __init__(self, root: str):
        super().__init__(root)

        self.root = root
        meta_file = os.path.join(root, "meta.txt")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Missing metadata file: {meta_file}")

        self.chunk_files = []
        self.cumulative_sizes = []
        total = 0

        with open(meta_file, "r") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                path, size = line.split(",", 1)
                size = int(size)

                self.chunk_files.append(path)
                total += size
                self.cumulative_sizes.append(total)

        self.total_len = total
        self.cache = None
        self.current_chunk_idx = -1

    def len(self) -> int:
        return self.total_len

    def get(self, idx: int) -> Data:
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Graph index {idx} out of range [0, {self.total_len}).")

        chunk_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        start = 0 if chunk_idx == 0 else self.cumulative_sizes[chunk_idx - 1]
        local_idx = idx - start

        if chunk_idx != self.current_chunk_idx:
            self.cache = torch.load(
                self.chunk_files[chunk_idx],
                weights_only=False,
            )
            self.current_chunk_idx = chunk_idx

        return self.cache[local_idx]

    def chunk_shuffle(self) -> None:
        """Shuffle all complete chunks except the final chunk."""
        if len(self.chunk_files) <= 1:
            return

        rest = self.chunk_files[:-1]
        random.shuffle(rest)
        self.chunk_files[:-1] = rest


def write_chunks(
    sequences: Sequence[str],
    output_dir: str,
    chunk_size: int = 1000,
    labels: Optional[Sequence] = None,
    charge: Optional[Sequence] = None,
    energy: Optional[Sequence] = None,
    with_position: bool = True,
    exclude_feature=None,
) -> None:
    """
    Build graphs and write them as PyTorch chunk files plus meta.txt.

    This is the direct replacement for the preprocessing side of the
    original streaming dataset when the source data is already AA/PTM text.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    meta_lines = []

    for start in range(0, len(sequences), chunk_size):
        end = min(start + chunk_size, len(sequences))

        labels_chunk = None if labels is None else labels[start:end]
        charge_chunk = None if charge is None else charge[start:end]
        energy_chunk = None if energy is None else energy[start:end]

        graphs = process_sequence_batch(
            sequences[start:end],
            labels=labels_chunk,
            charge=charge_chunk,
            energy=energy_chunk,
            with_position=with_position,
            exclude_feature=exclude_feature,
        )

        chunk_path = output / f"chunk_{start:09d}_{end:09d}.pt"
        torch.save(graphs, chunk_path)
        meta_lines.append(f"{chunk_path},{len(graphs)}")

        print(f"Wrote {len(graphs)} graphs -> {chunk_path}")

    with open(output / "meta.txt", "w") as handle:
        handle.write("\n".join(meta_lines) + ("\n" if meta_lines else ""))


# ---------------------------------------------------------------------------
# Minimal smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sequence = "ACDM(ox)"

    data, smiles = aa_ptm_to_pyg(
        sequence,
        with_position=False,
        return_smiles=True,
    )

    print("tokens :", data.sequence)
    print("SMILES :", smiles)
    print("x      :", tuple(data.x.shape))
    print("edges  :", tuple(data.edge_index.shape))
    print("e_attr :", tuple(data.edge_attr.shape))
    print("y      :", tuple(data.y.shape))
