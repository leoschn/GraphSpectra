import numpy as np
from typing import Union, List
import pandas as pd
import torch
import ast
import h5py
import os

from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.data import InMemoryDataset


from rdkit import Chem
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdPartialCharges

alphabet = [
    "",
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "M(ox)"
]

aa_to_int_dict = dict((aa, i) for i, aa in enumerate(alphabet))

int_to_aa_dict = dict((i, aa) for i, aa in enumerate(alphabet))

from rdkit import Chem
from rdkit.Chem import rdmolops
import re


# --- Step 1: Parse sequence and detect oxidized methionine ---
def parse_sequence(seq):
    """
    Returns:
        clean_seq: sequence without modifications
        ox_positions: list of positions (0-based) of M(ox)
    """
    ox_positions = []
    clean_seq = ""

    i = 0
    pos = 0
    while i < len(seq):
        if seq[i] == "M" and seq[i:i + 5] == "M(ox)":
            clean_seq += "M"
            ox_positions.append(pos)
            i += 5
        else:
            clean_seq += seq[i]
            i += 1
        pos += 1

    return clean_seq, ox_positions


# --- Step 2: Build peptide with RDKit ---
def build_peptide(seq):
    """
    Build peptide molecule from clean sequence
    """
    mol = Chem.MolFromFASTA(seq)
    if mol is None:
        raise ValueError("Failed to build peptide from sequence")
    return mol


# --- Step 3: Oxidize methionine sulfur ---
def oxidize_methionine(mol, residue_index):
    """
    Adds =O to sulfur atom of a methionine residue
    residue_index: 0-based index in peptide
    """
    mol = Chem.RWMol(mol)

    # RDKit stores residue info in atom properties
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            continue

        if info.GetResidueNumber() == residue_index + 1:
            # Find sulfur atom in methionine
            if atom.GetSymbol() == "S":
                s_idx = atom.GetIdx()

                # Add oxygen atom
                o_idx = mol.AddAtom(Chem.Atom("O"))

                # Add double bond S=O
                mol.AddBond(s_idx, o_idx, Chem.BondType.DOUBLE)

                break

    return mol.GetMol()


# --- Full pipeline ---
def seq_to_mol_with_ox(seq):
    clean_seq, ox_positions = parse_sequence(seq)

    mol = build_peptide(clean_seq)

    for pos in ox_positions:
        mol = oxidize_methionine(mol, pos)

    Chem.SanitizeMol(mol)
    return mol




#from deepGCN-RT
atom_features = [
    'chiral_center',# dim 1
    'cip_code', # dim 2
    'crippen_log_p_contrib', #dim 1
    'crippen_molar_refractivity_contrib', #dim 1
    'degree', #dim 6
    'element', #dim 12
    'formal_charge', #dim 3
    'gasteiger_charge', #dim 1
    'hybridization', #dim 6
    'is_aromatic',#dim 1
    'is_h_acceptor',#dim 1
    'is_h_donor',#dim 1
    'is_hetero',#dim 1
    'is_in_ring_size_n',#dim 9
    'labute_asa_contrib',#dim 1
    'mass',#dim 1
    'num_hs',#dim 5
    'num_radical_electrons',#dim 3
    'num_valence',#dim 7
    'tpsa_contrib',#dim 1
]

bond_features = [
    'bondstereo',#dim 4
    'bondtype',#dim 4
    'is_conjugated',#dim 1
    'is_in_ring',#dim 1
    'is_rotatable',#dim 1
]

'''adopted from: https://github.com/akensert/GCN-retention-time-predictions'''

def onehot_encode(x: Union[float, int, str],
                  allowable_set: List[Union[float, int, str]]) -> List[float]:
    return list(map(lambda s: float(x == s), allowable_set))

def encode(x: Union[float, int, str]) -> List[float]:
    if x is None or np.isnan(x):
        x = 0.0
    return [float(x)]

def bond_featurizer(bond: Chem.Bond,exclude_feature) -> np.ndarray:
    new_bond_features = [i for i in bond_features if i != exclude_feature]
    return np.concatenate([
        globals()[bond_feature](bond) for bond_feature in new_bond_features
    ], axis=0)

def atom_featurizer(atom, mol_feats, exclude_feature):
    new_atom_features = [i for i in atom_features if i != exclude_feature]

    features = []

    for atom_feature in new_atom_features:
        if atom_feature in [
            'crippen_log_p_contrib',
            'crippen_molar_refractivity_contrib',
            'tpsa_contrib',
            'labute_asa_contrib',
            'gasteiger_charge'
        ]:
            features.append(globals()[atom_feature](atom, mol_feats)) #molecule level prop
        else:
            features.append(globals()[atom_feature](atom)) #atome level prop

    return np.concatenate(features, axis=0)

def bondtype(bond: Chem.Bond) -> List[float]:
    return onehot_encode(
        x=bond.GetBondType(),
        allowable_set=[
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
    )

def is_in_ring(bond: Chem.Bond) -> List[float]:
    return encode(
        x=bond.IsInRing()
    )

def is_conjugated(bond):
    return encode(
        x=bond.GetIsConjugated()
    )

def is_rotatable(bond: Chem.Bond) -> List[float]:
    mol = bond.GetOwningMol()
    atom_indices = tuple(
        sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
    return encode(
        x=atom_indices in Lipinski._RotatableBonds(mol)
    )

def bondstereo(bond: Chem.Bond) -> List[float]:
    return onehot_encode(
        x=bond.GetStereo(),
        allowable_set=[
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOANY,
        ]
    )

def element(atom: Chem.Atom) -> List[float]:
    x = atom.GetSymbol()
    allowable_set = [
        'H', 'B', 'C', 'N', 'O', 'F', 'Si',
        'P', 'S', 'Cl', 'Br', 'I','other'
    ]
    symbol = atom.GetSymbol()
    if x not in allowable_set:
        x = 'other'
    return onehot_encode(x=x, allowable_set=allowable_set)

def hybridization(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetHybridization(),
        allowable_set=[
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]
    )

def cip_code(atom: Chem.Atom) -> List[float]:
    if atom.HasProp("_CIPCode"):
        return onehot_encode(
            x=atom.GetProp("_CIPCode"),
            allowable_set=[
                "R", "S"
            ]
        )
    return [0.0, 0.0]

def chiral_center(atom: Chem.Atom) -> List[float]:
    return encode(
        x=atom.HasProp("_ChiralityPossible")
    )

def formal_charge(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=min(max(atom.GetFormalCharge(), -1), 1),
        allowable_set=[-1, 0, 1]
    )

def mass(atom: Chem.Atom) -> List[float]:
    return encode(
        x=atom.GetMass() / 100
    )

def num_hs(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=min(atom.GetTotalNumHs(), 4),
        allowable_set=[0, 1, 2, 3, 4]
    )

def num_valence(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=min(atom.GetTotalValence(), 6),
        allowable_set=[0, 1, 2, 3, 4, 5, 6])

def degree(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=min(atom.GetDegree(), 5),
        allowable_set=[0, 1, 2, 3, 4, 5]
    )

def is_aromatic(atom: Chem.Atom) -> List[float]:
    return encode(
        x=atom.GetIsAromatic()
    )

def is_hetero(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)]
    )

def is_h_donor(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)]
    )

def is_h_acceptor(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)]
    )

def is_in_ring_size_n(atom: Chem.Atom) -> List[float]:
    for ring_size in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
        if atom.IsInRingSize(ring_size): break
    return onehot_encode(
        x=ring_size,
        allowable_set=[0, 3, 4, 5, 6, 7, 8, 9, 10]
    )

def num_radical_electrons(atom: Chem.Atom) -> List[float]:
    num_radical_electrons = atom.GetNumRadicalElectrons()
    return onehot_encode(
        x=min(num_radical_electrons, 2),
        allowable_set=[0, 1, 2]
    )

def crippen_log_p_contrib(atom, mol_feats):
    crippen, _, _, _ = mol_feats
    return encode(crippen[atom.GetIdx()][0])

def crippen_molar_refractivity_contrib(atom, mol_feats):
    crippen, _, _, _ = mol_feats
    return encode(crippen[atom.GetIdx()][1])

def tpsa_contrib(atom, mol_feats):
    _, tpsa, _, _ = mol_feats
    return encode(tpsa[atom.GetIdx()])

def labute_asa_contrib(atom, mol_feats):
    _, _, labute, _ = mol_feats
    return encode(labute[atom.GetIdx()])

def gasteiger_charge(atom, mol_feats):
    _, _, _, gasteiger = mol_feats
    return encode(gasteiger[atom.GetIdx()])


def get_edge_dim(exclude_feature=None):
    """Hacky way to get edge dim from bond_featurizer"""
    mol = Chem.MolFromSmiles('CC')
    if exclude_feature:
        edge_dim = len(bond_featurizer(mol.GetBonds()[0], exclude_feature))
    else:
        edge_dim = len(bond_featurizer(mol.GetBonds()[0], exclude_feature))

    return edge_dim

def get_node_dim(exclude_feature=None):
    """Hacky way to get node dim from atom_featurizer"""
    mol = Chem.MolFromSmiles('CC')
    if exclude_feature:
        node_dim = len(atom_featurizer(mol.GetAtoms()[0], exclude_feature))
    else:
        node_dim = len(atom_featurizer(mol.GetAtoms()[0], exclude_feature))
    return node_dim

def get_node_features(mol, exclude_feature=None):
    num_atoms = mol.GetNumAtoms()
    node_features = np.zeros((num_atoms, get_node_dim()), dtype=np.float32)

    for i, atom in enumerate(mol.GetAtoms()):
        node_features[i] = atom_featurizer(atom, mol_feats, exclude_feature)
    return node_features

def get_edge_features(mol, exclude_feature=None):
    num_edge = mol.GetNumBonds()
    edge_features = np.zeros((num_edge, get_edge_dim()), dtype=np.float32)

    for i, bond in enumerate(mol.GetBonds()):
        edge_features[i] =  bond_featurizer(bond, exclude_feature)

    return edge_features

def get_global_feature(mol,precursor_charge_onehot,energy):
    num_node = mol.GetNumAtoms()
    x_global = np.array([np.concatenate([precursor_charge_onehot,energy]) for n in range(num_node)])
    return x_global




def precompute_mol_features(mol):
    CrippenContribs = Crippen._GetAtomContribs(mol)
    TPSAContribs = rdMolDescriptors._CalcTPSAContribs(mol)
    LabuteASAContribs = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]

    rdPartialCharges.ComputeGasteigerCharges(mol)
    GasteigerCharges = [
        atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()
    ]

    return CrippenContribs, TPSAContribs, LabuteASAContribs, GasteigerCharges


# =========================
# GLOBALS for multiprocessing
# =========================
SEQ = None
INTY = None
CHARGE = None
ENERGY = None
LABEL_TYPE = None


# =========================
# Worker initializer
# =========================
def init_worker(seq, inty, charge, energy, label_type):
    global SEQ, INTY, CHARGE, ENERGY, LABEL_TYPE
    SEQ = seq
    INTY = inty
    CHARGE = charge
    ENERGY = energy
    LABEL_TYPE = label_type


# =========================
# Worker function
# =========================

def process_one(i):
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

    # ---- node features ----
    x_local = torch.from_numpy(get_node_features(mol)).float()
    x_global = torch.from_numpy(get_global_feature(mol, charge_ohe, energy)).float()
    x = torch.cat([x_local, x_global], dim=1)

    # ---- edges ----
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    if len(edges) == 0:
        return None

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.from_numpy(get_edge_features(mol)).float()

    edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    # ---- labels ----
    bonds_list = list(mol.GetBonds())

    # if LABEL_TYPE == "full":
    #     peptide_pattern = Chem.MolFromSmarts("C(=O)N[C]")
    #     matches = mol.GetSubstructMatches(peptide_pattern)
    #
    #     if charge == 0:
    #         bond_prob = [0, -1, -1, 0, -1, -1] * len(bonds_list)
    #     elif charge == 1:
    #         bond_prob = [0, 0, -1, 0, 0, -1] * len(bonds_list)
    #     else:
    #         bond_prob = [0] * 6 * len(bonds_list)
    #
    #     for i_bond, b in enumerate(bonds_list):
    #         for match in matches:
    #             c_idx, o_idx, n_idx, _ = match
    #             if b.GetBeginAtomIdx() == c_idx and b.GetEndAtomIdx() == n_idx:
    #                 bond_prob[6 * i_bond:6 * i_bond + 6] = inty[6 * matches.index(match):6 * matches.index(match) + 6]
    #
    # elif LABEL_TYPE == "scarce":
    bond_prob = inty


    y = torch.tensor(bond_prob, dtype=torch.float32).flatten()

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# =========================
# Batch processing with multiprocessing
# =========================
def process_batch(start, end, sequence, intensity, charge, energy, label_type):
    seq_batch = sequence[start:end]
    inty_batch = intensity[start:end]
    charge_batch = charge[start:end]
    energy_batch = energy[start:end]

    n_workers = min(cpu_count(), 40)

    with Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(seq_batch, inty_batch, charge_batch, energy_batch, label_type)
    ) as pool:
        results = pool.map(process_one, range(end - start))

    return [r for r in results if r is not None]


class SpectraGraphDatasetPrec(InMemoryDataset):
    def __init__(self, root, data_source, label_type="full",
                 transform=None, pre_transform=None):

        self.data_source = data_source
        self.label_type = label_type
        self.root = root
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.data_source)]

    @property
    def processed_file_names(self):
        return [self.root]

    def download(self):
        pass

    def process(self):

        data_list = []

        with h5py.File(self.data_source, "r") as f:
            intensity = f["intensities_raw"]
            sequence = f["sequence_integer"]
            precursor_charge_onehot = f["precursor_charge_onehot"]
            energy_list = f["collision_energy_aligned"]

            length = intensity.shape[0]
            batch_size = 512

            for start in range(0, length, batch_size):
                end = min(start + batch_size, length)

                print(f"Processing {start}-{end}/{length}")

                batch_data = process_batch(
                    start, end,
                    sequence,
                    intensity,
                    precursor_charge_onehot,
                    energy_list,
                    self.label_type
                )

                data_list.extend(batch_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])