from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdPartialCharges

from collections import defaultdict

from data.pICalculax import find_pKas, pI

from typing import Union, List
import numpy as np
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

                # Add oxygen atom with the same residue metadata as sulfur so
                # residue-based splitting keeps the oxidation inside methionine.
                oxygen = Chem.Atom("O")
                oxygen.SetMonomerInfo(info)
                o_idx = mol.AddAtom(oxygen)

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
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol




#from deepGCN-RT
atom_features = [
    'chiral_center',# dim 1
    'cip_code', # dim 2
    'crippen_log_p_contrib', #dim 1
    'crippen_molar_refractivity_contrib', #dim 1
    'degree', #dim 6
    'element', #dim 6
    'hybridization', #dim 5
    'is_h_acceptor',#dim 1
    'is_h_donor',#dim 1
    'is_hetero',#dim 1
    'is_in_ring_size_n',#dim 3
    'labute_asa_contrib',#dim 1
    'mass',#dim 1
    'num_hs',#dim 4
    'num_valence',#dim 7
    'tpsa_contrib',#dim 1
]

bond_features = [
    'bondtype',#dim 4
    'is_conjugated',#dim 1
    'is_in_ring',#dim 1
    'is_rotatable',#dim 1
]


aa_features = ['log_p',#dim 1
               'mol_weight', #dim 1
               'aromaticity',#dim 2
               'isoelectric_point',#dim 1
               'num_atom',#dim 1
               'pka_values' #dim 3 + mask 3 => 6
]


'''adopted from: https://github.com/akensert/GCN-retention-time-predictions'''

def onehot_encode(x: Union[float, int, str],
                  allowable_set: List[Union[float, int, str]]) -> List[float]:
    return list(map(lambda s: float(x == s), allowable_set))

def encode(x: Union[float, int, str]) -> List[float]:
    if x is None or np.isnan(x):
        x = 0.0
    return [float(x)]

def _normalize_exclude_features(exclude_feature):
    if exclude_feature is None:
        return set()
    if isinstance(exclude_feature, str):
        return {exclude_feature}
    return set(exclude_feature)


def bond_featurizer(bond: Chem.Bond,exclude_feature) -> np.ndarray:
    excluded_features = _normalize_exclude_features(exclude_feature)
    new_bond_features = [i for i in bond_features if i not in excluded_features]
    return np.concatenate([
        globals()[bond_feature](bond) for bond_feature in new_bond_features
    ], axis=0)

def atom_featurizer(atom, mol_feats, exclude_feature, concat=True):
    excluded_features = _normalize_exclude_features(exclude_feature)
    new_atom_features = [i for i in atom_features if i not in excluded_features]

    features = []

    for atom_feature in new_atom_features:
        if atom_feature in [
            'crippen_log_p_contrib',
            'crippen_molar_refractivity_contrib',
            'tpsa_contrib',
            'labute_asa_contrib',
        ]:
            features.append(globals()[atom_feature](atom, mol_feats)) #molecule level prop
        else:
            features.append(globals()[atom_feature](atom)) #atome level prop
    if concat :
        return np.concatenate(features, axis=0)
    else:
        return features

def aa_featurizer(aa, exclude_feature):
    excluded_features = _normalize_exclude_features(exclude_feature)
    new_aa_features = [i for i in aa_features if i not in excluded_features]

    features = []

    for aa_feature in new_aa_features:
        features.append(globals()[aa_feature](aa)) #atome level prop

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


def element(atom: Chem.Atom) -> List[float]:
    x = atom.GetSymbol()
    allowable_set = [
        'C', 'N', 'O','P', 'S','other'
    ]
    symbol = atom.GetSymbol()
    if x not in allowable_set:
        x = 'other'
    return onehot_encode(x=x, allowable_set=allowable_set)

def hybridization(atom: Chem.Atom) -> List[float]:
    x = atom.GetHybridization()
    allowable_set = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        'other'
    ]
    if x not in allowable_set:
        x = 'other'

    return onehot_encode(
        x=x, allowable_set=allowable_set)

        #Chem.rdchem.HybridizationType.S removed (rare)
        # Chem.rdchem.HybridizationType.SP3D2, removed (rare)
        # other added


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
        x=atom.HasProp("_CIPCode")
    )

def mass(atom: Chem.Atom) -> List[float]:
    return encode(
        x=atom.GetMass() / 100
    )

def num_hs(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=min(atom.GetTotalNumHs(), 3),
        allowable_set=[0, 1, 2, 3]
    )

def num_valence(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=min(atom.GetTotalValence(), 6),
        allowable_set=[0, 1, 2, 3, 4, 5, 6])

def degree(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=min(atom.GetTotalDegree(), 5),
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
    for ring_size in [6, 5, 0]:
        if atom.IsInRingSize(ring_size): break
    return onehot_encode(
        x=max(0,ring_size),
        allowable_set=[0, 5, 6,]
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

def conformation(atom, mol_feats):
    _, _, _, conf = mol_feats
    if conf :
        pos = conf.GetAtomPosition(atom.GetIdx())
        return [pos.x, pos.y, pos.z]
    else:
        return [0., 0., 0.]

#custom aa features

def log_p(mol):
    return encode(
        x=Descriptors.MolLogP(mol)
    )

def mol_weight(mol):
    return encode(
        x=Descriptors.MolWt(mol)
    )

def aromaticity(mol):
    has_aromatic_atom = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
    return onehot_encode(
        x=has_aromatic_atom,
        allowable_set=[False, True]
    )

def pad_and_mask(values, max_len=3, pad_value=0.0):
    n = min(len(values), max_len)

    padded = values[:max_len] + [pad_value] * (max_len - n)
    mask = [1] * n + [0] * (max_len - n)

    return padded, mask

def pka_values(mol):
    pkalist, _ = find_pKas(mol)
    padded_list, mask = pad_and_mask(pkalist)
    return padded_list + mask

def isoelectric_point(mol):
    pkalist, charge = find_pKas(mol)
    return encode(
        x=pI(pkalist, charge)
    )

def num_atom(mol):
    return encode(
        x=mol.GetNumAtoms()
    )


def get_edge_dim(exclude_feature=None):
    """Hacky way to get edge dim from bond_featurizer"""
    mol = Chem.MolFromSmiles('CC')
    edge_dim = len(bond_featurizer(mol.GetBonds()[0], exclude_feature))

    return edge_dim

def precompute_mol_features(mol,exclude_feature=None):
    excluded_features = _normalize_exclude_features(exclude_feature)
    CrippenContribs = None
    TPSAContribs = None
    LabuteASAContribs = None
    conf = None

    if (
        'crippen_log_p_contrib' not in excluded_features
        or 'crippen_molar_refractivity_contrib' not in excluded_features
    ):
        CrippenContribs = Crippen._GetAtomContribs(mol)

    if 'tpsa_contrib' not in excluded_features:
        TPSAContribs = rdMolDescriptors._CalcTPSAContribs(mol)

    if 'labute_asa_contrib' not in excluded_features:
        LabuteASAContribs = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]


    if 'conformation' not in excluded_features:
        #conformation
        mol_h = Chem.AddHs(mol)

        # 3. Generate 3D coordinates using ETKDGv3

        ret = AllChem.EmbedMolecule(mol_h,randomSeed = 42, useRandomCoords=True)

        if ret == 0:
            if AllChem.MMFFHasAllMoleculeParams(mol_h):
                AllChem.MMFFOptimizeMolecule(mol_h)

            mol_no_h = Chem.RemoveHs(mol_h)
            conf = mol_no_h.GetConformer()
        else:
            print('Failed conformation optimization')
            conf = None

    return CrippenContribs, TPSAContribs, LabuteASAContribs, conf



def get_node_dim(exclude_feature=None):
    """Hacky way to get node dim from atom_featurizer"""
    mol = Chem.MolFromSmiles('CC')
    mol_feats = precompute_mol_features(mol, exclude_feature)
    node_dim = len(atom_featurizer(mol.GetAtoms()[0], mol_feats, exclude_feature))
    return node_dim

def get_node_aa_dim(exclude_feature=None):
    mol = Chem.MolFromFASTA('A')
    node_dim = len(aa_featurizer(mol, exclude_feature))
    return node_dim

NODE_DIM = get_node_dim()
EDGE_DIM = get_edge_dim()
NODE_AA_DIM = get_node_aa_dim()



def get_node_features(mol, exclude_feature=None, mol_feats=None):
    if mol_feats is None:
        mol_feats = precompute_mol_features(mol, exclude_feature)

    node_features = [
        atom_featurizer(atom, mol_feats, exclude_feature)
        for atom in mol.GetAtoms()
    ]

    if len(node_features) == 0:
        return np.empty((0, get_node_dim(exclude_feature)), dtype=np.float32)

    return np.asarray(node_features, dtype=np.float32)

def get_edge_features(mol, exclude_feature=None):
    num_edge = mol.GetNumBonds()
    edge_features = np.zeros((num_edge, EDGE_DIM), dtype=np.float32)

    for i, bond in enumerate(mol.GetBonds()):
        edge_features[i] =  bond_featurizer(bond, exclude_feature)

    return edge_features

def get_global_feature(mol,precursor_charge_onehot,energy):
    num_node = mol.GetNumAtoms()
    x_global = np.array([np.concatenate([precursor_charge_onehot,energy]) for n in range(num_node)])
    return x_global

def _atom_residue_number(atom):
    info = atom.GetMonomerInfo()
    if info is None:
        return None
    return info.GetResidueNumber()


def _has_carbonyl_oxygen(atom, selected_atoms):
    for bond in atom.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue

        other = bond.GetOtherAtom(atom)
        if other.GetIdx() in selected_atoms and other.GetSymbol() == "O":
            return True

    return False


def _is_broken_peptide_carboxyl(atom, outside_atom, selected_atoms):
    if atom.GetSymbol() != "C" or outside_atom.GetSymbol() != "N":
        return False
    return _has_carbonyl_oxygen(atom, selected_atoms)


def _build_capped_residue_mol(mol, atom_indices):
    selected_atoms = set(atom_indices)
    editable_mol = Chem.RWMol()
    old_to_new = {}

    for atom_idx in atom_indices:
        atom = Chem.Atom(mol.GetAtomWithIdx(atom_idx))
        old_to_new[atom_idx] = editable_mol.AddAtom(atom)

    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in selected_atoms and end_idx in selected_atoms:
            editable_mol.AddBond(
                old_to_new[begin_idx],
                old_to_new[end_idx],
                bond.GetBondType(),
            )

    added_carboxyl_caps = set()
    for atom_idx in atom_indices:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx in selected_atoms:
                continue
            if atom_idx in added_carboxyl_caps:
                continue
            if not _is_broken_peptide_carboxyl(atom, neighbor, selected_atoms):
                continue

            oxygen = Chem.Atom("O")
            info = atom.GetMonomerInfo()
            if info is not None:
                oxygen.SetMonomerInfo(info)
            oxygen_idx = editable_mol.AddAtom(oxygen)
            editable_mol.AddBond(old_to_new[atom_idx], oxygen_idx, Chem.BondType.SINGLE)
            added_carboxyl_caps.add(atom_idx)

    residue_mol = editable_mol.GetMol()
    Chem.SanitizeMol(residue_mol)
    Chem.AssignStereochemistry(residue_mol, cleanIt=True, force=True)
    return residue_mol


def split_peptide_by_residue(mol):
    # Group atom indices by residue number, including PTM atoms that carry the
    # same residue metadata as the modified residue.
    residue_atoms = defaultdict(list)

    for atom in mol.GetAtoms():
        residue_number = _atom_residue_number(atom)
        if residue_number is None:
            continue
        residue_atoms[residue_number].append(atom.GetIdx())

    residue_mols = []

    for residue_number, atom_indices in sorted(residue_atoms.items()):
        residue_mols.append((residue_number, _build_capped_residue_mol(mol, atom_indices)))

    return residue_mols

def get_aa_node_features(mol, exclude_feature=None):
    #split mol into aa
    aa_list = split_peptide_by_residue(mol)
    num_aa = len(aa_list)
    node_features = np.zeros((num_aa, NODE_AA_DIM), dtype=np.float32)
    for i, aa in enumerate(aa_list):
        node_features[i] = aa_featurizer(aa[1], exclude_feature)
    return node_features


def _conformer_from_mol_features(mol_feats):
    if mol_feats is None:
        return None
    _, _, _, conf = mol_feats
    return conf


def _conformer_position(conf, atom_idx):
    pos = conf.GetAtomPosition(atom_idx)
    return [pos.x, pos.y, pos.z]


def get_atom_positions(mol, mol_feats=None):
    if mol_feats is None:
        mol_feats = precompute_mol_features(
            mol,
            exclude_feature=[
                'crippen_log_p_contrib',
                'crippen_molar_refractivity_contrib',
                'tpsa_contrib',
                'labute_asa_contrib',
            ],
        )

    conf = _conformer_from_mol_features(mol_feats)
    atom_positions = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float32)

    if conf is None:
        return atom_positions

    for atom in mol.GetAtoms():
        atom_positions[atom.GetIdx()] = _conformer_position(conf, atom.GetIdx())

    return atom_positions


def _find_residue_carboxyl_carbon_idx(mol, residue_number):
    for atom in mol.GetAtoms():
        info = atom.GetMonomerInfo()
        if info is None:
            continue
        if info.GetResidueNumber() != residue_number:
            continue
        if atom.GetSymbol() == "C" and info.GetName().strip() == "C":
            return atom.GetIdx()

    selected_atoms = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if _atom_residue_number(atom) == residue_number
    }

    for atom_idx in selected_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == "C" and _has_carbonyl_oxygen(atom, selected_atoms):
            return atom_idx

    return None


def get_aa_node_positions(mol, mol_feats=None):
    aa_list = split_peptide_by_residue(mol)
    aa_positions = np.zeros((len(aa_list), 3), dtype=np.float32)
    conf = _conformer_from_mol_features(mol_feats)

    if conf is None:
        return aa_positions

    for i, (residue_number, _) in enumerate(aa_list):
        atom_idx = _find_residue_carboxyl_carbon_idx(mol, residue_number)
        if atom_idx is not None:
            aa_positions[i] = _conformer_position(conf, atom_idx)

    return aa_positions
