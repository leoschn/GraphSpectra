from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdPartialCharges

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


aa_features = ['log_p',#dim 1
               'mol_weight', #dim 1
               'aromaticity',#dim 2
               'isoelectric_point',#dim 1
               'num_atom'#dim 1
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

def aa_featurizer(aa, exclude_feature):
    new_aa_features = [i for i in aa_features if i != exclude_feature]

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

def isoelectric_point(mol):
    pkalist, charge = find_pKas(mol)
    return encode(
        x=pI(pkalist, charge)
    )

def isoelectic_point(mol):
    return isoelectric_point(mol)

def num_atom(mol):
    return encode(
        x=mol.GetNumAtoms()
    )



def get_edge_dim(exclude_feature=None):
    """Hacky way to get edge dim from bond_featurizer"""
    mol = Chem.MolFromSmiles('CC')
    edge_dim = len(bond_featurizer(mol.GetBonds()[0], exclude_feature))

    return edge_dim

def precompute_mol_features(mol):
    CrippenContribs = Crippen._GetAtomContribs(mol)
    TPSAContribs = rdMolDescriptors._CalcTPSAContribs(mol)
    LabuteASAContribs = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]

    rdPartialCharges.ComputeGasteigerCharges(mol)
    GasteigerCharges = [
        atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()
    ]

    return CrippenContribs, TPSAContribs, LabuteASAContribs, GasteigerCharges



def get_node_dim(exclude_feature=None):
    """Hacky way to get node dim from atom_featurizer"""
    mol = Chem.MolFromSmiles('CC')
    mol_feats = precompute_mol_features(mol)
    node_dim = len(atom_featurizer(mol.GetAtoms()[0], mol_feats, exclude_feature))
    return node_dim

def get_node_aa_dim(exclude_feature=None):
    mol = Chem.MolFromFASTA('A')
    node_dim = len(aa_featurizer(mol, exclude_feature))
    return node_dim

NODE_DIM = get_node_dim()
EDGE_DIM = get_edge_dim()
NODE_AA_DIM = get_node_aa_dim()



def get_node_features(mol, exclude_feature=None):
    num_atoms = mol.GetNumAtoms()
    node_features = np.zeros((num_atoms, NODE_DIM), dtype=np.float32)
    mol_feats = precompute_mol_features(mol)
    for i, atom in enumerate(mol.GetAtoms()):
        node_features[i] = atom_featurizer(atom, mol_feats, exclude_feature)
    return node_features

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

def split_peptide_by_residue(mol):
    # Group atom indices by residue number
    residue_atoms = defaultdict(list)

    for atom in mol.GetAtoms():
        info = atom.GetMonomerInfo()
        residue_number = info.GetResidueNumber()
        residue_atoms[residue_number].append(atom.GetIdx())

    residue_mols = []

    for residue_number, atom_indices in residue_atoms.items():
        # Create submolecule containing only residue atoms
        submol = Chem.PathToSubmol(mol, atom_indices)

        residue_mols.append((residue_number, submol))

    return residue_mols

def get_aa_node_features(mol, exclude_feature=None):
    #split mol into aa
    aa_list = split_peptide_by_residue(mol)
    num_atoms = mol.GetNumAtoms()
