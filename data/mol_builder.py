import rdkit.Chem
from rdkit import Chem
import random
# ---------------------------------------------------------------------
# Residue library
# Each residue has:
#   [*:1] = N-terminal attachment
#   [*:2] = C-terminal attachment
#
# PTMs are simply additional residue entries.
# ---------------------------------------------------------------------

RESIDUES = {
    # Standard amino acids
    "G": "[*:1]NCC(=O)[*:2]",
    "A": "[*:1]N[C@@H](C)C(=O)[*:2]",
    "S": "[*:1]N[C@@H](CO)C(=O)[*:2]",
    "T": "[*:1]N[C@@H]([C@H](O)C)C(=O)[*:2]",
    "C": "[*:1]N[C@@H](CS)C(=O)[*:2]",
    "V": "[*:1]N[C@@H](C(C)C)C(=O)[*:2]",
    "L": "[*:1]N[C@@H](CC(C)C)C(=O)[*:2]",
    "I": "[*:1]N[C@@H]([C@@H](CC)C)C(=O)[*:2]",
    "M": "[*:1]N[C@@H](CCSC)C(=O)[*:2]",
    "F": "[*:1]N[C@@H](Cc1ccccc1)C(=O)[*:2]",
    "Y": "[*:1]N[C@@H](Cc1ccc(O)cc1)C(=O)[*:2]",
    "W": "[*:1]N[C@@H](Cc1c[nH]c2ccccc12)C(=O)[*:2]",
    "D": "[*:1]N[C@@H](CC(=O)O)C(=O)[*:2]",
    "E": "[*:1]N[C@@H](CCC(=O)O)C(=O)[*:2]",
    "N": "[*:1]N[C@@H](CC(=O)N)C(=O)[*:2]",
    "Q": "[*:1]N[C@@H](CCC(=O)N)C(=O)[*:2]",
    "H": "[*:1]N[C@@H](Cc1c[nH]cn1)C(=O)[*:2]",
    "K": "[*:1]N[C@@H](CCCCN)C(=O)[*:2]",
    "R": "[*:1]N[C@@H](CCCNC(N)=N)C(=O)[*:2]",
    "P": "[*:1]N1CCC[C@H]1C(=O)[*:2]",

    # ---------------------- PTMs ----------------------

    # ---------------------- Lysine PTMs ----------------------

    # Acetylation
    "AcK": "[*:1]N[C@@H](CCCCNC(=O)C)C(=O)[*:2]",

    # Biotinylation
    "BiotinK": "[*:1]N[C@@H](CCCCNC(=O)CCCC[C@@H]1SC[C@]2([H])NC(=O)N[C@]12[H])C(=O)[*:2]",


    # Butyrylation
    "ButK": "[*:1]N[C@@H](CCCCNC(=O)CCC)C(=O)[*:2]",

    # Crotonylation
    "CroK": "[*:1]N[C@@H](CCCCNC(=O)/C=C/C)C(=O)[*:2]",

    # Monomethylation
    "MeK": "[*:1]N[C@@H](CCCCNC)C(=O)O[*:2]",

    # Dimethylation
    "Me2K": "[*:1]N[C@@H](CCCCN(C)C)C(=O)[*:2]",

    # Trimethylation
    "Me3K": "[*:1]N[C@@H](CCCC[N+](C)(C)C)C(=O)[*:2]",

    # Formylation
    "FormylK": "[*:1]N[C@@H](CCCCNC([H])=O)C(=O)[*:2]",


    # Glutarylation
    "GlutarylK": "[*:1]N[C@@H](CCCCNC(=O)CCCC(=O)O)C(=O)[*:2]",

    # Hydroxyisobutyrylation
    "HibK": "[*:1]N[C@@H](CCCCNC(=O)C(C)(C)O)C(=O)[*:2]",

    # Malonylation
    "MalK": "[*:1]N[C@@H](CCCCNC(=O)CC(=O)O)C(=O)[*:2]",

    # Propionylation
    "PropK": "[*:1]N[C@@H](CCCCNC(=O)CC)C(=O)[*:2]",

    # Succinylation
    "SucK": "[*:1]N[C@@H](CCCCNC(=O)CCC(=O)O)C(=O)[*:2]",

    # # GlyGly remnant (ubiquitin)
    # "GGK": "[*:1]N[C@@H](CCCCNC(=O)CNC(=O)CN)C(=O)[*:2]",


    # ---------------------- Arginine PTMs ----------------------

    # Citrullination
    "CitR": "[*:1]N[C@@H](CCCNC(N)=O)C(=O)[*:2]",

    # Monomethylation
    "MeR": "[*:1]N[C@@H](CCCNC(=N)NC)C(=O)[*:2]",


    # Asymmetric dimethylation (ADMA)
    "Me2aR": "[*:1]N[C@@H](CCCNC(=N)N(C)C)C(=O)[*:2]",

    # Symmetric dimethylation (SDMA)
    "Me2sR": "[*:1]N[C@@H](CCCNC(=NC)NC)C(=O)[*:2]",


    # ---------------------- Proline PTMs ----------------------

    # trans-4-Hydroxyproline
    "Hyp": "[*:1]N1CC[C@H](O)[C@H]1C(=O)[*:2]",


    # ---------------------- Tyrosine PTMs ----------------------

    # Nitration
    "NO2Y": "[*:1]N[C@@H](Cc1ccc(O)c([N+](=O)[O-])c1)C(=O)[*:2]",

    # Phosphorylation
    "pY": "[*:1]N[C@@H](Cc1ccc(OP(=O)(O)O)cc1)C(=O)[*:2]",
    }


def _dummy_atoms(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 0]


def _connect(mol1, mol2):
    combo = Chem.CombineMols(mol1, mol2)
    rw = Chem.RWMol(combo)

    dummies = _dummy_atoms(rw)

    # mol1
    left1, right1 = dummies[0], dummies[1]

    # mol2
    left2, right2 = dummies[2], dummies[3]

    c_atom = rw.GetAtomWithIdx(right1).GetNeighbors()[0].GetIdx()
    n_atom = rw.GetAtomWithIdx(left2).GetNeighbors()[0].GetIdx()

    rw.AddBond(c_atom, n_atom, Chem.BondType.SINGLE)

    # remove dummy atoms (highest index first)
    for idx in sorted([right1, left2], reverse=True):
        rw.RemoveAtom(idx)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)

    return mol


def build_peptide(sequence):
    """
    sequence : list of residue names
    """

    mol = Chem.MolFromSmiles(RESIDUES[sequence[0]])

    for res in sequence[1:]:
        mol = _connect(mol, Chem.MolFromSmiles(RESIDUES[res]))

    rw = Chem.RWMol(mol)

    # Remaining dummies: one N-terminus (*:1) and one C-terminus (*:2)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue

        amap = atom.GetAtomMapNum()
        dummy_idx = atom.GetIdx()
        neighbor_idx = atom.GetNeighbors()[0].GetIdx()

        if amap == 2:
            # Replace the C-terminal dummy with an OH group
            o = Chem.Atom("O")
            o_idx = rw.AddAtom(o)
            rw.AddBond(neighbor_idx, o_idx, Chem.BondType.SINGLE)

    # Remove all remaining dummy atoms
    for idx in sorted(_dummy_atoms(rw), reverse=True):
        rw.RemoveAtom(idx)

    mol = rw.GetMol()
    Chem.SanitizeMol(mol)

    return mol

# ---------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------
STANDARD_AA = [
    "A", "R", "N", "D", "C",
    "E", "Q", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V"
]
for i in range(20):
    seq = ''
    seq2 = []
    for _ in range(5):
        aa_to_add = random.choice(STANDARD_AA)
        seq+=aa_to_add
        seq2.extend(aa_to_add)
    mol = build_peptide(seq2)
    mol2 = rdkit.Chem.MolFromFASTA(seq)
    print(seq)
    print('built',Chem.MolToSmiles(mol))
    print('true ',Chem.MolToSmiles(mol2))
    print('-------------------------------------')
    assert mol.HasSubstructMatch(mol2)
    assert mol2.HasSubstructMatch(mol)