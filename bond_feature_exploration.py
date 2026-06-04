import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

import data.graph_creation_utils as gcu
from feature_exploration import sequence_to_mol, write_feature_exploration_report


BOND_ONE_HOT_FEATURE_LABELS = {
    "bondstereo": ["none", "Z", "E", "any"],
    "bondtype": ["single", "double", "triple", "aromatic"],
}


def compute_bond_feature_dataframe(
    mol_seq_list: Iterable[str],
    *,
    include_bond_metadata: bool = True,
    skip_invalid: bool = False,
) -> pd.DataFrame:
    """Build one row per bond with each bond feature kept as its original vector."""
    rows = []

    for sequence_index, mol_seq in enumerate(mol_seq_list):
        try:
            mol = sequence_to_mol(mol_seq)
        except Exception:
            if skip_invalid:
                continue
            raise

        for bond in mol.GetBonds():
            row = {
                feature_name: getattr(gcu, feature_name)(bond)
                for feature_name in gcu.bond_features
            }

            if include_bond_metadata:
                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()
                row.update(
                    {
                        "sequence_index": sequence_index,
                        "sequence": mol_seq,
                        "bond_index": bond.GetIdx(),
                        "begin_atom_index": begin_atom.GetIdx(),
                        "end_atom_index": end_atom.GetIdx(),
                        "begin_atom_symbol": begin_atom.GetSymbol(),
                        "end_atom_symbol": end_atom.GetSymbol(),
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


def write_bond_feature_report(
    df: pd.DataFrame,
    output_path: str | Path = "bond_feature_report.html",
    *,
    bins: int = 30,
) -> Path:
    return write_feature_exploration_report(
        df,
        output_path,
        bins=bins,
        feature_names=gcu.bond_features,
        default_feature_labels=BOND_ONE_HOT_FEATURE_LABELS,
        observation_label="bonds",
        title="Bond Feature Exploration",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _load_sequences(csv_path: str | Path, sequence_column: str, limit: int) -> List[str]:
    input_df = pd.read_csv(csv_path)
    return input_df[sequence_column].dropna().astype(str).head(limit).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore bond feature distributions from peptide sequences.",
    )
    parser.add_argument(
        "--csv",
        default="dataset_dummy/proteomeTools_train_val.csv",
        help="CSV file containing peptide sequences.",
    )
    parser.add_argument(
        "--sequence-column",
        default="sequence",
        help="Name of the sequence column in the CSV file.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=1000,
        help="Maximum number of sequences to process.",
    )
    parser.add_argument(
        "--output",
        default="bond_feature_report.html",
        help="Output HTML report path.",
    )
    parser.add_argument(
        "--bins",
        type=_positive_int,
        default=30,
        help="Number of bins for scalar histograms.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip sequences that RDKit cannot parse.",
    )
    args = parser.parse_args()

    sequences = _load_sequences(args.csv, args.sequence_column, args.limit)
    feature_df = compute_bond_feature_dataframe(sequences, skip_invalid=args.skip_invalid)
    output_path = write_bond_feature_report(feature_df, args.output, bins=args.bins)
    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.scalar_summary.csv')}")
    print(f"Wrote {output_path.with_suffix('.class_summary.csv')}")


if __name__ == "__main__":
    main()
