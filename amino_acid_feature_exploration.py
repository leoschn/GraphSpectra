import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

import data.graph_creation_utils as gcu
from atom_feature_exploration import sequence_to_mol, write_feature_exploration_report


AA_ONE_HOT_FEATURE_LABELS = {
    "aromaticity": ["non-aromatic", "aromatic"],
}


def compute_amino_acid_feature_dataframe(
    mol_seq_list: Iterable[str],
    *,
    include_residue_metadata: bool = True,
    skip_invalid: bool = False,
) -> pd.DataFrame:
    """Build one row per peptide residue with each AA feature kept as its original vector."""
    rows = []

    for sequence_index, mol_seq in enumerate(mol_seq_list):
        try:
            mol = sequence_to_mol(mol_seq)
            residue_labels = _residue_labels(mol_seq)
            residues = sorted(gcu.split_peptide_by_residue(mol), key=lambda item: item[0])
        except Exception:
            if skip_invalid:
                continue
            raise

        for residue_index, (residue_number, residue_mol) in enumerate(residues):
            row = {
                feature_name: getattr(gcu, feature_name)(residue_mol)
                for feature_name in gcu.aa_features
            }

            if include_residue_metadata:
                residue_label = (
                    residue_labels[residue_index]
                    if residue_index < len(residue_labels)
                    else None
                )
                row.update(
                    {
                        "sequence_index": sequence_index,
                        "sequence": mol_seq,
                        "residue_index": residue_index,
                        "residue_number": residue_number,
                        "residue_label": residue_label,
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


def write_amino_acid_feature_report(
    df: pd.DataFrame,
    output_path: str | Path = "amino_acid_feature_report.html",
    *,
    bins: int = 30,
) -> Path:
    return write_feature_exploration_report(
        df,
        output_path,
        bins=bins,
        feature_names=gcu.aa_features,
        default_feature_labels=AA_ONE_HOT_FEATURE_LABELS,
        observation_label="residues",
        title="Amino Acid Feature Exploration",
    )


def _residue_labels(sequence: str) -> List[str]:
    labels = []
    index = 0
    while index < len(sequence):
        if sequence.startswith("M(ox)", index):
            labels.append("M(ox)")
            index += 5
            continue

        labels.append(sequence[index])
        index += 1

    return labels


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
        description="Explore amino-acid feature distributions from peptide sequences.",
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
        default="amino_acid_feature_report.html",
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
    feature_df = compute_amino_acid_feature_dataframe(
        sequences,
        skip_invalid=args.skip_invalid,
    )
    output_path = write_amino_acid_feature_report(feature_df, args.output, bins=args.bins)
    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.scalar_summary.csv')}")
    print(f"Wrote {output_path.with_suffix('.class_summary.csv')}")


if __name__ == "__main__":
    main()
