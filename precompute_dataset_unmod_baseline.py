"""
Dataset 3: unmodified-peptide baseline.

Built from the 4 "no variable PTM searched" MaxQuant control folders
(SEARCH_Kmod_Unmod, SEARCH_Rmod_Unmod, SEARCH_Ymod_Unmod,
SEARCH_Pmod_Unmod) -- the same synthetic peptide library as the 21 PTM
searches, run without a residue-specific modification in the search
space. This gives a baseline, PTM-free dataset to compare against
public (e.g. Prosit-style) training data.

This is disjoint raw data from the 21 PTM searches (different raw
files), so it gets its own feature store
(`ptm_pipeline_common.UNMOD_STORE_DIR`) -- nothing to share/reuse with
datasets 1/2 here, but features are still only computed once per
spectrum (resumable feature store, same as the other two scripts).

Split 80/10/10 by sequence, same leakage-safe grouping as datasets 1/2.

This script only PREPARES the run -- see `if __name__ == "__main__"` at
the bottom to actually launch it.
"""

import os

import pandas as pd

import ptm_pipeline_common as pc


# ============================================================
# OUTPUT LAYOUT
# ============================================================
SPLIT_CSV_DIR = os.path.join(pc.OUTPUT_ROOT, "csv", "dataset3_unmod_baseline_splits")
DATASET_ROOT = os.path.join(pc.OUTPUT_ROOT, "datasets", "dataset3_unmod_baseline")


def build_dataset3(force_rebuild_csv=False):
    print("CHUNK_SIZE:", pc.CHUNK_SIZE)
    print("BATCH_SIZE:", pc.BATCH_SIZE)

    # ------------------------------------------------------------
    # 1. Raw "unmodified" MaxQuant searches -> per-source CSVs ->
    #    combined, row_id-tagged df_unmod_baseline.csv.
    # ------------------------------------------------------------
    if os.path.exists(pc.UNMOD_CSV) and not force_rebuild_csv:
        print(f"[skip] {pc.UNMOD_CSV} already exists, reusing it")
        df_complete = pd.read_csv(pc.UNMOD_CSV)
    else:
        csv_paths = pc.build_all_sources(
            pc.UNMOD_SOURCES,
            pc.PRIDE_ROOT,
            os.path.join(pc.OUTPUT_ROOT, "csv", "per_unmod_source"),
            pc.build_unmod_csv,
            force=force_rebuild_csv,
        )
        df_complete = pc.combine_and_tag(csv_paths, pc.UNMOD_CSV)

    print(f"Total spectra: {len(df_complete):,}")
    print(f"Unique underlying sequences: {df_complete['sequence_no_mod'].nunique():,}")

    # ------------------------------------------------------------
    # 2. 80/10/10 split by underlying sequence.
    # ------------------------------------------------------------
    train_df, val_df, test_df = pc.split_dataframe_by_sequence(
        df_complete,
        group_col="sequence_no_mod",
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        seed=42,
    )

    pc.verify_disjoint(
        {"train": train_df, "val": val_df, "test": test_df},
        group_col="sequence_no_mod",
    )

    os.makedirs(SPLIT_CSV_DIR, exist_ok=True)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, split_df in splits.items():
        split_df.to_csv(os.path.join(SPLIT_CSV_DIR, f"{name}.csv"), index=False)
    print(f"\nCSV splits saved under: {SPLIT_CSV_DIR}")

    # ------------------------------------------------------------
    # 3. Compute graph features ONCE per spectrum into this dataset's
    #    own feature store (disjoint raw data from datasets 1/2, so
    #    nothing to reuse there -- but still resumable/idempotent).
    #    seq_col="sequence" here is equivalent to "sequence_no_mod"
    #    for genuinely unmodified spectra; "sequence" is used so a
    #    residual M(ox), if any, is still represented.
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FEATURE STORE: unmodified baseline")
    print("=" * 60)

    pc.compute_feature_store(
        df_complete,
        seq_col="sequence",
        store_dir=pc.UNMOD_STORE_DIR,
        with_position=False,
    )

    # ------------------------------------------------------------
    # 4. Assemble the final train/val/test chunk directories.
    # ------------------------------------------------------------
    for name, split_df in splits.items():
        print(f"\n{'=' * 60}\nDATASET 3 - {name.upper()}\n{'=' * 60}")
        pc.gather_rows_into_dataset(
            row_ids=split_df["row_id"].tolist(),
            store_dir=pc.UNMOD_STORE_DIR,
            out_dir=os.path.join(DATASET_ROOT, name),
        )

    print("\n" + "=" * 60)
    print("DATASET 3 (unmodified baseline) CREATED SUCCESSFULLY")
    print(f"-> {DATASET_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    build_dataset3()
