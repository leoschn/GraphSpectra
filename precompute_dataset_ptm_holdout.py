"""
Dataset 2: PTM-generalization split -- 15 "seen" PTM types vs. 6
"unseen" PTM types.

Goal: measure how well the model generalizes to modification types it
never saw during training/finetuning, as opposed to dataset 1's
in-distribution 80/10/10 split.

- 15 PTM types ("seen") are pooled and split 80/10/10 by underlying
  sequence (`sequence_no_mod`), exactly like dataset 1, so that
  "seen_val"/"seen_test" measure ordinary training convergence /
  in-domain performance.
- 6 PTM types ("unseen") are held out ENTIRELY -- every spectrum from
  those types goes into a single `unseen_test` split, never used for
  training. This measures generalization to novel PTMs.

UNSEEN_PTM_TYPES below was chosen once with a fixed seed (42) for
reproducibility -- edit the list directly if you want to hold out
specific modifications instead (e.g. grouped by chemistry class).

This uses the exact same 21 PTM-type spectra as dataset 1
(`precompute_dataset_ptms.py`) -- just partitioned differently -- so it
reads/writes the SAME shared feature store
(`ptm_pipeline_common.ALL_PTMS_STORE_DIR`). Whichever of the two dataset
scripts runs first pays for the graph-feature computation; the other
reuses it via `row_id`, with zero recomputation.

This script only PREPARES the run -- see `if __name__ == "__main__"` at
the bottom to actually launch it.
"""

import os
import random

import pandas as pd

import ptm_pipeline_common as pc


# ============================================================
# WHICH 6 PTM TYPES ARE HELD OUT AS "UNSEEN"
# ============================================================
# Reproducible default: random.Random(42).sample(all 21 folders, 6).
# Edit this list directly to hold out specific PTMs instead.
_ALL_FOLDERS = [entry["folder"] for entry in pc.PTM_SOURCES]
UNSEEN_PTM_TYPES = sorted(random.Random(42).sample(_ALL_FOLDERS, 6))
SEEN_PTM_TYPES = sorted(set(_ALL_FOLDERS) - set(UNSEEN_PTM_TYPES))


# ============================================================
# OUTPUT LAYOUT
# ============================================================
SPLIT_CSV_DIR = os.path.join(pc.OUTPUT_ROOT, "csv", "dataset2_ptm_holdout_splits")
DATASET_ROOT = os.path.join(pc.OUTPUT_ROOT, "datasets", "dataset2_ptm_holdout")


def build_dataset2(force_rebuild_csv=False):
    print("CHUNK_SIZE:", pc.CHUNK_SIZE)
    print("BATCH_SIZE:", pc.BATCH_SIZE)
    print(f"\nSEEN PTM types ({len(SEEN_PTM_TYPES)}):")
    for f in SEEN_PTM_TYPES:
        print("  ", f)
    print(f"\nUNSEEN PTM types ({len(UNSEEN_PTM_TYPES)}):")
    for f in UNSEEN_PTM_TYPES:
        print("  ", f)

    # ------------------------------------------------------------
    # 1. Reuse (or build) the same combined, row_id-tagged CSV that
    #    dataset 1 uses -- these are the exact same 21 PTM-type spectra.
    # ------------------------------------------------------------
    if os.path.exists(pc.ALL_PTMS_CSV) and not force_rebuild_csv:
        print(f"\n[skip] {pc.ALL_PTMS_CSV} already exists, reusing it")
        df_complete = pd.read_csv(pc.ALL_PTMS_CSV)
    else:
        csv_paths = pc.build_all_sources(
            pc.PTM_SOURCES,
            pc.PRIDE_ROOT,
            os.path.join(pc.OUTPUT_ROOT, "csv", "per_ptm_type"),
            pc.build_ptm_type_csv,
            force=force_rebuild_csv,
        )
        df_complete = pc.combine_and_tag(csv_paths, pc.ALL_PTMS_CSV)

    assert set(df_complete["ptm_type"].unique()) == set(_ALL_FOLDERS), (
        "df_21_ptms.csv does not contain exactly the 21 expected PTM "
        "types -- was it built from a different PTM_SOURCES list?"
    )

    # ------------------------------------------------------------
    # 2. Partition by PTM type first.
    # ------------------------------------------------------------
    seen_df = df_complete[df_complete["ptm_type"].isin(SEEN_PTM_TYPES)].copy()
    unseen_test_df = df_complete[df_complete["ptm_type"].isin(UNSEEN_PTM_TYPES)].copy()

    print(f"\nSeen pool: {len(seen_df):,} spectra")
    print(f"Unseen (held-out) pool: {len(unseen_test_df):,} spectra")

    # ------------------------------------------------------------
    # 3. Split the "seen" pool 80/10/10 by underlying sequence, so
    #    seen_val/seen_test track ordinary training convergence.
    # ------------------------------------------------------------
    seen_train_df, seen_val_df, seen_test_df = pc.split_dataframe_by_sequence(
        seen_df,
        group_col="sequence_no_mod",
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        seed=42,
    )

    splits = {
        "seen_train": seen_train_df,
        "seen_val": seen_val_df,
        "seen_test": seen_test_df,
        "unseen_test": unseen_test_df,
    }

    # sequence_no_mod leakage within the seen pool...
    pc.verify_disjoint(
        {"seen_train": seen_train_df, "seen_val": seen_val_df, "seen_test": seen_test_df},
        group_col="sequence_no_mod",
    )
    # ...and PTM-type leakage between seen and unseen.
    pc.verify_disjoint(
        {"seen": seen_df, "unseen_test": unseen_test_df},
        group_col="ptm_type",
    )

    os.makedirs(SPLIT_CSV_DIR, exist_ok=True)
    for name, split_df in splits.items():
        split_df.to_csv(os.path.join(SPLIT_CSV_DIR, f"{name}.csv"), index=False)
    print(f"\nCSV splits saved under: {SPLIT_CSV_DIR}")

    # ------------------------------------------------------------
    # 4. Compute graph features ONCE for every spectrum into the SAME
    #    shared feature store dataset 1 uses -- rows already featurized
    #    by dataset 1's run (or a previous run of this script) are
    #    skipped automatically.
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FEATURE STORE: all 21 PTMs (shared with dataset 1)")
    print("=" * 60)

    pc.compute_feature_store(
        df_complete,
        seq_col="sequence",
        store_dir=pc.ALL_PTMS_STORE_DIR,
        with_position=False,
    )

    # ------------------------------------------------------------
    # 5. Assemble the final split directories by gathering (not
    #    recomputing) from the feature store.
    # ------------------------------------------------------------
    for name, split_df in splits.items():
        print(f"\n{'=' * 60}\nDATASET 2 - {name.upper()}\n{'=' * 60}")
        pc.gather_rows_into_dataset(
            row_ids=split_df["row_id"].tolist(),
            store_dir=pc.ALL_PTMS_STORE_DIR,
            out_dir=os.path.join(DATASET_ROOT, name),
        )

    print("\n" + "=" * 60)
    print("DATASET 2 (PTM holdout / generalization) CREATED SUCCESSFULLY")
    print(f"-> {DATASET_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    build_dataset2()
