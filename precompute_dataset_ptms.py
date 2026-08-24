import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

from data.hierarchical_aa_ptm_dataset import process_sequence_batch
from data.sequence_preprocessing import convert_msms_to_prosit, annotate_msms_with_acquisition

# =========================
# CONFIG
# =========================
CHUNK_SIZE = 2048
BATCH_SIZE = 2048

# Column names in the CSV. Adjust here if your CSV uses different headers.
SEQUENCE_COL = "sequence"
INTENSITIES_COL = "intensities"
CHARGE_COL = "precursor_charge_onehot"
ENERGY_COL = "collision_energy"


# =========================
# SAVE CHUNK + METADATA
# =========================
def save_chunk(buffer, out_dir, chunk_id):
    path = os.path.join(out_dir, f"chunk_{chunk_id}.pt")
    torch.save(buffer, path)

    meta_path = os.path.join(out_dir, "meta.txt")
    with open(meta_path, "a") as f:
        f.write(f"{path},{len(buffer)}\n")


# =========================
# CSV FIELD PARSING
# =========================
def parse_stringified_list(value):
    """Parse a column value like "[0.21, -1.0, ...]" (stored as a string
    in the CSV) into a numpy array."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=np.float32)


def load_sequences(df):
    return [
        str(sequence).strip().strip("_")
        for sequence in df[SEQUENCE_COL].tolist()
    ]


def load_intensities(df):
    return np.stack(
        [parse_stringified_list(v) for v in df[INTENSITIES_COL].tolist()]
    )


def load_charge(df):
    return np.stack(
        [parse_stringified_list(v) for v in df[CHARGE_COL].tolist()]
    )


def load_energy(df):
    return df[ENERGY_COL].to_numpy(dtype=np.float32)


def count_csv_rows(csv_path):
    # Total data rows = total lines - 1 header line.
    with open(csv_path, "r") as f:
        total_lines = sum(1 for _ in f)
    return total_lines - 1


# =========================
# MAIN PREPROCESSING LOOP
# =========================
def preprocess_ptm_hierarchical_to_chunks(
    data_source,
    out_dir,
    with_position=True,
):
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, "meta.txt")

    # ---------- RESUME ----------
    if os.path.exists(meta_path) and os.path.getsize(meta_path) > 0:
        with open(meta_path, "r") as f:
            lines = f.readlines()

        chunk_id = len(lines)
        processed = sum(
            int(line.strip().split(",")[1])
            for line in lines
        )

        print(f"Resuming from chunk {chunk_id}")
        print(f"Already processed {processed} graphs")
    else:
        open(meta_path, "w").close()
        chunk_id = 0
        processed = 0
        print("Starting from scratch")

    buffer = []

    length = count_csv_rows(data_source)
    print(f"Using CSV source: {data_source} ({length} rows)")

    # Skip the header row (row 0) plus the data rows already processed
    # (rows 1..processed), then stream the remainder in BATCH_SIZE chunks.
    skiprows = range(1, processed + 1)
    reader = pd.read_csv(
        data_source,
        skiprows=skiprows,
        chunksize=BATCH_SIZE,
    )

    start = processed
    for df_batch in reader:
        end = min(start + len(df_batch), length)
        print(f"Processing {start}-{end}/{length}")

        batch_data = process_sequence_batch(
            load_sequences(df_batch),
            labels=load_intensities(df_batch),
            charge=load_charge(df_batch),
            energy=load_energy(df_batch),
            with_position=with_position,
        )

        for data in batch_data:
            buffer.append(data)

            if len(buffer) >= CHUNK_SIZE:
                save_chunk(buffer, out_dir, chunk_id)
                buffer = []
                chunk_id += 1

        start = end

    if buffer:
        save_chunk(buffer, out_dir, chunk_id)

    print("Preprocessing DONE")





if __name__ == "__main__":
    print("CHUNK_SIZE:", CHUNK_SIZE)
    print("BATCH_SIZE:", BATCH_SIZE)
    raw_msms_path_list = ['SEARCH_Kmod_Formyl',
                          'SEARCH_Kmod_Propion',
                          'SEARCH_Ymod_Nitrotyr',
                          'SEARCH_Kmod_Acetyl',
                          'SEARCH_Kmod_Glutaryl',
                          'SEARCH_Kmod_Succinyl',
                          'SEARCH_Rmod_Citrullin',
                          'SEARCH_Ymod_Phospho',
                          'SEARCH_Kmod_Biotin',
                          'SEARCH_Kmod_GlyGly',
                          'SEARCH_Kmod_Trimethyl',
                          'SEARCH_Rmod_Dimethyl-as',
                          'SEARCH_Kmod_Butyryl',
                          'SEARCH_Kmod_Hydroxyisobut',
                          'SEARCH_Rmod_Dimethyl-sym',
                          'SEARCH_Kmod_Crotonyl',
                          'SEARCH_Kmod_Malonyl',
                          'SEARCH_Rmod_Methyl',
                          'SEARCH_Kmod_Dimethyl',
                          'SEARCH_Kmod_Methyl',
                          'SEARCH_Pmod_Hydroxypro'
    ]

    columns_name = ['Formyl (K) Probabilities',
                    'Propion  (K) Probabilities',
                    'Nitrotyrosine (Y) Probabilities',
                    'Acetyl (K) Probabilities',
                    'Glutaryl (K) Probabilities',
                    'Succinyl (K) Probabilities',
                    'Citrullin (R) Probabilities',
                    'Phospho (Y) Probabilities',
                    'Biotin (K) Probabilities',
                    'GlyGly (K) Probabilities',
                    'Trimethyl (K) Probabilities',
                    'Dimethyl (R) Probabilities',
                    'Butyryl (K) Probabilities',
                    'Hydroxyisobutyryl (K) Probabilities',
                    'Dimethyl (R) Probabilities',
                    'Crotonyl (K) Probabilities',
                    'Malonyl (K) Probabilities',
                    'Methyl (R) Probabilities',
                    'Dimethyl (K) Probabilities',
                    'Methyl (KR) Probabilities',
                    'Hydroxyproline manuell (M) Probabilities',
                    ]

    residue_list=['K',
                  'K',
                  'Y',
                  'K',
                  'K',
                  'K',
                  'R',
                  'Y',
                  'K',
                  'K',
                  'K',
                  'R',
                  'K',
                  'K',
                  'R',
                  'K',
                  'K',
                  'R'
                  'K',
                  'K',
                  'P',
                  ]

    mod_code_list=['fo',
                  'pr',
                  'ni',
                  'ac',
                  'gl'
                  'su',
                  'ci',
                  'ph',
                  'bi',
                  'gl',
                  'tr',
                  'di',
                  'bu',
                  'hy',
                  'di',
                  'cr',
                  'ma',
                  'me',
                  'di',
                  'cr',
                  'ma',
                  'me',
                  'di',
                  'me',
                  'hy',
    ]

    mod_code_list_modified=[None,
                  None,
                  None,
                  None,
                  None,
                  None,
                  None,
                  None,
                  None,
                  'gy',
                  None,
                  'ds',
                  None,
                  None,
                  'da',
                  None,
                  None,
                  None,
                  None,
                  None,
                  None,
                  None,
                  None,
                  None,
                  None,
    ]


    full_path = [os.path.join('/lustre/fsn1/projects/rech/bun/ucg81ws/data/pride/',base_path,base_path.split('_')[1]+'_'+base_path.split('_')[2],'combined/txt/msms') for base_path in raw_msms_path_list]
    path_csv_list=[]
    for i in range(len(full_path)):
        print(full_path[i])
        annotate_msms_with_acquisition(input_file=full_path[i]+'.txt',raw_msms_dir=raw_msms_dir_list[i],output_file=full_path[i]+'_annotated.txt',)
        convert_msms_to_prosit(msms_file=full_path[i]+'_annotated.txt',prob_col_name=columns_name[i],output_file=raw_msms_path_list[i]+'.csv',residue=residue_list[i],mod_code=mod_code_list[i],mod_code_modified=mod_cod_list_modified[i])
        path_csv_list.append(raw_msms_path_list[i]+'.csv')

    #merge all dataset
    list_df = [pd.read_csv(path_csv_list[i]) for i in range(len(path_csv_list))]
    df_complete = pd.concat(list_df)
    df_complete.to_csv('df_21_ptms.csv', index=False)

    if not(os.path.exists('/lustre/fsn1/projects/rech/bun/ucg81ws/hr_graph_21_ptms')):
        os.mkdir('/lustre/fsn1/projects/rech/bun/ucg81ws/hr_graph_21_ptms')
    preprocess_ptm_hierarchical_to_chunks('/lustre/fsn1/projects/rech/bun/ucg81ws/hr_graph_21_ptms',with_position=False,out_dir='/lustre/fsn1/projects/rech/bun/ucg81ws/hr_graph_21_ptms')
    # 'test_output',
    # with_position=False,)