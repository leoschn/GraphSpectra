import pandas as pd
import numpy as np
import re
import ast
from pathlib import Path

def annotate_msms_with_acquisition(
    input_file,
    raw_msms_dir,
    output_file
):
    """
    Annotate a MaxQuant msms.txt file with fragmentation and
    activation metadata extracted from corresponding .raw.msms files.

    Parameters
    ----------
    input_file : str or pathlib.Path
        Path to MaxQuant msms.txt.

    raw_msms_dir : str or pathlib.Path
        Directory containing files named:

        <Raw file>.raw.msms

        For example:

        02330a_GG1_3990_07_PTM_TrainKit_Kmod_Acetyl_200fmol_2xIT_2xHCD_R1.raw.msms

    output_file : str or pathlib.Path
        Path where the annotated TSV file will be written.

    Returns
    -------
    annotated_msms : pandas.DataFrame
        Original msms.txt DataFrame with additional RAW acquisition
        metadata columns.

    Added columns
    -------------
    RAW Scan type
    RAW Mass analyzer
    RAW Fragmentation
    RAW Collision energy
    RAW ETD parameter
    RAW Supplemental activation
    Fragmentation match
    """

    # ============================================================
    # Convert inputs to Path objects
    # ============================================================

    input_file = Path(input_file)
    raw_msms_dir = Path(raw_msms_dir)
    output_file = Path(output_file)


    # ============================================================
    # Helper: find corresponding .raw.msms file
    # ============================================================

    def find_raw_msms_file(raw_file_name):

        # Convert to clean string
        raw_file_name = str(raw_file_name).strip()

        # Exact expected filename
        expected_name = f"{raw_file_name}.raw.msms"

        expected_file = raw_msms_dir / expected_name

        # 1. Direct exact match
        if expected_file.is_file():
            return expected_file

        # 2. Search all .raw.msms files
        candidates = list(raw_msms_dir.glob("*.raw.msms"))

        # Normalize names for robust matching
        normalized_target = raw_file_name.strip()

        matches = []

        for candidate in candidates:

            filename = candidate.name

            # Remove ".raw.msms"
            if filename.endswith(".raw.msms"):
                candidate_raw_name = filename[:-len(".raw.msms")]

                if candidate_raw_name.strip() == normalized_target:
                    matches.append(candidate)

        # Exactly one match
        if len(matches) == 1:
            return matches[0]

        # Multiple matches
        if len(matches) > 1:

            print(
                f"\nWARNING: Multiple matching .raw.msms "
                f"files found for:\n"
                f"  {repr(raw_file_name)}"
            )

            for match in matches:
                print(f"  {repr(match.name)}")

            return None

        # No matches: print diagnostics
        print(
            f"\nDEBUG: No file found for:\n"
            f"  Raw file from msms.txt: {repr(raw_file_name)}\n"
            f"  Expected filename:      {repr(expected_name)}\n"
            f"  Search directory:       {raw_msms_dir.resolve()}"
        )

        return None


    # ============================================================
    # Helper: parse scan_type
    # ============================================================

    def parse_scan_type(scan_type):

        result = {
            "RAW Scan type": scan_type,
            "RAW Mass analyzer": None,
            "RAW Fragmentation": None,
            "RAW Collision energy": None,
            "RAW ETD parameter": None,
            "RAW Supplemental activation": None
        }

        if pd.isna(scan_type):
            return result

        text = str(scan_type).strip()
        text_lower = text.lower()


        # --------------------------------------------------------
        # Mass analyzer
        # --------------------------------------------------------

        analyzer_match = re.match(
            r"^\s*(ITMS|FTMS)",
            text,
            flags=re.IGNORECASE
        )

        if analyzer_match:

            result["RAW Mass analyzer"] = (
                analyzer_match.group(1).upper()
            )


        # --------------------------------------------------------
        # Extract all activation events
        #
        # Examples:
        #
        # @cid35.00
        # @hcd28.00
        # @etd123.04
        # @etd123.04@hcd28.00
        # @etd123.04@cid35.00
        # --------------------------------------------------------

        activation_matches = re.findall(
            r"@(cid|hcd|etd)(\d+(?:\.\d+)?)",
            text_lower,
            flags=re.IGNORECASE
        )

        activations = [
            (
                activation.upper(),
                float(value)
            )
            for activation, value
            in activation_matches
        ]

        activation_types = [
            activation
            for activation, value
            in activations
        ]


        # --------------------------------------------------------
        # EThcD
        #
        # Example:
        # @etd123.04@hcd28.00
        # --------------------------------------------------------

        if (
            "ETD" in activation_types
            and
            "HCD" in activation_types
        ):

            result["RAW Fragmentation"] = "ETHCD"

            etd_value = next(
                value
                for activation, value in activations
                if activation == "ETD"
            )

            hcd_value = next(
                value
                for activation, value in activations
                if activation == "HCD"
            )

            result["RAW ETD parameter"] = etd_value

            result[
                "RAW Supplemental activation"
            ] = "HCD"

            result[
                "RAW Collision energy"
            ] = hcd_value/100


        # --------------------------------------------------------
        # ETciD
        #
        # Example:
        # @etd123.04@cid35.00
        # --------------------------------------------------------

        elif (
            "ETD" in activation_types
            and
            "CID" in activation_types
        ):

            result["RAW Fragmentation"] = "ETCID"

            etd_value = next(
                value
                for activation, value in activations
                if activation == "ETD"
            )

            cid_value = next(
                value
                for activation, value in activations
                if activation == "CID"
            )

            result["RAW ETD parameter"] = etd_value

            result[
                "RAW Supplemental activation"
            ] = "CID"

            result[
                "RAW Collision energy"
            ] = cid_value/100


        # --------------------------------------------------------
        # Pure ETD
        # --------------------------------------------------------

        elif activation_types == ["ETD"]:

            result["RAW Fragmentation"] = "ETD"

            result[
                "RAW ETD parameter"
            ] = activations[0][1]


        # --------------------------------------------------------
        # Pure HCD
        # --------------------------------------------------------

        elif activation_types == ["HCD"]:

            result["RAW Fragmentation"] = "HCD"

            result[
                "RAW Collision energy"
            ] = activations[0][1]/100


        # --------------------------------------------------------
        # Pure CID
        # --------------------------------------------------------

        elif activation_types == ["CID"]:

            result["RAW Fragmentation"] = "CID"

            result[
                "RAW Collision energy"
            ] = activations[0][1]/100


        # --------------------------------------------------------
        # Fallback
        # --------------------------------------------------------

        else:

            if "hcd" in text_lower:

                result["RAW Fragmentation"] = "HCD"

            elif "cid" in text_lower:

                result["RAW Fragmentation"] = "CID"

            elif "etd" in text_lower:

                result["RAW Fragmentation"] = "ETD"


        return result


    # ============================================================
    # 1. Load MaxQuant msms.txt
    # ============================================================

    print("=" * 70)
    print("LOADING MAXQUANT msms.txt")
    print("=" * 70)

    msms = pd.read_csv(
        input_file,
        sep="\t",
        low_memory=False
    )

    required_columns = [
        "Raw file",
        "Scan number"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in msms.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns in msms.txt: "
            f"{missing_columns}"
        )

    print(f"Total PSMs: {len(msms):,}")

    print(
        f"Unique raw files: "
        f"{msms['Raw file'].nunique():,}"
    )


    # ------------------------------------------------------------
    # Clean scan number
    # ------------------------------------------------------------

    msms["Scan number"] = pd.to_numeric(
        msms["Scan number"],
        errors="coerce"
    )

    invalid_scan_numbers = (
        msms["Scan number"].isna().sum()
    )

    if invalid_scan_numbers > 0:

        print(
            f"WARNING: {invalid_scan_numbers:,} rows "
            f"have invalid scan numbers."
        )

    msms = msms.dropna(
        subset=[
            "Raw file",
            "Scan number"
        ]
    ).copy()

    msms["Scan number"] = (
        msms["Scan number"].astype(int)
    )


    # ============================================================
    # 2. Process every unique raw file
    # ============================================================

    all_metadata = []

    raw_files = sorted(
        msms["Raw file"]
        .dropna()
        .unique()
    )

    print("\n" + "=" * 70)
    print("PROCESSING .raw.msms FILES")
    print("=" * 70)


    for i, raw_file_name in enumerate(
        raw_files,
        start=1
    ):

        print(
            f"\n[{i}/{len(raw_files)}] "
            f"{raw_file_name}"
        )

        raw_msms_file = find_raw_msms_file(
            raw_file_name
        )

        if raw_msms_file is None:

            print(
                "  WARNING: Corresponding "
                ".raw.msms file not found."
            )

            continue

        print(
            f"  Found: {raw_msms_file.name}"
        )


        # --------------------------------------------------------
        # Load required columns
        # --------------------------------------------------------

        try:

            raw_df = pd.read_csv(
                raw_msms_file,
                usecols=[
                    "scan_number",
                    "scan_type"
                ],
                low_memory=False
            )

        except ValueError:

            print(
                "  WARNING: Required columns "
                "'scan_number' and/or 'scan_type' "
                "not found."
            )

            continue


        # --------------------------------------------------------
        # Clean scan number
        # --------------------------------------------------------

        raw_df["scan_number"] = pd.to_numeric(
            raw_df["scan_number"],
            errors="coerce"
        )

        raw_df = raw_df.dropna(
            subset=["scan_number"]
        ).copy()

        raw_df["scan_number"] = (
            raw_df["scan_number"].astype(int)
        )


        # --------------------------------------------------------
        # Parse scan types
        # --------------------------------------------------------

        parsed = raw_df[
            "scan_type"
        ].apply(parse_scan_type)

        parsed_df = pd.DataFrame(
            parsed.tolist(),
            index=raw_df.index
        )

        raw_df = pd.concat(
            [
                raw_df,
                parsed_df
            ],
            axis=1
        )


        # --------------------------------------------------------
        # Add MaxQuant-compatible raw file name
        # --------------------------------------------------------

        raw_df["Raw file"] = raw_file_name


        # --------------------------------------------------------
        # Keep only required metadata
        # --------------------------------------------------------

        raw_df = raw_df[
            [
                "Raw file",
                "scan_number",
                "RAW Scan type",
                "RAW Mass analyzer",
                "RAW Fragmentation",
                "RAW Collision energy",
                "RAW ETD parameter",
                "RAW Supplemental activation"
            ]
        ]


        # --------------------------------------------------------
        # Restrict to scan numbers actually used by MaxQuant
        #
        # This can significantly reduce memory usage.
        # --------------------------------------------------------

        mq_scans = set(
            msms.loc[
                msms["Raw file"] == raw_file_name,
                "Scan number"
            ]
        )

        raw_df = raw_df[
            raw_df["scan_number"].isin(
                mq_scans
            )
        ]


        print(
            f"  Matching RAW spectra retained: "
            f"{len(raw_df):,}"
        )

        all_metadata.append(raw_df)


    # ============================================================
    # 3. Combine metadata
    # ============================================================

    print("\n" + "=" * 70)
    print("COMBINING ACQUISITION METADATA")
    print("=" * 70)

    if not all_metadata:

        raise RuntimeError(
            "No valid .raw.msms files could be loaded."
        )

    acquisition_metadata = pd.concat(
        all_metadata,
        ignore_index=True
    )


    # Ensure unique Raw file + scan_number pairs

    duplicate_spectra = (
        acquisition_metadata
        .duplicated(
            subset=[
                "Raw file",
                "scan_number"
            ],
            keep=False
        )
        .sum()
    )

    if duplicate_spectra > 0:

        print(
            f"WARNING: {duplicate_spectra:,} duplicate "
            "Raw file + scan_number entries found."
        )

        acquisition_metadata = (
            acquisition_metadata
            .drop_duplicates(
                subset=[
                    "Raw file",
                    "scan_number"
                ],
                keep="first"
            )
        )


    print(
        f"Total extracted RAW spectra: "
        f"{len(acquisition_metadata):,}"
    )


    # ============================================================
    # 4. Merge with MaxQuant
    # ============================================================

    print("\n" + "=" * 70)
    print("MERGING WITH MAXQUANT msms.txt")
    print("=" * 70)

    annotated_msms = msms.merge(

        acquisition_metadata,

        left_on=[
            "Raw file",
            "Scan number"
        ],

        right_on=[
            "Raw file",
            "scan_number"
        ],

        how="left",

        validate="many_to_one"
    )


    # ============================================================
    # 5. Check fragmentation agreement
    # ============================================================

    if "Fragmentation" in annotated_msms.columns:

        annotated_msms[
            "Fragmentation match"
        ] = (
            annotated_msms[
                "Fragmentation"
            ]
            .astype("string")
            .str.upper()

            ==

            annotated_msms[
                "RAW Fragmentation"
            ]
            .astype("string")
            .str.upper()
        )

    else:

        annotated_msms[
            "Fragmentation match"
        ] = pd.NA


    # ============================================================
    # 6. QC summary
    # ============================================================

    print("\n" + "=" * 70)
    print("QC SUMMARY")
    print("=" * 70)

    total_psms = len(annotated_msms)

    matched_raw = (
        annotated_msms[
            "RAW Fragmentation"
        ]
        .notna()
        .sum()
    )

    print(f"Total PSMs: {total_psms:,}")

    print(
        f"PSMs matched to RAW metadata: "
        f"{matched_raw:,} "
        f"({matched_raw / total_psms * 100:.2f}%)"
    )


    if "Fragmentation" in annotated_msms.columns:

        valid_comparison = annotated_msms[
            annotated_msms[
                "RAW Fragmentation"
            ].notna()
        ]

        n_matching = (
            valid_comparison[
                "Fragmentation match"
            ]
            .sum()
        )

        print(
            f"Fragmentation matches: "
            f"{n_matching:,} / "
            f"{len(valid_comparison):,} "
            f"({n_matching / len(valid_comparison) * 100:.2f}%)"
            if len(valid_comparison) > 0
            else "No fragmentation matches available."
        )


    print(
        "\nFragmentation / analyzer / energy:"
    )

    condition_summary = (
        annotated_msms
        .groupby(
            [
                "RAW Fragmentation",
                "RAW Mass analyzer",
                "RAW Collision energy"
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="n_PSMs"
        )
        .sort_values(
            "n_PSMs",
            ascending=False
        )
    )

    print(
        condition_summary.to_string(
            index=False
        )
    )


    # ============================================================
    # 7. Display mismatches
    # ============================================================

    if "Fragmentation" in annotated_msms.columns:

        mismatches = annotated_msms[
            annotated_msms[
                "RAW Fragmentation"
            ].notna()
            &
            ~annotated_msms[
                "Fragmentation match"
            ]
        ]

        print(
            f"\nFragmentation mismatches: "
            f"{len(mismatches):,}"
        )

        if len(mismatches) > 0:

            mismatch_columns = [
                "Raw file",
                "Scan number",
                "Fragmentation",
                "RAW Fragmentation",
                "RAW Collision energy",
                "RAW ETD parameter",
                "RAW Scan type"
            ]

            available_columns = [
                column
                for column in mismatch_columns
                if column in mismatches.columns
            ]

            print(
                mismatches[
                    available_columns
                ]
                .drop_duplicates()
                .head(20)
                .to_string(index=False)
            )


    # ============================================================
    # 8. Save output
    # ============================================================

    print("\n" + "=" * 70)
    print("WRITING OUTPUT")
    print("=" * 70)

    annotated_msms.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print(
        f"Annotated msms file saved to:\n"
        f"{output_file.resolve()}"
    )


    # ============================================================
    # 9. Return DataFrame
    # ============================================================

    return annotated_msms

# ----------------------------------------------------------
# Mapping of fragment -> index in Prosit intensity vector
# Order:
# y1+, y1++, y1+++, b1+, b1++, b1+++,
# y2+, y2++, ...
# ...
# y29+, y29++, ...
# ----------------------------------------------------------

def prosit_index(ion_type, number, charge):
    """
    Returns index in the 174-dimensional Prosit intensity vector.
    """
    if number < 1 or number > 29:
        return None

    if charge not in [1, 2, 3]:
        return None

    base = (number - 1) * 6

    if ion_type == "y":
        return base + (charge - 1)

    elif ion_type == "b":
        return base + 3 + (charge - 1)

    return None

import re
import pandas as pd


def parse_probabilities(probability_string, residue):
    """
    Parse a MaxQuant '* Probabilities' string.

    Examples
    --------
    KAAK(0.98)            -> [0.0, 0.98]
    K(0.02)AAK(0.98)      -> [0.02, 0.98]
    KAAK                  -> [0.0, 0.0]
    """

    probs = []

    s = str(probability_string)

    i = 0

    while i < len(s):

        if s[i] != residue:
            i += 1
            continue

        # residue(prob)
        if i + 1 < len(s) and s[i + 1] == "(":

            j = s.find(")", i + 1)

            probs.append(float(s[i + 2:j]))

            i = j + 1

        else:
            probs.append(0.0)
            i += 1

    return probs


def resolve_localized_modification(
    modified_sequence,
    probability_string,
    residue="K",
    mod_code="cr",
    low=0.05,
    high=0.95,
):
    """
    Resolve PTM localization using a MaxQuant '* Probabilities' column.

    Parameters
    ----------
    modified_sequence : str
        e.g. "_AAAK(cr)AAK_"

    probability_string : str or NaN
        e.g. "AAAK(0.02)AAK(0.98)"

    residue : str
        Residue carrying the modification.

    mod_code : str
        Modification code used in Modified sequence
        ("cr", "ox", "ac", ...)

    Returns
    -------
    str
        Corrected modified sequence (without surrounding "_")

    None
        If localization is ambiguous.
    """
    # --------------------------------------------------
    # Invalid sequence
    # --------------------------------------------------

    if modified_sequence is None:
        return None

    if pd.isna(modified_sequence):
        return None

    # --------------------------------------------------
    # No localization probabilities
    #
    # Important: preserve the existing sequence!
    # --------------------------------------------------

    if probability_string is None:
        return modified_sequence

    if pd.isna(probability_string):
        return modified_sequence

    probability_string = str(probability_string).strip()

    if probability_string == "":
        return modified_sequence

    # --------------------------------------------------
    # Normal localization
    # --------------------------------------------------

    seq = modified_sequence.strip("_")

    mod_pattern = f"{residue}({mod_code})"

    # Number of candidate residues
    residue_positions = [
        m.start()
        for m in re.finditer(re.escape(residue), seq)
    ]

    # Empty probability column => every residue has probability 0
    if pd.isna(probability_string) or str(probability_string).strip() == "":
        probs = [0.0] * len(residue_positions)

    else:
        probs = parse_probabilities(probability_string, residue)

        if len(probs) != len(residue_positions):
            raise ValueError(
                f"Found {len(residue_positions)} {residue} residues but "
                f"{len(probs)} probabilities.\n"
                f"Modified sequence: {modified_sequence}\n"
                f"Probability string: {probability_string}"
            )

    modified_indices = set()

    for i, p in enumerate(probs):

        if p >= high:
            modified_indices.add(i)

        elif p <= low:
            pass

        else:
            # ambiguous localization
            return None

    # ------------------------------------------------------------------
    # Rebuild sequence
    # ------------------------------------------------------------------

    out = []

    residue_index = 0
    i = 0

    while i < len(seq):

        if seq.startswith(mod_pattern, i):

            if residue_index in modified_indices:
                out.append(mod_pattern)
            else:
                out.append(residue)

            residue_index += 1
            i += len(mod_pattern)

        elif seq[i] == residue:

            if residue_index in modified_indices:
                out.append(mod_pattern)
            else:
                out.append(residue)

            residue_index += 1
            i += 1

        else:
            out.append(seq[i])
            i += 1

    return "".join(out)

# ----------------------------------------------------------
# Charge one-hot
# ----------------------------------------------------------

def charge_onehot(z):

    vec = [0] * 6

    if 1 <= int(z) <= 6:
        vec[int(z) - 1] = 1

    return vec


# ----------------------------------------------------------
# Convert MaxQuant modified sequence
# ----------------------------------------------------------

def convert_sequence(seq):

    # remove terminal underscores
    seq = seq.strip("_")

    # oxidation
    seq = seq.replace("(ox)", "(ox)")

    # extend here if necessary
    # seq = seq.replace("(ac)", "(ac)")
    # ...

    return seq


# ----------------------------------------------------------
# Parse ion annotation
# Examples:
#
# b5
# y7
# b4(2+)
# y10(3+)
#
# Ignore:
# y5-H2O
# b7-NH3
# a ions
# M ions
# ----------------------------------------------------------

ion_regex = re.compile(r"^([by])(\d+)(?:\((\d)\+\))?$")


def parse_ion(annotation):

    annotation = annotation.strip()

    if "-" in annotation:
        return None

    if annotation.startswith("a"):
        return None

    m = ion_regex.match(annotation)

    if m is None:
        return None

    ion_type = m.group(1)
    number = int(m.group(2))

    charge = 1
    if m.group(3):
        charge = int(m.group(3))

    return ion_type, number, charge


# ----------------------------------------------------------
# Main converter
# ----------------------------------------------------------

def convert_msms_to_prosit(
    msms_file,
    output_file,
    fragmentation_filter='HCD',
    residue="K",
    mod_code="cr",
    mod_code_modified=None,
    prob_col_name='Crotonyl (K) Probabilities',
    low=0.05,
    high=0.95,
):

    df = pd.read_csv(msms_file, sep="\t", low_memory=False)
    #only keep relevant frag type if filter is specified
    if fragmentation_filter is not None :
        df = df[df['RAW Fragmentation']==fragmentation_filter]
    results = []

    for _, row in df.iterrows():

        sequence = row["Modified sequence"]

        # Resolve Variable localization
        sequence = resolve_localized_modification(
            modified_sequence=sequence,
            probability_string=row[prob_col_name],
            residue=residue,
            mod_code=mod_code
        )

        #Resolve Mox localization
        sequence = resolve_localized_modification(
            modified_sequence=sequence,
            probability_string=row['Oxidation (M) Probabilities'],
            residue='M',
            mod_code='ox'
        )

        # Ambiguous localization -> discard spectrum
        if sequence is None:
            continue

        # Remove leading/trailing "_", convert notation, etc.
        sequence = convert_sequence(sequence)
        seq_length = len(sequence) - 4*sequence.count('(') #to account for PTMs

        precursor_charge = int(row["Charge"])

        intensity_norm_vector = np.zeros(174, dtype=float)
        intensity_raw_vector= np.zeros(174, dtype=float)
        #set impossible config to -1 :

        impossible_index = np.full(174, False)

        #based on length
        impossible_index[6*(seq_length-1):] = True

        #based on charge
        if precursor_charge == 1:
            impossible_index[1::6] = True
            impossible_index[2::6] = True
            impossible_index[4::6] = True
            impossible_index[5::6] = True

        elif precursor_charge == 2:
            impossible_index[2::6] = True
            impossible_index[5::6] = True

        intensity_norm_vector[impossible_index] = -1
        intensity_raw_vector[impossible_index] = -1

        matches = str(row["Matches"]).split(";")
        intensities = str(row["Intensities"]).split(";")

        parsed = []

        for ion, inten in zip(matches, intensities):

            p = parse_ion(ion)

            if p is None:
                continue

            try:
                inten = float(inten)
            except:
                continue

            parsed.append((p, inten))

        if len(parsed) == 0:
            continue

        max_intensity = max(i for _, i in parsed)

        if max_intensity == 0:
            continue

        for (ion_type, number, charge), inten in parsed:

            idx = prosit_index(ion_type, number, charge)

            if idx is None:
                continue

            norm = inten / max_intensity

            # keep largest intensity if duplicated
            if intensity_norm_vector[idx] != -1:
                intensity_norm_vector[idx] = max(
                    intensity_norm_vector[idx],
                    norm
                )

            if intensity_raw_vector[idx] != -1:
                intensity_raw_vector[idx] = max(
                    intensity_raw_vector[idx],
                    inten
                )

        if mod_code_modified != None:
            sequence=sequence.replace(residue+'('+mod_code+')',residue+'('+mod_code_modified+')')

        results.append({
            "intensities_norm": intensity_norm_vector.tolist(),
            "intensities_raw": intensity_raw_vector.tolist(),
            "sequence": sequence,
            "precursor_charge_onehot": charge_onehot(
                precursor_charge
            ),
            "collision_energy": row['RAW Collision energy'],
        })

    out = pd.DataFrame(results)

    out.to_csv(output_file, index=False)

    print(f"Wrote {len(out)} spectra")
    print(output_file)


# ----------------------------------------------------------
# Example
# ----------------------------------------------------------

if __name__ == "__main__":

    annotate_msms_with_acquisition(input_file="../dataset_dummy/SEARCH_Kmod_Formyl/Kmod_Formyl/combined/txt/msms.txt",raw_msms_dir="../dataset_dummy/SEARCH_Kmod_Formyl/Kmod_Formyl/",output_file="../dataset_dummy/SEARCH_Kmod_Formyl/Kmod_Formyl/combined/txt/msms_annotated.txt")

    convert_msms_to_prosit(
        msms_file="../dataset_dummy/SEARCH_Kmod_Formyl/Kmod_Formyl/combined/txt/msms_annotated.txt",
        output_file="../dataset_dummy/prosit_(cr)_2.csv",
        residue="K",
        mod_code="fo",
        low=0.05,
        high=0.95,
        prob_col_name='Formyl (K) Probabilities'
    )

    df_prosit = pd.read_csv('../dataset_dummy/prosit_(cr)_2.csv')

    # Convert string representations to lists
    df_prosit["intensities_raw"] = df_prosit["intensities_raw"].apply(ast.literal_eval)

    group_cols = [
        "sequence",
        "precursor_charge_onehot",
        "collision_energy"
    ]

    # Element-wise mean
    def mean_intensities(arrays):
        return np.mean(np.vstack(arrays), axis=0).tolist()


    # Normalize while preserving -1
    def normalize_intensities(x):
        x = np.array(x, dtype=float)

        mask = x != -1

        # Normalize only valid values
        x[mask] = x[mask] / x[mask].max()

        return x.tolist()

    # Merge rows and compute mean raw intensities
    merged = (
        df_prosit.groupby(group_cols, as_index=False)
        .agg({
            "intensities_raw": mean_intensities
        })
    )

    # Compute normalized intensities from averaged raw intensities
    merged["intensities"] = merged["intensities_raw"].apply(
        normalize_intensities
    )

    # Reorder columns
    merged = merged[
        [
            "intensities",
            "sequence",
            "precursor_charge_onehot",
            "collision_energy",
        ]
    ]

    merged.to_csv('../dataset_dummy/prosit_(cr)_mean.csv',index=False)