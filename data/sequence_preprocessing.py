import pandas as pd
import numpy as np
import re
import ast


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
    collision_energy=0.35,
    fragmentation=2.0,
    residue="K",
    mod_code="cr",
    prob_col_name='Crotonyl (K) Probabilities',
    low=0.05,
    high=0.95,
):

    df = pd.read_csv(msms_file, sep="\t", low_memory=False)

    results = []

    for _, row in df.iterrows():

        sequence = row["Modified sequence"]

        # Resolve Crotonyl localization
        sequence = resolve_localized_modification(
            modified_sequence=sequence,
            probability_string=row[prob_col_name],
            residue=residue,
            mod_code=mod_code
        )

        # Ambiguous localization -> discard spectrum
        if sequence is None:
            continue

        # Remove leading/trailing "_", convert notation, etc.
        sequence = convert_sequence(sequence)

        precursor_charge = int(row["Charge"])

        intensity_vector = np.full(174, -1.0)

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
            if intensity_vector[idx] == -1:
                intensity_vector[idx] = norm
            else:
                intensity_vector[idx] = max(
                    intensity_vector[idx],
                    norm
                )

        results.append({
            "intensities": intensity_vector.tolist(),
            "sequence": sequence,
            "precursor_charge_onehot": charge_onehot(
                precursor_charge
            ),
            "collision_energy": collision_energy,
            "fragmentation": fragmentation
        })

    out = pd.DataFrame(results)

    out.to_csv(output_file, index=False)

    print(f"Wrote {len(out)} spectra")
    print(output_file)


# ----------------------------------------------------------
# Example
# ----------------------------------------------------------

if __name__ == "__main__":

    convert_msms_to_prosit(
        msms_file="../dataset_dummy/msms.txt",
        output_file="../dataset_dummy/prosit_(cr)_2.csv",
        collision_energy=0.35,
        residue="K",
        mod_code="cr",
        low=0.05,
        high=0.95,
    )