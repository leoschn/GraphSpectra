import argparse
import html
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from data.graph_creation_utils import (
    atom_features,
    atom_featurizer,
    precompute_mol_features,
    seq_to_mol_with_ox,
)


ONE_HOT_FEATURE_LABELS: Dict[str, List[str]] = {
    "cip_code": ["R", "S"],
    "degree": ["0", "1", "2", "3", "4", "5+"],
    "element": [
        "B",
        "C",
        "N",
        "O",
        "Si",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
        "other",
    ],
    "hybridization": ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2"],
    "is_in_ring_size_n": ["0", "3", "4", "5", "6", "7", "8", "9+"],
    "num_hs": ["0", "1", "2", "3", "4+"],
    "num_valence": ["0", "1", "2", "3", "4", "5", "6+"],
}

SVG_WIDTH = 720
SVG_HEIGHT = 260
SVG_MARGIN_LEFT = 58
SVG_MARGIN_RIGHT = 18
SVG_MARGIN_TOP = 18
SVG_MARGIN_BOTTOM = 48


def compute_atom_feature_dataframe(
    mol_seq_list: Iterable[str],
    *,
    include_atom_metadata: bool = True,
    skip_invalid: bool = False,
) -> pd.DataFrame:
    """Build one row per atom with each atom feature kept as its original vector."""
    rows = []

    for sequence_index, mol_seq in enumerate(mol_seq_list):
        try:
            mol = _sequence_to_mol(mol_seq)
            mol_feat = precompute_mol_features(mol)
        except Exception:
            if skip_invalid:
                continue
            raise

        for atom in mol.GetAtoms():
            feature_vectors = atom_featurizer(
                atom,
                mol_feat,
                exclude_feature=None,
                concat=False,
            )
            row = dict(zip(atom_features, feature_vectors))

            if include_atom_metadata:
                row.update(
                    {
                        "sequence_index": sequence_index,
                        "sequence": mol_seq,
                        "atom_index": atom.GetIdx(),
                        "atom_symbol": atom.GetSymbol(),
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_atom_features(
    df: pd.DataFrame,
    *,
    feature_names: Optional[Sequence[str]] = None,
    default_feature_labels: Optional[Mapping[str, Sequence[str]]] = None,
    feature_labels: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return scalar histogram rows and one-hot class count rows."""
    labels = dict(default_feature_labels or ONE_HOT_FEATURE_LABELS)
    if feature_labels:
        labels.update({name: list(values) for name, values in feature_labels.items()})

    scalar_rows = []
    class_rows = []
    for feature in _iter_feature_columns(df, feature_names):
        values = [_as_1d_float_array(value) for value in df[feature].dropna()]
        if not values:
            continue

        width = max(len(value) for value in values)
        if _is_one_hot_feature(feature, values, labels):
            category_counts, missing_count = _one_hot_counts(values, width)
            category_labels = _labels_for_feature(feature, width, labels)
            total = category_counts.sum()
            denom = total if total else 1.0

            for label, count in zip(category_labels, category_counts):
                class_rows.append(
                    {
                        "feature": feature,
                        "category": label,
                        "count": int(count),
                        "fraction": float(count / denom),
                    }
                )

            if missing_count:
                class_rows.append(
                    {
                        "feature": feature,
                        "category": "none",
                        "count": int(missing_count),
                        "fraction": float(missing_count / len(values)),
                    }
                )
            continue

        scalar_values = _scalar_values(values)
        if scalar_values.size == 0:
            continue

        scalar_rows.append(
            {
                "feature": feature,
                "count": int(scalar_values.size),
                "mean": float(np.mean(scalar_values)),
                "std": float(np.std(scalar_values)),
                "min": float(np.min(scalar_values)),
                "q25": float(np.quantile(scalar_values, 0.25)),
                "median": float(np.median(scalar_values)),
                "q75": float(np.quantile(scalar_values, 0.75)),
                "max": float(np.max(scalar_values)),
            }
        )

    return pd.DataFrame(scalar_rows), pd.DataFrame(class_rows)


def write_feature_exploration_report(
    df: pd.DataFrame,
    output_path: str | Path = "atom_feature_report.html",
    *,
    bins: int = 30,
    feature_names: Optional[Sequence[str]] = None,
    default_feature_labels: Optional[Mapping[str, Sequence[str]]] = None,
    feature_labels: Optional[Mapping[str, Sequence[str]]] = None,
    observation_label: str = "atoms",
    title: str = "Atom Feature Exploration",
) -> Path:
    """Write an HTML report with histograms for scalar features and bars for one-hot features."""
    labels = dict(default_feature_labels or ONE_HOT_FEATURE_LABELS)
    if feature_labels:
        labels.update({name: list(values) for name, values in feature_labels.items()})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sections = []
    scalar_summary_rows = []
    class_summary_rows = []

    for feature in _iter_feature_columns(df, feature_names):
        values = [_as_1d_float_array(value) for value in df[feature].dropna()]
        if not values:
            continue

        width = max(len(value) for value in values)
        if _is_one_hot_feature(feature, values, labels):
            counts, missing_count = _one_hot_counts(values, width)
            chart_labels = _labels_for_feature(feature, width, labels)
            sections.append(
                _feature_card(
                    feature,
                    _bar_chart_svg(
                        chart_labels,
                        counts,
                        observation_label=observation_label,
                    ),
                )
            )

            total = counts.sum()
            denom = total if total else 1.0
            for label, count in zip(chart_labels, counts):
                class_summary_rows.append(
                    {
                        "feature": feature,
                        "category": label,
                        "count": int(count),
                        "fraction": float(count / denom),
                    }
                )
            if missing_count:
                class_summary_rows.append(
                    {
                        "feature": feature,
                        "category": "none",
                        "count": int(missing_count),
                        "fraction": float(missing_count / len(values)),
                    }
                )
            continue

        scalar_values = _scalar_values(values)
        if scalar_values.size == 0:
            continue

        sections.append(
            _feature_card(
                feature,
                _histogram_svg(
                    scalar_values,
                    bins=bins,
                    observation_label=observation_label,
                ),
            )
        )
        scalar_summary_rows.append(
            {
                "feature": feature,
                "count": int(scalar_values.size),
                "mean": float(np.mean(scalar_values)),
                "std": float(np.std(scalar_values)),
                "min": float(np.min(scalar_values)),
                "q25": float(np.quantile(scalar_values, 0.25)),
                "median": float(np.median(scalar_values)),
                "q75": float(np.quantile(scalar_values, 0.75)),
                "max": float(np.max(scalar_values)),
            }
        )

    scalar_summary = pd.DataFrame(scalar_summary_rows)
    class_summary = pd.DataFrame(class_summary_rows)
    _write_summaries(output_path, scalar_summary, class_summary)
    output_path.write_text(
        _html_document(
            title=title,
            n_rows=len(df),
            observation_label=observation_label,
            n_features=len(_iter_feature_columns(df, feature_names)),
            scalar_summary=scalar_summary,
            class_summary=class_summary,
            sections=sections,
        ),
        encoding="utf-8",
    )
    return output_path


def sequence_to_mol(sequence: str):
    mol = seq_to_mol_with_ox(sequence)
    if mol is None:
        raise ValueError(f"Failed to build molecule from sequence: {sequence}")
    return mol


def _sequence_to_mol(sequence: str):
    return sequence_to_mol(sequence)


def _iter_feature_columns(
    df: pd.DataFrame,
    feature_names: Optional[Sequence[str]] = None,
) -> List[str]:
    names = atom_features if feature_names is None else feature_names
    return [feature for feature in names if feature in df.columns]


def _as_1d_float_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value)
    else:
        array = np.asarray([value])

    return array.astype(float).ravel()


def _is_one_hot_feature(
    feature: str,
    values: Sequence[np.ndarray],
    labels: Mapping[str, Sequence[str]],
) -> bool:
    if feature in labels:
        return True

    width = max(len(value) for value in values)
    if width <= 1:
        return False

    padded = np.vstack([_pad_array(value, width) for value in values])
    return bool(
        np.all(np.isin(padded, [0.0, 1.0]))
        and np.all(padded.sum(axis=1) <= 1.0 + 1e-9)
    )


def _one_hot_counts(values: Sequence[np.ndarray], width: int) -> Tuple[np.ndarray, int]:
    padded = np.vstack([_pad_array(value, width) for value in values])
    missing_count = int(np.sum(padded.sum(axis=1) == 0.0))
    return padded.sum(axis=0).astype(int), missing_count


def _scalar_values(values: Sequence[np.ndarray]) -> np.ndarray:
    scalars = []
    for value in values:
        if value.size == 0:
            continue
        scalars.append(float(value[0]))
    return np.asarray(scalars, dtype=float)


def _pad_array(value: np.ndarray, width: int) -> np.ndarray:
    if len(value) == width:
        return value
    padded = np.zeros(width, dtype=float)
    padded[: len(value)] = value
    return padded


def _labels_for_feature(
    feature: str,
    width: int,
    labels: Mapping[str, Sequence[str]],
) -> List[str]:
    feature_labels = list(labels.get(feature, []))
    if len(feature_labels) >= width:
        return feature_labels[:width]

    return feature_labels + [str(index) for index in range(len(feature_labels), width)]


def _histogram_svg(
    values: np.ndarray,
    *,
    bins: int,
    observation_label: str,
) -> str:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return "<p>No finite values.</p>"

    if np.min(finite_values) == np.max(finite_values):
        counts = np.asarray([finite_values.size])
        edges = np.asarray([finite_values[0] - 0.5, finite_values[0] + 0.5])
    else:
        counts, edges = np.histogram(finite_values, bins=bins)

    plot_width = SVG_WIDTH - SVG_MARGIN_LEFT - SVG_MARGIN_RIGHT
    plot_height = SVG_HEIGHT - SVG_MARGIN_TOP - SVG_MARGIN_BOTTOM
    bar_width = plot_width / len(counts)
    max_count = max(int(np.max(counts)), 1)

    bars = []
    for index, count in enumerate(counts):
        height = (count / max_count) * plot_height
        x = SVG_MARGIN_LEFT + index * bar_width
        y = SVG_MARGIN_TOP + plot_height - height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 1, 1):.2f}" '
            f'height="{height:.2f}"><title>{int(count)} {html.escape(observation_label)}</title></rect>'
        )

    axis = _axis_svg(
        x_min=float(edges[0]),
        x_max=float(edges[-1]),
        y_max=max_count,
        x_label="value",
        y_label=observation_label,
    )
    stats = (
        f"n={finite_values.size} | mean={np.mean(finite_values):.4g} | "
        f"median={np.median(finite_values):.4g} | "
        f"min={np.min(finite_values):.4g} | max={np.max(finite_values):.4g}"
    )
    return _svg_wrap("".join(bars) + axis, footer=stats)


def _bar_chart_svg(
    labels: Sequence[str],
    counts: np.ndarray,
    *,
    observation_label: str,
) -> str:
    plot_width = SVG_WIDTH - SVG_MARGIN_LEFT - SVG_MARGIN_RIGHT
    plot_height = SVG_HEIGHT - SVG_MARGIN_TOP - SVG_MARGIN_BOTTOM
    max_count = max(int(np.max(counts)) if len(counts) else 0, 1)
    bar_width = plot_width / max(len(labels), 1)

    bars = []
    for index, (label, count) in enumerate(zip(labels, counts)):
        height = (count / max_count) * plot_height
        x = SVG_MARGIN_LEFT + index * bar_width
        y = SVG_MARGIN_TOP + plot_height - height
        label_x = x + bar_width / 2
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 6, 1):.2f}" '
            f'height="{height:.2f}"><title>{html.escape(label)}: {int(count)} '
            f'{html.escape(observation_label)}</title></rect>'
        )
        bars.append(
            f'<text class="tick x-category" x="{label_x:.2f}" y="{SVG_HEIGHT - 23}" '
            f'text-anchor="end" transform="rotate(-35 {label_x:.2f} {SVG_HEIGHT - 23})">'
            f"{html.escape(label)}</text>"
        )

    axis = _axis_svg(
        x_min=None,
        x_max=None,
        y_max=max_count,
        x_label="",
        y_label=observation_label,
    )
    total = int(np.sum(counts))
    return _svg_wrap(
        "".join(bars) + axis,
        footer=f"n={total} assigned {observation_label}",
    )


def _axis_svg(
    *,
    x_min: Optional[float],
    x_max: Optional[float],
    y_max: int,
    x_label: str,
    y_label: str,
) -> str:
    plot_height = SVG_HEIGHT - SVG_MARGIN_TOP - SVG_MARGIN_BOTTOM
    plot_bottom = SVG_MARGIN_TOP + plot_height
    plot_right = SVG_WIDTH - SVG_MARGIN_RIGHT

    parts = [
        f'<line class="axis" x1="{SVG_MARGIN_LEFT}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" />',
        f'<line class="axis" x1="{SVG_MARGIN_LEFT}" y1="{SVG_MARGIN_TOP}" x2="{SVG_MARGIN_LEFT}" y2="{plot_bottom}" />',
        f'<text class="tick" x="{SVG_MARGIN_LEFT - 8}" y="{plot_bottom}" text-anchor="end">0</text>',
        f'<text class="tick" x="{SVG_MARGIN_LEFT - 8}" y="{SVG_MARGIN_TOP + 4}" text-anchor="end">{y_max}</text>',
        f'<text class="axis-label" x="18" y="{SVG_MARGIN_TOP + plot_height / 2}" '
        f'text-anchor="middle" transform="rotate(-90 18 {SVG_MARGIN_TOP + plot_height / 2})">{y_label}</text>',
    ]
    if x_min is not None and x_max is not None:
        parts.extend(
            [
                f'<text class="tick" x="{SVG_MARGIN_LEFT}" y="{SVG_HEIGHT - 12}" text-anchor="middle">{x_min:.3g}</text>',
                f'<text class="tick" x="{plot_right}" y="{SVG_HEIGHT - 12}" text-anchor="middle">{x_max:.3g}</text>',
                f'<text class="axis-label" x="{SVG_MARGIN_LEFT + (plot_right - SVG_MARGIN_LEFT) / 2}" '
                f'y="{SVG_HEIGHT - 12}" text-anchor="middle">{x_label}</text>',
            ]
        )
    return "".join(parts)


def _svg_wrap(body: str, *, footer: str) -> str:
    return (
        f'<svg viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img">'
        f"{body}"
        f'<text class="footer" x="{SVG_MARGIN_LEFT}" y="{SVG_HEIGHT - 4}">{html.escape(footer)}</text>'
        "</svg>"
    )


def _feature_card(feature: str, chart_svg: str) -> str:
    return (
        '<section class="feature-card">'
        f"<h2>{html.escape(feature)}</h2>"
        f"{chart_svg}"
        "</section>"
    )


def _write_summaries(
    output_path: Path,
    scalar_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
) -> None:
    if not scalar_summary.empty:
        scalar_summary.to_csv(output_path.with_suffix(".scalar_summary.csv"), index=False)
    if not class_summary.empty:
        class_summary.to_csv(output_path.with_suffix(".class_summary.csv"), index=False)


def _html_document(
    *,
    title: str,
    n_rows: int,
    observation_label: str,
    n_features: int,
    scalar_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    sections: Sequence[str],
) -> str:
    scalar_count = 0 if scalar_summary.empty else scalar_summary["feature"].nunique()
    class_count = 0 if class_summary.empty else class_summary["feature"].nunique()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f7f4ee;
      --panel: #ffffff;
      --ink: #20211f;
      --muted: #6d7168;
      --line: #d8d4ca;
      --bar: #2f6f73;
      --bar-hover: #1f5559;
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background: linear-gradient(180deg, #f7f4ee 0%, #eef3ef 100%);
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      width: min(1160px, calc(100% - 32px));
      margin: 0 auto;
      padding: 34px 0 56px;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      margin-bottom: 26px;
      padding-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3.5rem);
      font-weight: 500;
      letter-spacing: 0;
    }}
    .summary {{
      color: var(--muted);
      font-size: 1rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 18px;
    }}
    .feature-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 8px 28px rgba(35, 39, 31, 0.07);
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 1rem;
      font-family: "Courier New", monospace;
      font-weight: 700;
    }}
    svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    rect {{
      fill: var(--bar);
    }}
    rect:hover {{
      fill: var(--bar-hover);
    }}
    .axis {{
      stroke: var(--line);
      stroke-width: 1;
    }}
    .tick, .axis-label, .footer {{
      fill: var(--muted);
      font-family: "Courier New", monospace;
      font-size: 11px;
    }}
    .footer {{
      font-size: 10px;
    }}
    @media (max-width: 520px) {{
      main {{
        width: min(100% - 20px, 1160px);
        padding-top: 20px;
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
      .feature-card {{
        padding: 12px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{html.escape(title)}</h1>
      <div class="summary">
        {n_rows} {html.escape(observation_label)} | {n_features} feature columns | {scalar_count} scalar histograms | {class_count} one-hot class charts
      </div>
    </header>
    <div class="grid">
      {"".join(sections)}
    </div>
  </main>
</body>
</html>
"""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore atom feature distributions from peptide sequences.",
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
        default="atom_feature_report.html",
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

    input_df = pd.read_csv(args.csv)
    sequences = input_df[args.sequence_column].dropna().astype(str).head(args.limit).tolist()
    feature_df = compute_atom_feature_dataframe(sequences, skip_invalid=args.skip_invalid)
    output_path = write_feature_exploration_report(feature_df, args.output, bins=args.bins)
    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.scalar_summary.csv')}")
    print(f"Wrote {output_path.with_suffix('.class_summary.csv')}")


if __name__ == "__main__":
    main()
