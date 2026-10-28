"""Run one self-curated binary trait through the reconstruction pipeline.

Example
-------
python examples/reconstruct_binary_trait.py \
    /path/to/binary_curation_example.csv \
    --output-dir example_output/binary_trait

The example input contains one row per positive observation. It therefore
does not include ``Attribute`` or ``Attribute_value`` columns. This script
adds them before calling the public :func:`reconstruct_trait` wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Permit running this file directly from a source checkout. An installed copy
# of the package does not need this path adjustment.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bugphyzz_reconstruction.asr import reconstruct_trait
from bugphyzz_reconstruction.asr.ltp import ltp
from bugphyzz_reconstruction.taxonomy.frozen import FrozenTaxonomyProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct one self-curated binary microbial trait."
    )
    parser.add_argument("observations", type=Path, help="Input CSV file.")
    parser.add_argument(
        "--attribute-name",
        help=(
            "Name of the binary attribute. By default, use the single "
            "Attribute_group value in the input table."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example_output") / "binary_trait",
        help="Directory for inspectable pipeline stages.",
    )
    parser.add_argument(
        "--min-annotated-tips",
        type=int,
        default=None,
        help=(
            "Run ASR after this many distinct LTP tips are annotated. "
            "When omitted, use the historical one-percent rule."
        ),
    )
    return parser.parse_args()


def load_positive_binary_observations(
    path: Path,
    *,
    attribute_name: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Load a table whose rows each assert that one binary trait is present."""
    table = pd.read_csv(
        path,
        dtype={"NCBI_ID": "string", "Parent_NCBI_ID": "string"},
        na_values=["NA", ""],
        keep_default_na=True,
    )

    groups = (
        table["Attribute_group"]
        .astype("string")
        .dropna()
        .str.strip()
        .loc[lambda values: values.ne("")]
        .drop_duplicates()
        .tolist()
    )
    if len(groups) != 1:
        raise ValueError(
            "The input must contain exactly one non-missing Attribute_group; "
            f"found {groups}."
        )

    attribute_group = groups[0]
    attribute = attribute_name or attribute_group

    # This particular curation format lists only positive observations.
    # For a table containing both states, supply explicit Attribute and
    # Attribute_value columns instead of adding True to every row.
    table["Attribute"] = attribute
    table["Attribute_value"] = True
    return table, attribute_group, attribute


def main() -> None:
    args = parse_args()
    raw, attribute_group, attribute = load_positive_binary_observations(
        args.observations,
        attribute_name=args.attribute_name,
    )

    taxonomy = FrozenTaxonomyProvider.from_tsv(load_attributes=True)
    resources = ltp()

    result = reconstruct_trait(
        raw,
        taxonomy,
        resources,
        allow_custom_vocabulary=True,
        min_annotated_tips=args.min_annotated_tips,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.raw.to_csv(args.output_dir / "00_raw.tsv", sep="\t", index=False)
    result.filtered.to_csv(
        args.output_dir / "01_filtered.tsv", sep="\t", index=False
    )
    if not isinstance(result.prepared, pd.DataFrame):
        raise TypeError("A binary trait should produce one prepared DataFrame.")
    result.prepared.to_csv(
        args.output_dir / "05_prepared.tsv", sep="\t", index=False
    )
    result.propagated.to_csv(
        args.output_dir / "99_propagated.tsv", sep="\t", index=False
    )
    result.coverage.to_csv(
        args.output_dir / "coverage.tsv", sep="\t", index=False
    )

    print(f"Trait: {result.attribute_group}")
    print(f"ASR run: {result.asr_run}")
    if not result.coverage.empty:
        print(result.coverage.to_string(index=False))
    print(f"Propagated rows: {len(result.propagated):,}")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
