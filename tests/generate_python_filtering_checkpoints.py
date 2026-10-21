"""Generate Python ``01_filtered.tsv`` files from frozen raw observations.

This is a diagnostic utility rather than a pytest assertion.  It reads each
frozen ``00_raw.tsv``, runs the public observation loader and ``filter_data``,
and writes the Python result to a separate checkpoint directory.  Frozen R
inputs and reference outputs are never modified.

Run from the repository root::

    python tests/generate_python_filtering_checkpoints.py

Use ``--case binary_example`` to generate one case.  Existing case output is
preserved unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# Permit direct execution from a src-layout checkout without requiring an
# editable installation in the selected interpreter.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bugphyzz_reconstruction.io.observations import (  # noqa: E402
    read_trait_observations,
)
from bugphyzz_reconstruction.preprocessing.filtering import (  # noqa: E402
    filter_data,
)
from bugphyzz_reconstruction.taxonomy.frozen import (  # noqa: E402
    FrozenTaxonomyProvider,
)


DESKTOP_ROOT = REPOSITORY_ROOT.parent
DEFAULT_INPUT_ROOT = DESKTOP_ROOT / "asr_python" / "frozen_r_outputs"
DEFAULT_OUTPUT_ROOT = (
    DESKTOP_ROOT / "asr_python" / "python_filtering_checkpoints"
)
DEFAULT_RESOURCE_DIR = (
    REPOSITORY_ROOT
    / "src"
    / "bugphyzz_reconstruction"
    / "resources"
    / "taxonomy_2023"
)


def prepare_case_output(path: Path, *, overwrite: bool) -> None:
    """Create an empty case destination without touching frozen inputs."""
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. "
                "Pass --overwrite to replace this Python checkpoint."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def export_case(
    raw_file: Path,
    output_root: Path,
    taxonomy: FrozenTaxonomyProvider,
    resource_dir: Path,
    *,
    overwrite: bool,
) -> Path:
    """Filter one frozen raw table and write its Python checkpoint."""
    case_name = raw_file.parent.name
    case_output_dir = output_root / case_name
    prepare_case_output(case_output_dir, overwrite=overwrite)

    raw = read_trait_observations(raw_file)
    filtered = filter_data(raw, taxonomy=taxonomy)
    if filtered is None:
        raise ValueError(
            f"Filtering produced no usable rows for {case_name!r}: {raw_file}"
        )

    output_file = case_output_dir / "01_filtered.tsv"
    filtered.to_csv(
        output_file,
        sep="\t",
        index=False,
        na_rep="NA",
    )

    metadata = {
        "implementation": "python",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "source_raw_file": str(raw_file.resolve()),
        "output_filtered_file": str(output_file.resolve()),
        "taxonomy_resource_dir": str(resource_dir.resolve()),
        "raw_rows": len(raw),
        "filtered_rows": len(filtered),
        "rows_dropped": len(raw) - len(filtered),
        "python_version": sys.version,
        "pandas_version": pd.__version__,
    }
    (case_output_dir / "generator_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Python 01_filtered.tsv files from frozen 00_raw.tsv "
            "observation tables."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing case directories with 00_raw.tsv files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Separate destination for generated Python checkpoints.",
    )
    parser.add_argument(
        "--resource-dir",
        type=Path,
        default=DEFAULT_RESOURCE_DIR,
        help="Frozen taxonomy resource directory containing attributes.tsv.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Case directory name to generate. Repeat for multiple cases; "
            "omit to process every case containing 00_raw.tsv."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Python output directories for selected cases.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    resource_dir = args.resource_dir.resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if not resource_dir.is_dir():
        raise FileNotFoundError(
            f"Taxonomy resource directory does not exist: {resource_dir}"
        )

    raw_files = sorted(input_root.glob("*/00_raw.tsv"))
    if args.case:
        selected = set(args.case)
        raw_files = [
            path for path in raw_files if path.parent.name in selected
        ]
        missing_cases = selected.difference(
            path.parent.name for path in raw_files
        )
        if missing_cases:
            raise FileNotFoundError(
                "No 00_raw.tsv was found for: "
                + ", ".join(sorted(missing_cases))
            )

    if not raw_files:
        raise FileNotFoundError(
            "No case directories containing 00_raw.tsv were found in "
            f"{input_root}."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    taxonomy = FrozenTaxonomyProvider.from_tsv(
        resource_dir=resource_dir,
        load_attributes=True,
    )

    exported = [
        export_case(
            raw_file,
            output_root,
            taxonomy,
            resource_dir,
            overwrite=args.overwrite,
        )
        for raw_file in raw_files
    ]

    print("Generated Python filtering checkpoints:")
    for output_file in exported:
        print(f"  {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
