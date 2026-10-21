"""Export Python preprocessing checkpoints for comparison with frozen R TSVs.

This is a diagnostic utility, not a pytest test.  It reads each frozen
``01_filtered.tsv`` table, reruns the Python taxonomy/preparation stages, and
writes provenance-labelled Python outputs without modifying the R fixtures.

Run from the repository root after installing the package in the active
environment::

    python tests/generate_python_preprocessing_checkpoints.py

Use ``--case binary_example`` to export only one frozen case.  Existing case
directories are preserved unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

# Allow this diagnostic to be executed directly from a source checkout without
# requiring an editable installation first.  Opening the repository in VS Code
# does not add its src directory to the interpreter's import path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bugphyzz_reconstruction.preprocessing.completion import (
    complete_binary_data,
)
from bugphyzz_reconstruction.preprocessing.pipeline import prepare_trait_data
from bugphyzz_reconstruction.preprocessing.taxonomy_sets import (
    get_set_with_ids,
    get_set_without_ids,
)
from bugphyzz_reconstruction.taxonomy.frozen import FrozenTaxonomyProvider


DESKTOP_ROOT = REPOSITORY_ROOT.parent
DEFAULT_INPUT_ROOT = DESKTOP_ROOT / "asr_python" / "frozen_r_outputs"
DEFAULT_OUTPUT_ROOT = (
    DESKTOP_ROOT / "asr_python" / "python_preprocessing_checkpoints"
)
DEFAULT_RESOURCE_DIR = (
    REPOSITORY_ROOT
    / "src"
    / "bugphyzz_reconstruction"
    / "resources"
    / "taxonomy_2023"
)

NA_VALUES = ["NA", "NaN", "nan", "<NA>", "NULL"]


def read_filtered_table(path: Path) -> pd.DataFrame:
    """Read an R checkpoint while preserving taxonomy identifiers as text."""
    table = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        na_values=NA_VALUES,
        keep_default_na=True,
    )

    row_name_columns = [
        column
        for column in table.columns
        if str(column).startswith("Unnamed:")
        or str(column) in {"row.names", "X"}
    ]
    if row_name_columns:
        table = table.drop(columns=row_name_columns)

    return table


def drop_all_missing_columns(
    table: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Apply the same all-missing-column cleanup as the Python pipeline."""
    if table is None:
        return None
    return table.dropna(axis="columns", how="all")


def combine_tables(
    *tables: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine nonempty tables using the pipeline's bind-rows semantics."""
    usable = [
        table
        for table in tables
        if table is not None and not table.empty
    ]
    if not usable:
        return pd.DataFrame()
    return pd.concat(usable, ignore_index=True, sort=False)


def single_attribute_type(table: pd.DataFrame) -> str:
    """Return the normalized attribute type for one filtered trait table."""
    values = (
        table["Attribute_type"]
        .astype("string")
        .dropna()
        .str.strip()
        .str.lower()
        .replace({"mltistate-union": "multistate-union"})
        .drop_duplicates()
        .tolist()
    )
    if len(values) != 1:
        raise ValueError(
            "Expected one Attribute_type, but found "
            f"{values!r}."
        )
    return str(values[0])


def ordinary_stages(
    filtered: pd.DataFrame,
    taxonomy: FrozenTaxonomyProvider,
) -> dict[str, pd.DataFrame | None]:
    """Build stages 02-05 for binary, intersection, and range traits."""
    with_ids = drop_all_missing_columns(
        get_set_with_ids(filtered, taxonomy)
    )
    without_ids = drop_all_missing_columns(
        get_set_without_ids(
            filtered,
            taxonomy,
            set_with_ids=with_ids,
        )
    )
    combined = combine_tables(with_ids, without_ids)
    prepared = prepare_trait_data(filtered, taxonomy)

    if isinstance(prepared, dict):
        raise TypeError(
            "An ordinary trait unexpectedly produced multiple prepared tables."
        )

    return {
        "02_with_ids": with_ids,
        "03_without_ids": without_ids,
        "04_combined_before_completion": combined,
        "05_prepared": prepared,
    }


def union_stages(
    filtered: pd.DataFrame,
    taxonomy: FrozenTaxonomyProvider,
) -> dict[str, Mapping[str, pd.DataFrame]]:
    """Build per-component stages 02-05 for a multistate-union trait."""
    working = filtered.copy()
    working["Attribute_group_2"] = (
        working["Attribute"]
        .astype("string")
        .str.replace(r"--(?:TRUE|FALSE)$", "", regex=True)
    )

    with_ids_tables: dict[str, pd.DataFrame] = {}
    without_ids_tables: dict[str, pd.DataFrame] = {}
    combined_tables: dict[str, pd.DataFrame] = {}
    prepared_tables: dict[str, pd.DataFrame] = {}

    for component in (
        working["Attribute_group_2"].dropna().drop_duplicates().tolist()
    ):
        subset = (
            working.loc[working["Attribute_group_2"].eq(component)]
            .drop(columns="Attribute_group_2")
            .copy()
        )
        with_ids = drop_all_missing_columns(
            get_set_with_ids(subset, taxonomy)
        )
        without_ids = drop_all_missing_columns(
            get_set_without_ids(
                subset,
                taxonomy,
                set_with_ids=with_ids,
            )
        )
        combined = combine_tables(with_ids, without_ids)

        component_name = str(component)
        if with_ids is not None:
            with_ids_tables[component_name] = with_ids
        if without_ids is not None:
            without_ids_tables[component_name] = without_ids
        combined_tables[component_name] = combined
        if not combined.empty:
            prepared_tables[component_name] = complete_binary_data(combined)

    public_prepared = prepare_trait_data(filtered, taxonomy)
    if public_prepared is None:
        public_prepared = {}
    if not isinstance(public_prepared, dict):
        raise TypeError(
            "A multistate-union trait did not produce component tables."
        )

    # Use the public pipeline result for stage 05.  The independently assembled
    # component tables above make stage 04 available for diagnosis.
    prepared_tables = public_prepared

    return {
        "02_with_ids": with_ids_tables,
        "03_without_ids": without_ids_tables,
        "04_combined_before_completion": combined_tables,
        "05_prepared": prepared_tables,
    }


def safe_filename(value: str) -> str:
    """Return a stable filesystem-safe component label."""
    cleaned = "".join(
        character.lower() if character.isalnum() else "_"
        for character in value
    )
    return "_".join(part for part in cleaned.split("_") if part) or "item"


def write_table(table: pd.DataFrame | None, path: Path) -> None:
    """Write one checkpoint with R-compatible explicit missing values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if table is None:
        path.write_text("# NULL\n", encoding="utf-8")
        return
    table.to_csv(path, sep="\t", index=False, na_rep="NA")


def write_stage(
    value: pd.DataFrame | None | Mapping[str, pd.DataFrame],
    stage_name: str,
    case_output_dir: Path,
) -> None:
    """Write a scalar stage or a multistate-union component directory."""
    if not isinstance(value, Mapping):
        write_table(value, case_output_dir / f"{stage_name}.tsv")
        return

    stage_dir = case_output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for number, (component, table) in enumerate(value.items(), start=1):
        filename = f"{number:03d}_{safe_filename(component)}.tsv"
        write_table(table, stage_dir / filename)
        manifest_rows.append(
            {
                "item_number": number,
                "item_name": component,
                "filename": filename,
                "n_rows": len(table),
                "n_columns": len(table.columns),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(
        stage_dir / "_manifest.tsv",
        sep="\t",
        index=False,
        na_rep="NA",
    )


def prepare_case_output(case_output_dir: Path, *, overwrite: bool) -> None:
    """Create an empty, case-scoped destination without touching R inputs."""
    if case_output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {case_output_dir}. "
                "Pass --overwrite to replace this Python checkpoint directory."
            )
        shutil.rmtree(case_output_dir)
    case_output_dir.mkdir(parents=True)


def export_case(
    filtered_file: Path,
    output_root: Path,
    taxonomy: FrozenTaxonomyProvider,
    resource_dir: Path,
    *,
    overwrite: bool,
) -> Path:
    """Export all Python preprocessing checkpoints for one frozen R case."""
    case_name = filtered_file.parent.name
    case_output_dir = output_root / case_name
    prepare_case_output(case_output_dir, overwrite=overwrite)

    filtered = read_filtered_table(filtered_file)
    attribute_type = single_attribute_type(filtered)
    if attribute_type == "multistate-union":
        stages = union_stages(filtered, taxonomy)
    else:
        stages = ordinary_stages(filtered, taxonomy)

    for stage_name, value in stages.items():
        write_stage(value, stage_name, case_output_dir)

    metadata = {
        "implementation": "python",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "source_filtered_file": str(filtered_file.resolve()),
        "taxonomy_resource_dir": str(resource_dir.resolve()),
        "attribute_type": attribute_type,
        "python_version": sys.version,
        "pandas_version": pd.__version__,
    }
    (case_output_dir / "generator_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return case_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Python stages 02-05 from frozen R 01_filtered.tsv files."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Directory containing frozen R case directories.",
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
        help="Frozen taxonomy_2023 resource directory.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Case directory name to export. Repeat for multiple cases; "
            "omit to export every case containing 01_filtered.tsv."
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

    filtered_files = sorted(input_root.glob("*/01_filtered.tsv"))
    if args.case:
        selected = set(args.case)
        filtered_files = [
            path for path in filtered_files if path.parent.name in selected
        ]
        missing_cases = selected.difference(
            path.parent.name for path in filtered_files
        )
        if missing_cases:
            raise FileNotFoundError(
                "No 01_filtered.tsv was found for: "
                + ", ".join(sorted(missing_cases))
            )

    if not filtered_files:
        raise FileNotFoundError(
            f"No case directories containing 01_filtered.tsv were found in "
            f"{input_root}."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    taxonomy = FrozenTaxonomyProvider.from_tsv(
        resource_dir=resource_dir,
        load_attributes=True,
    )

    exported = [
        export_case(
            filtered_file,
            output_root,
            taxonomy,
            resource_dir,
            overwrite=args.overwrite,
        )
        for filtered_file in filtered_files
    ]

    print("Generated Python preprocessing checkpoints:")
    for directory in exported:
        print(f"  {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
