"""Export Python discrete-ASR checkpoints from a prepared trait table.

This diagnostic mirrors the numbered files in ``asr_reference_2023`` without
reading from or writing to that frozen R reference directory.  By default it
uses the Python motility ``05_prepared.tsv`` produced by
``generate_python_preprocessing_checkpoints.py``.

Run from the repository root::

    python tests/generate_python_asr_checkpoints.py

Existing output is preserved unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


# Permit direct execution from a src-layout checkout without an editable
# installation in the selected interpreter.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bugphyzz_reconstruction.asr.asr_shared import (  # noqa: E402
    combine_observed_and_predictions,
    format_final_asr_output,
    make_tree_taxonomy_table,
)
from bugphyzz_reconstruction.asr.discrete_pipeline import (  # noqa: E402
    run_discrete_asr_pipeline,
)
from bugphyzz_reconstruction.asr.ltp import ltp  # noqa: E402


DESKTOP_ROOT = REPOSITORY_ROOT.parent
DEFAULT_PREPARED = (
    DESKTOP_ROOT
    / "asr_python"
    / "python_preprocessing_checkpoints"
    / "binary_example"
    / "05_prepared.tsv"
)
DEFAULT_OUTPUT_ROOT = (
    DESKTOP_ROOT / "asr_python" / "python_asr_checkpoints"
)
DEFAULT_RESOURCE_DIR = (
    REPOSITORY_ROOT
    / "src"
    / "bugphyzz_reconstruction"
    / "resources"
    / "taxonomy_2023"
)
NA_VALUES = ["NA", "NaN", "nan", "<NA>", "NULL"]


def read_prepared_table(path: Path) -> pd.DataFrame:
    """Read a prepared TSV while preserving taxonomy identifiers as text."""
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

    if "Score" in table:
        table["Score"] = pd.to_numeric(table["Score"], errors="coerce")
    return table


def single_metadata_value(table: pd.DataFrame, column: str) -> str:
    """Return one nonmissing metadata value from observed prepared rows."""
    if column not in table:
        raise ValueError(f"Prepared table is missing {column!r}.")
    values = (
        table[column]
        .astype("string")
        .dropna()
        .str.strip()
        .drop_duplicates()
        .tolist()
    )
    if len(values) != 1:
        raise ValueError(
            f"Expected one nonmissing {column!r}, but found {values!r}."
        )
    return str(values[0])


def observed_rows(prepared: pd.DataFrame) -> pd.DataFrame:
    """Return real observations, excluding completion-generated zero rows."""
    if "Evidence" not in prepared:
        raise ValueError("Prepared table is missing 'Evidence'.")
    return prepared.loc[prepared["Evidence"].notna()].copy()


def write_tsv(
    table: pd.DataFrame,
    path: Path,
    *,
    index_label: str | None = None,
) -> None:
    """Write a checkpoint TSV with explicit missing values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        path,
        sep="\t",
        index=index_label is not None,
        index_label=index_label,
        na_rep="NA",
    )


def prepare_output_directory(path: Path, *, overwrite: bool) -> None:
    """Create a clean, explicitly scoped diagnostic destination."""
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def matrix_frame(
    values: np.ndarray,
    states: tuple[str, ...],
    *,
    row_name: str,
) -> pd.DataFrame:
    """Create a labelled square state matrix for TSV export."""
    frame = pd.DataFrame(values, index=list(states), columns=list(states))
    frame.index.name = row_name
    return frame


def export_checkpoints(
    prepared_path: Path,
    output_dir: Path,
    resource_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """Run discrete ASR and export numbered diagnostic stages."""
    prepare_output_directory(output_dir, overwrite=overwrite)
    prepared = read_prepared_table(prepared_path)
    attribute_group = single_metadata_value(prepared, "Attribute_group")
    attribute_type = single_metadata_value(prepared, "Attribute_type")
    if attribute_type not in {"binary", "multistate-intersection"}:
        raise ValueError(
            "This diagnostic supports binary and multistate-intersection "
            f"traits, not {attribute_type!r}."
        )

    resources = ltp(
        resource_dir=resource_dir,
        remove_gn_nodes=True,
        node_names=True,
    )
    tree_taxonomy = make_tree_taxonomy_table(
        resources.tip_data,
        resources.node_data,
    )
    stages = run_discrete_asr_pipeline(
        resources.tree,
        resources.tip_data,
        tree_taxonomy,
        prepared,
        trait_name=attribute_group,
    )
    inputs = stages.prepared_input
    fit = stages.fit

    observed = observed_rows(prepared)
    combined = combine_observed_and_predictions(
        observed,
        stages.predictions_long,
        attribute_group=attribute_group,
        attribute_type=attribute_type,
        predictions_first=True,
    )
    final = format_final_asr_output(
        combined,
        attribute_group=attribute_group,
        attribute_type=attribute_type,
    )

    write_tsv(prepared, output_dir / "00_prepared_input.tsv")
    write_tsv(inputs.joined_tip_data, output_dir / "01_joined_tip_data.tsv")
    write_tsv(
        inputs.annotated_tips_long,
        output_dir / "02_annotated_tips_long.tsv",
    )
    write_tsv(
        inputs.annotated_tip_state_matrix,
        output_dir / "03_annotated_tip_state_matrix.tsv",
        index_label="tip_label",
    )
    write_tsv(inputs.coverage, output_dir / "04_coverage.tsv")
    write_tsv(
        inputs.unannotated_uniform_state_matrix,
        output_dir / "05_unannotated_uniform_state_matrix.tsv",
        index_label="tip_label",
    )
    write_tsv(
        inputs.input_before_tree_order,
        output_dir / "06_input_matrix_before_tree_order.tsv",
        index_label="tip_label",
    )

    validation = pd.DataFrame(
        [
            {
                "tree_tip_count": len(resources.tree.get_terminals()),
                "matrix_row_count": len(inputs.input_tree_order),
                "state_count": len(inputs.input_tree_order.columns),
                "unique_tip_labels": inputs.input_tree_order.index.is_unique,
                "all_values_finite": bool(
                    np.isfinite(
                        inputs.input_tree_order.to_numpy(dtype=float)
                    ).all()
                ),
                "all_row_sums_positive": bool(
                    (
                        inputs.input_tree_order.sum(axis=1).to_numpy()
                        > 0
                    ).all()
                ),
            }
        ]
    )
    write_tsv(validation, output_dir / "07_input_validation.tsv")
    write_tsv(
        inputs.input_tree_order,
        output_dir / "08_input_matrix_tree_order.tsv",
        index_label="tip_label",
    )

    rates = pd.DataFrame(
        {
            "rate_index": np.arange(1, len(fit.rates) + 1),
            "estimate": fit.rates,
        }
    )
    write_tsv(rates, output_dir / "10_model_rates.tsv")
    scalars = pd.DataFrame(
        {
            "parameter": [
                "logLik",
                "convergence",
                "iterations",
                "evaluations",
            ],
            "value": [
                fit.log_likelihood,
                fit.optimization.convergence,
                fit.optimization.iterations,
                fit.optimization.evaluations,
            ],
        }
    )
    write_tsv(scalars, output_dir / "11_model_scalars.tsv")
    write_tsv(
        matrix_frame(
            fit.index_matrix,
            fit.states,
            row_name="state",
        ),
        output_dir / "12_model_index_matrix.tsv",
        index_label="state",
    )
    root_prior = pd.DataFrame(
        {
            "state": fit.states,
            "root_probability": fit.pi,
        }
    )
    write_tsv(root_prior, output_dir / "13_model_root_prior.tsv")
    write_tsv(
        matrix_frame(
            fit.q_matrix,
            fit.states,
            row_name="state",
        ),
        output_dir / "14_model_q_matrix.tsv",
        index_label="state",
    )

    write_tsv(
        stages.reconstruction.ace,
        output_dir / "20_ancr_all_tip_node_probabilities.tsv",
        index_label="label",
    )
    reconstruction_summary = pd.DataFrame(
        [
            {
                "reconstruction_type": "/".join(
                    stages.reconstruction.reconstruction_type
                ),
                "log_likelihood": stages.reconstruction.log_likelihood,
                "row_count": len(stages.reconstruction.ace),
                "state_count": len(stages.reconstruction.ace.columns),
                "maximum_row_sum_error": float(
                    (
                        stages.reconstruction.ace.sum(axis=1) - 1.0
                    ).abs().max()
                ),
            }
        ]
    )
    write_tsv(
        reconstruction_summary,
        output_dir / "21_ancr_summary.tsv",
    )
    write_tsv(
        stages.mappable_unannotated,
        output_dir / "22_reconstructed_mappable_unannotated.tsv",
    )
    write_tsv(
        stages.joined_to_taxonomy,
        output_dir / "23_reconstructed_joined_to_taxonomy.tsv",
    )
    write_tsv(
        stages.deduplicated_by_taxonomy,
        output_dir / "24_reconstructed_deduplicated_by_ncbi.tsv",
    )
    write_tsv(
        stages.predictions_long,
        output_dir / "25_asr_predictions_long.tsv",
    )
    write_tsv(observed, output_dir / "26_observed_rows_retained.tsv")
    write_tsv(combined, output_dir / "27_combined_before_formatting.tsv")
    write_tsv(final, output_dir / "99_final_propagated.tsv")

    metadata = {
        "implementation": "python",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "prepared_input": str(prepared_path.resolve()),
        "taxonomy_resource_dir": str(resource_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "attribute_group": attribute_group,
        "attribute_type": attribute_type,
        "model": "ER",
        "root_prior": "equal",
        "optimizer": fit.method,
        "state_order": list(fit.states),
        "python_version": sys.version,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    (output_dir / "00_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Python discrete-ASR checkpoints from a 05_prepared TSV."
        )
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=DEFAULT_PREPARED,
        help="Python 05_prepared.tsv to propagate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory. By default, use "
            "python_asr_checkpoints/<Attribute_group>."
        ),
    )
    parser.add_argument(
        "--resource-dir",
        type=Path,
        default=DEFAULT_RESOURCE_DIR,
        help="Directory containing the frozen 2023 LTP resources.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the selected Python diagnostic output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prepared_path = args.prepared.resolve()
    resource_dir = args.resource_dir.resolve()
    if not prepared_path.is_file():
        raise FileNotFoundError(
            f"Prepared input does not exist: {prepared_path}"
        )
    if not resource_dir.is_dir():
        raise FileNotFoundError(
            f"LTP resource directory does not exist: {resource_dir}"
        )

    prepared = read_prepared_table(prepared_path)
    attribute_group = single_metadata_value(prepared, "Attribute_group")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (DEFAULT_OUTPUT_ROOT / attribute_group.replace(" ", "_")).resolve()
    )
    export_checkpoints(
        prepared_path,
        output_dir,
        resource_dir,
        overwrite=args.overwrite,
    )
    print(f"Generated Python ASR checkpoints: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
