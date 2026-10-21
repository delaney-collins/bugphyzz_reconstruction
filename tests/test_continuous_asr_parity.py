"""End-to-end parity test for continuous ancestral-state reconstruction.

The test follows the frozen growth-temperature checkpoints from prepared input
through squared-change parsimony, NSTI annotation, taxonomy mapping, and final
Bugphyzz formatting.  The Python reconstruction is used throughout; frozen R
intermediate values are comparison targets only.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bugphyzz_reconstruction.asr.asr_shared import (  # noqa: E402
    combine_observed_and_predictions,
    format_final_asr_output,
    make_tree_taxonomy_table,
    prepare_mapped_predictions,
)
from bugphyzz_reconstruction.asr.continuous import (  # noqa: E402
    add_continuous_prediction_evidence,
    attach_continuous_nsti,
    calculate_continuous_nsti,
    prepare_continuous_asr_input,
    reconstruct_continuous_states,
    select_mappable_unannotated_states,
)
from bugphyzz_reconstruction.asr.ltp import LTPResources, ltp  # noqa: E402


REFERENCE_ROOT = (
    REPOSITORY_ROOT.parent / "asr_python" / "asr_reference_2023"
)
CASE_DIR = REFERENCE_ROOT / "growth_temperature"
TREE_REFERENCE_DIR = REFERENCE_ROOT / "_tree_reference"
RESOURCE_DIR = (
    REPOSITORY_ROOT
    / "src"
    / "bugphyzz_reconstruction"
    / "resources"
    / "taxonomy_2023"
)
NA_VALUES = ["NA", "NaN", "nan", "<NA>", "NULL"]


def read_checkpoint(path: Path) -> pd.DataFrame:
    """Read a frozen TSV while preserving identifiers and trait labels."""
    if not path.is_file():
        raise FileNotFoundError(f"ASR parity checkpoint does not exist: {path}")
    return pd.read_csv(
        path,
        sep="\t",
        dtype={
            "NCBI_ID": "string",
            "taxid": "string",
            "Attribute": "string",
            "Attribute_value": "float64",
        },
        na_values=NA_VALUES,
        keep_default_na=True,
        low_memory=False,
    )


def read_nonempty_lines(path: Path) -> list[str]:
    """Read the frozen ordered node-label list used by the R NSTI call."""
    if not path.is_file():
        raise FileNotFoundError(f"Tree reference does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        return [
            line.rstrip("\r\n")
            for line in handle
            if line.rstrip("\r\n")
        ]


def assert_checkpoint_frame(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    checkpoint: str,
    numeric_columns: set[str] | None = None,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Compare one mixed-type checkpoint with stage-specific diagnostics."""
    numeric_columns = numeric_columns or set()
    assert actual.columns.tolist() == expected.columns.tolist(), (
        f"{checkpoint}: columns differ.\n"
        f"Python: {actual.columns.tolist()}\n"
        f"R:      {expected.columns.tolist()}"
    )
    assert len(actual) == len(expected), (
        f"{checkpoint}: row counts differ; Python={len(actual):,}, "
        f"R={len(expected):,}."
    )

    left = actual.reset_index(drop=True)
    right = expected.reset_index(drop=True)
    for column in left.columns:
        if column in numeric_columns:
            np.testing.assert_allclose(
                pd.to_numeric(left[column], errors="coerce"),
                pd.to_numeric(right[column], errors="coerce"),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
                err_msg=f"{checkpoint}: numeric column {column!r} differs",
            )
        else:
            assert_series_equal(
                left[column].astype("string"),
                right[column].astype("string"),
                check_names=False,
                check_dtype=False,
                obj=f"{checkpoint}: column {column!r}",
            )


@pytest.fixture(scope="module")
def resources() -> LTPResources:
    """Load the frozen LTP resources once for the parity test."""
    if not RESOURCE_DIR.is_dir():
        pytest.fail(f"LTP resource directory does not exist: {RESOURCE_DIR}")
    return ltp(
        resource_dir=RESOURCE_DIR,
        remove_gn_nodes=True,
        node_names=True,
    )


def test_continuous_asr_matches_frozen_r(resources: LTPResources) -> None:
    """Run and compare the complete growth-temperature ASR branch."""
    if not CASE_DIR.is_dir():
        pytest.fail(f"ASR reference case does not exist: {CASE_DIR}")

    prepared = read_checkpoint(CASE_DIR / "00_prepared_input.tsv")
    inputs = prepare_continuous_asr_input(
        resources.tree,
        resources.tip_data,
        prepared,
        trait_name="growth_temperature",
    )

    assert inputs.coverage.iloc[0].to_dict() == {
        "trait": "growth_temperature",
        "annotated_tips": 14460,
        "total_tips": 23501,
        "coverage_percent": 62,
        "rounding": "round",
        "asr_run": True,
    }
    assert_checkpoint_frame(
        inputs.input_tree_order,
        read_checkpoint(CASE_DIR / "07_input_vector_tree_order.tsv"),
        checkpoint="07_input_vector_tree_order",
        numeric_columns={"Attribute_value"},
        rtol=0.0,
        atol=0.0,
    )

    reconstruction = reconstruct_continuous_states(
        resources.tree,
        inputs,
        weighted=True,
    )
    assert reconstruction.fit.success
    assert np.isfinite(reconstruction.fit.total_sum_of_squared_changes)
    assert_checkpoint_frame(
        reconstruction.states,
        read_checkpoint(CASE_DIR / "09_all_reconstructed_states.tsv"),
        checkpoint="09_all_reconstructed_states",
        numeric_columns={"Attribute_value"},
        rtol=1e-10,
        atol=1e-9,
    )

    tree_taxonomy = make_tree_taxonomy_table(
        resources.tip_data,
        resources.node_data,
    )
    mappable = select_mappable_unannotated_states(
        reconstruction.states,
        mappable_labels=tree_taxonomy["label"],
        annotated_tip_labels=inputs.annotated_tip_labels,
    )
    assert_checkpoint_frame(
        mappable,
        read_checkpoint(CASE_DIR / "10_mappable_unannotated_states.tsv"),
        checkpoint="10_mappable_unannotated_states",
        numeric_columns={"Attribute_value"},
        rtol=1e-10,
        atol=1e-9,
    )

    # Reproduce the node set used by the frozen R pipeline's getNsti call.
    nsti = calculate_continuous_nsti(
        resources.tree,
        annotated_tip_labels=inputs.annotated_tip_labels,
        node_labels=read_nonempty_lines(
            TREE_REFERENCE_DIR / "nodes_with_taxid.txt"
        ),
    )
    assert_checkpoint_frame(
        nsti,
        read_checkpoint(CASE_DIR / "11_nsti.tsv"),
        checkpoint="11_nsti",
        numeric_columns={"nsti"},
        rtol=0.0,
        # The historical pipeline checkpoint was printed to six decimals.
        atol=5e-7,
    )

    with_nsti = attach_continuous_nsti(mappable, nsti)
    assert_checkpoint_frame(
        with_nsti,
        read_checkpoint(CASE_DIR / "12_reconstructed_states_with_nsti.tsv"),
        checkpoint="12_reconstructed_states_with_nsti",
        numeric_columns={"Attribute_value", "nsti"},
        rtol=1e-10,
        atol=5e-7,
    )

    joined = with_nsti.merge(
        tree_taxonomy,
        how="left",
        on="label",
        sort=False,
        validate="many_to_many",
    )
    assert_checkpoint_frame(
        joined,
        read_checkpoint(CASE_DIR / "13_reconstructed_joined_to_taxonomy.tsv"),
        checkpoint="13_reconstructed_joined_to_taxonomy",
        numeric_columns={"Attribute_value", "nsti"},
        rtol=1e-10,
        atol=5e-7,
    )

    predictions = prepare_mapped_predictions(
        add_continuous_prediction_evidence(with_nsti),
        tree_taxonomy,
        prepared,
        deduplicate_by="Attribute_value",
    )
    expected_predictions = read_checkpoint(
        CASE_DIR / "14_final_asr_predictions.tsv"
    )
    # Python retains the tree label as useful internal provenance. The R
    # checkpoint drops it and selects its final column order at this stage.
    predictions_checkpoint = predictions.drop(
        columns="label",
        errors="ignore",
    ).loc[:, expected_predictions.columns]
    assert_checkpoint_frame(
        predictions_checkpoint,
        expected_predictions,
        checkpoint="14_final_asr_predictions",
        numeric_columns={"Attribute_value", "nsti", "Score"},
        rtol=1e-10,
        atol=5e-7,
    )

    observed = prepared.loc[prepared["Evidence"].notna()].copy()
    assert_checkpoint_frame(
        observed,
        read_checkpoint(CASE_DIR / "15_observed_rows_retained.tsv"),
        checkpoint="15_observed_rows_retained",
        numeric_columns={"Attribute_value", "Score"},
        rtol=0.0,
        atol=1e-15,
    )
    combined = combine_observed_and_predictions(
        observed,
        predictions,
        attribute_group="growth temperature",
        attribute_type="range",
    )
    final = format_final_asr_output(
        combined,
        attribute_group="growth temperature",
        attribute_type="range",
    )
    assert_checkpoint_frame(
        final,
        read_checkpoint(CASE_DIR / "99_final_propagated.tsv"),
        checkpoint="99_final_propagated",
        numeric_columns={"Attribute_value", "nsti", "Score"},
        rtol=1e-10,
        atol=5e-7,
    )
