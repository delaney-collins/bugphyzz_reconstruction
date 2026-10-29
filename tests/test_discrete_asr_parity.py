"""End-to-end parity tests for the discrete ancestral-state pipeline.

Unlike the focused ``ancr`` test, this suite deliberately reconstructs states
from the model fitted by :func:`run_discrete_asr_pipeline`.  It therefore tests
the complete Python path from a prepared trait table through final Bugphyzz
formatting, while retaining checkpoint-specific failure messages.
"""

from __future__ import annotations

from dataclasses import dataclass
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
)
from bugphyzz_reconstruction.asr.discrete_pipeline import (  # noqa: E402
    run_discrete_asr_pipeline,
)
from bugphyzz_reconstruction.asr.ltp import LTPResources, ltp  # noqa: E402


REFERENCE_ROOT = (
    REPOSITORY_ROOT / "r_reference_outputs" / "r_asr_pipeline_2023"
)
RESOURCE_DIR = (
    REPOSITORY_ROOT
    / "src"
    / "bugphyzz_reconstruction"
    / "resources"
    / "taxonomy_2023"
)
NA_VALUES = ["NA", "NaN", "nan", "<NA>", "NULL"]


@dataclass(frozen=True)
class DiscreteParityCase:
    name: str
    attribute_type: str
    annotated_tips: int
    coverage_percent: int
    reconstruction_rtol: float
    reconstruction_atol: float


CASES = (
    DiscreteParityCase(
        name="motility",
        attribute_type="binary",
        annotated_tips=8040,
        coverage_percent=34,
        reconstruction_rtol=2e-5,
        reconstruction_atol=2e-6,
    ),
    DiscreteParityCase(
        name="shape",
        attribute_type="multistate-intersection",
        annotated_tips=2050,
        coverage_percent=8,
        reconstruction_rtol=2e-5,
        reconstruction_atol=5e-6,
    ),
)


def read_checkpoint(
    path: Path,
    *,
    index_col: str | None = None,
) -> pd.DataFrame:
    """Read a frozen checkpoint while preserving identifier columns."""
    if not path.is_file():
        raise FileNotFoundError(f"ASR parity checkpoint does not exist: {path}")
    return pd.read_csv(
        path,
        sep="\t",
        dtype={
            "NCBI_ID": "string",
            "taxid": "string",
            "Attribute": "string",
            "Attribute_value": "string",
        },
        na_values=NA_VALUES,
        keep_default_na=True,
        low_memory=False,
        index_col=index_col,
    )


def read_model_scalar(case_dir: Path, parameter: str) -> str:
    """Read one value from the frozen model-scalar checkpoint."""
    scalars = read_checkpoint(case_dir / "11_model_scalars.tsv")
    values = scalars.loc[scalars["parameter"].eq(parameter), "value"]
    if len(values) != 1:
        raise AssertionError(
            f"Expected one {parameter!r} value in 11_model_scalars.tsv."
        )
    return str(values.iloc[0])


def assert_checkpoint_frame(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    checkpoint: str,
    numeric_columns: set[str] | None = None,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Compare a mixed-type checkpoint with useful stage-specific errors."""
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
    """Load the frozen LTP tree and mapping tables once."""
    if not RESOURCE_DIR.is_dir():
        pytest.fail(f"LTP resource directory does not exist: {RESOURCE_DIR}")
    return ltp(
        resource_dir=RESOURCE_DIR,
        remove_gn_nodes=True,
        node_names=True,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_discrete_asr_matches_frozen_r(
    resources: LTPResources,
    case: DiscreteParityCase,
) -> None:
    """Run the fitted Python model through every discrete parity checkpoint."""
    case_dir = REFERENCE_ROOT / case.name
    if not case_dir.is_dir():
        pytest.fail(f"ASR reference case does not exist: {case_dir}")

    prepared = read_checkpoint(case_dir / "00_prepared_input.tsv")
    tree_taxonomy = make_tree_taxonomy_table(
        resources.tip_data,
        resources.node_data,
    )
    result = run_discrete_asr_pipeline(
        resources.tree,
        resources.tip_data,
        tree_taxonomy,
        prepared,
        trait_name=case.name,
    )

    # Prepared tree input and coverage (checkpoints 04 and 08).
    assert result.prepared_input.coverage.iloc[0].to_dict() == {
        "trait": case.name,
        "annotated_tips": case.annotated_tips,
        "total_tips": 23501,
        "coverage_percent": case.coverage_percent,
        "rounding": "floor",
        "asr_run": True,
    }
    expected_input = read_checkpoint(
        case_dir / "08_input_matrix_tree_order.tsv",
        index_col="tip_label",
    )
    assert result.prepared_input.input_tree_order.index.tolist() == (
        expected_input.index.tolist()
    ), "08_input_matrix_tree_order: tip order differs"
    assert result.prepared_input.input_tree_order.columns.tolist() == (
        expected_input.columns.tolist()
    ), "08_input_matrix_tree_order: state order differs"
    np.testing.assert_allclose(
        result.prepared_input.input_tree_order.to_numpy(dtype=float),
        expected_input.to_numpy(dtype=float),
        rtol=1e-15,
        atol=1e-15,
        err_msg="08_input_matrix_tree_order: likelihood matrix differs",
    )

    # Fitted model (checkpoints 10-13). This is the same fitted model used by
    # the reconstruction below; no frozen R rate is injected into ancr().
    expected_rate = float(
        read_checkpoint(case_dir / "10_model_rates.tsv")["estimate"].iloc[0]
    )
    np.testing.assert_allclose(
        result.fit.rates,
        [expected_rate],
        rtol=5e-6,
        atol=2e-5,
        err_msg="10_model_rates: fitted ER rate differs",
    )
    np.testing.assert_allclose(
        result.fit.log_likelihood,
        float(read_model_scalar(case_dir, "logLik")),
        rtol=0.0,
        atol=1e-5,
        err_msg="11_model_scalars: fitted log likelihood differs",
    )
    assert result.fit.optimization.convergence == int(
        float(read_model_scalar(case_dir, "optimizer_convergence"))
    )

    expected_index = read_checkpoint(case_dir / "12_model_index_matrix.tsv")
    expected_index_values = expected_index.drop(columns="from_state").to_numpy(
        dtype=float
    )
    np.testing.assert_allclose(
        result.fit.index_matrix,
        expected_index_values,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
        err_msg="12_model_index_matrix: rate-index matrix differs",
    )
    expected_prior = read_checkpoint(case_dir / "13_model_root_prior.tsv")
    assert list(result.fit.states) == expected_prior["state"].astype(str).tolist()
    np.testing.assert_allclose(
        result.fit.pi,
        expected_prior["root_probability"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-15,
        err_msg="13_model_root_prior: root probabilities differ",
    )

    # Marginal reconstruction from the optimized Python fit (checkpoint 20).
    expected_reconstruction = read_checkpoint(
        case_dir / "20_ancr_all_tip_node_probabilities.tsv",
        index_col="label",
    )
    assert result.reconstruction.ace.index.tolist() == (
        expected_reconstruction.index.tolist()
    ), "20_ancr: reconstructed label order differs"
    assert result.reconstruction.ace.columns.tolist() == (
        expected_reconstruction.columns.tolist()
    ), "20_ancr: reconstructed state order differs"
    np.testing.assert_allclose(
        result.reconstruction.ace.to_numpy(dtype=float),
        expected_reconstruction.to_numpy(dtype=float),
        rtol=case.reconstruction_rtol,
        atol=case.reconstruction_atol,
        err_msg="20_ancr: fitted-model marginal probabilities differ",
    )

    state_columns = set(result.fit.states)
    assert_checkpoint_frame(
        result.mappable_unannotated,
        read_checkpoint(case_dir / "22_reconstructed_mappable_unannotated.tsv"),
        checkpoint="22_reconstructed_mappable_unannotated",
        numeric_columns=state_columns,
        rtol=case.reconstruction_rtol,
        atol=case.reconstruction_atol,
    )
    assert_checkpoint_frame(
        result.joined_to_taxonomy,
        read_checkpoint(case_dir / "23_reconstructed_joined_to_taxonomy.tsv"),
        checkpoint="23_reconstructed_joined_to_taxonomy",
        numeric_columns=state_columns,
        rtol=case.reconstruction_rtol,
        atol=case.reconstruction_atol,
    )
    assert_checkpoint_frame(
        result.deduplicated_by_taxonomy,
        read_checkpoint(case_dir / "24_reconstructed_deduplicated_by_ncbi.tsv"),
        checkpoint="24_reconstructed_deduplicated_by_ncbi",
        numeric_columns=state_columns,
        rtol=case.reconstruction_rtol,
        atol=case.reconstruction_atol,
    )
    assert_checkpoint_frame(
        result.predictions_long,
        read_checkpoint(case_dir / "25_asr_predictions_long.tsv"),
        checkpoint="25_asr_predictions_long",
        numeric_columns={"Score"},
        rtol=case.reconstruction_rtol,
        atol=case.reconstruction_atol,
    )

    # Shared final assembly (checkpoints 26, 27, and 99).
    observed = prepared.loc[prepared["Evidence"].notna()].copy()
    assert_checkpoint_frame(
        observed,
        read_checkpoint(case_dir / "26_observed_rows_retained.tsv"),
        checkpoint="26_observed_rows_retained",
        numeric_columns={"Score"},
        rtol=0.0,
        atol=1e-15,
    )
    combined = combine_observed_and_predictions(
        observed,
        result.predictions_long,
        attribute_group=case.name,
        attribute_type=case.attribute_type,
        predictions_first=True,
    )
    assert_checkpoint_frame(
        combined,
        read_checkpoint(case_dir / "27_combined_before_formatting.tsv"),
        checkpoint="27_combined_before_formatting",
        numeric_columns={"Score"},
        rtol=case.reconstruction_rtol,
        atol=case.reconstruction_atol,
    )
    final = format_final_asr_output(
        combined,
        attribute_group=case.name,
        attribute_type=case.attribute_type,
    )
    assert_checkpoint_frame(
        final,
        read_checkpoint(case_dir / "99_final_propagated.tsv"),
        checkpoint="99_final_propagated",
        numeric_columns={"Score"},
        rtol=case.reconstruction_rtol,
        atol=case.reconstruction_atol,
    )
