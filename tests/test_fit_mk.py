"""Parity tests for the phytools ``fitMk(..., model='ER')`` port."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bugphyzz.trees.discrete import fit_mk
from bugphyzz.trees.ltp import ltp


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PACKAGE_DIR / "asr_reference_2023"
RESOURCE_DIR = PACKAGE_DIR / "resources"


@pytest.fixture(scope="module")
def tree():
    return ltp(
        resource_dir=RESOURCE_DIR,
        remove_gn_nodes=True,
        node_names=True,
    ).tree


def _read_scalar(case_dir: Path, parameter: str) -> float:
    scalars = pd.read_csv(case_dir / "11_model_scalars.tsv", sep="\t")
    values = scalars.loc[scalars["parameter"].eq(parameter), "value"]
    if len(values) != 1:
        raise AssertionError(
            f"Expected exactly one {parameter!r} in 11_model_scalars.tsv."
        )
    return float(values.iloc[0])


@pytest.mark.parametrize("case_name", ["motility", "shape"])
def test_fit_mk_er_matches_frozen_phytools_fit(
    tree,
    case_name: str,
) -> None:
    case_dir = REFERENCE_DIR / case_name
    tip_likelihoods = pd.read_csv(
        case_dir / "08_input_matrix_tree_order.tsv",
        sep="\t",
        index_col="tip_label",
    )
    expected_rate = float(
        pd.read_csv(
            case_dir / "10_model_rates.tsv",
            sep="\t",
        )["estimate"].iloc[0]
    )
    expected_log_likelihood = _read_scalar(case_dir, "logLik")
    expected_prior = pd.read_csv(
        case_dir / "13_model_root_prior.tsv",
        sep="\t",
    )

    result = fit_mk(
        tree,
        tip_likelihoods,
        model="ER",
        pi="equal",
    )

    assert result.optimization.convergence == 0
    assert result.root_prior == "flat"
    assert result.states == tuple(tip_likelihoods.columns)
    np.testing.assert_allclose(
        result.rates,
        [expected_rate],
        rtol=5e-6,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        result.log_likelihood,
        expected_log_likelihood,
        rtol=0,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        result.pi,
        expected_prior["root_probability"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-15,
    )

    state_count = tip_likelihoods.shape[1]
    assert result.index_matrix.shape == (state_count, state_count)
    assert np.isnan(np.diag(result.index_matrix)).all()
    assert np.equal(
        result.index_matrix[~np.eye(state_count, dtype=bool)],
        1,
    ).all()

    expected_q = np.full((state_count, state_count), expected_rate)
    np.fill_diagonal(expected_q, -(state_count - 1) * expected_rate)
    np.testing.assert_allclose(
        result.q_matrix,
        expected_q,
        rtol=5e-6,
        atol=2e-5,
    )


def test_fit_mk_rejects_unimplemented_rate_models(tree) -> None:
    case_dir = REFERENCE_DIR / "motility"
    tip_likelihoods = pd.read_csv(
        case_dir / "08_input_matrix_tree_order.tsv",
        sep="\t",
        index_col="tip_label",
    )

    with pytest.raises(NotImplementedError, match="model='ER'"):
        fit_mk(tree, tip_likelihoods, model="SYM")
