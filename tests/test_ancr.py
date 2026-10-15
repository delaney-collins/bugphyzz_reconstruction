"""Parity tests for the relevant ``phytools::ancr`` path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bugphyzz.trees.discrete import AncrResult, ancr, fit_mk
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


@pytest.mark.parametrize(
    ("case_name", "absolute_tolerance"),
    [
        ("motility", 1e-12),
        ("shape", 5e-12),
    ],
)
def test_ancr_global_marginals_match_frozen_r(
    tree,
    case_name: str,
    absolute_tolerance: float,
) -> None:
    case_dir = REFERENCE_DIR / case_name
    tip_likelihoods = pd.read_csv(
        case_dir / "08_input_matrix_tree_order.tsv",
        sep="\t",
        index_col="tip_label",
    )
    frozen_rate = float(
        pd.read_csv(
            case_dir / "10_model_rates.tsv",
            sep="\t",
        )["estimate"].iloc[0]
    )
    expected = pd.read_csv(
        case_dir / "20_ancr_all_tip_node_probabilities.tsv",
        sep="\t",
        index_col="label",
    )

    # Use the exact frozen rate to isolate ancr parity from harmless
    # differences between the R and Python optimizer stopping points.
    fit = fit_mk(
        tree,
        tip_likelihoods,
        model="ER",
        pi="equal",
        q_init=frozen_rate,
        opt_method="none",
    )
    result = ancr(fit, tips=True)

    assert isinstance(result, AncrResult)
    assert result.reconstruction_type == ("marginal", "global")
    assert result.ace.index.tolist() == expected.index.tolist()
    assert result.ace.columns.tolist() == expected.columns.tolist()

    np.testing.assert_allclose(
        result.ace.to_numpy(),
        expected.to_numpy(dtype=float),
        rtol=1e-11,
        atol=absolute_tolerance,
    )
    np.testing.assert_allclose(
        result.ace.sum(axis=1).to_numpy(),
        1.0,
        rtol=0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        result.log_likelihood,
        fit.log_likelihood,
        rtol=0,
        atol=1e-10,
    )


def test_ancr_without_tips_returns_only_internal_nodes(tree) -> None:
    case_dir = REFERENCE_DIR / "motility"
    tip_likelihoods = pd.read_csv(
        case_dir / "08_input_matrix_tree_order.tsv",
        sep="\t",
        index_col="tip_label",
    )
    frozen_rate = float(
        pd.read_csv(
            case_dir / "10_model_rates.tsv",
            sep="\t",
        )["estimate"].iloc[0]
    )
    fit = fit_mk(
        tree,
        tip_likelihoods,
        q_init=frozen_rate,
        opt_method="none",
    )

    result = ancr(fit, tips=False)
    expected_labels = [
        str(node.name)
        for node in tree.get_nonterminals(order="preorder")
    ]

    assert result.ace.index.tolist() == expected_labels
    assert len(result.ace) == len(expected_labels)
