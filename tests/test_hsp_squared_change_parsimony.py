"""Parity tests for continuous hidden-state prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bugphyzz.trees.ltp import (
    HSPResult,
    hsp_squared_change_parsimony,
    ltp,
)


PACKAGE_DIR = Path(__file__).resolve().parents[1]
RESOURCE_DIR = PACKAGE_DIR / "resources"
REFERENCE_DIR = PACKAGE_DIR / "asr_reference_2023" / "growth_temperature"


@pytest.fixture(scope="module")
def parity_data() -> tuple[object, pd.DataFrame, pd.DataFrame]:
    resources = ltp(
        resource_dir=RESOURCE_DIR,
        remove_gn_nodes=True,
        node_names=True,
    )
    input_values = pd.read_csv(
        REFERENCE_DIR / "07_input_vector_tree_order.tsv",
        sep="\t",
        dtype={"tip_label": "string", "Attribute_value": "float64"},
        na_values=["NA"],
    )
    expected = pd.read_csv(
        REFERENCE_DIR / "09_all_reconstructed_states.tsv",
        sep="\t",
        dtype={
            "label": "string",
            "Attribute_value": "float64",
            "location": "string",
        },
    )
    return resources.tree, input_values, expected


def test_hsp_squared_change_parsimony_matches_frozen_r_states(
    parity_data: tuple[object, pd.DataFrame, pd.DataFrame],
) -> None:
    tree, input_values, expected = parity_data
    named_tip_states = pd.Series(
        input_values["Attribute_value"].to_numpy(),
        index=input_values["tip_label"],
        dtype="float64",
    )

    result = hsp_squared_change_parsimony(
        tree,
        named_tip_states,
        weighted=True,
        check_input=True,
    )

    assert isinstance(result, HSPResult)
    assert result.success

    expected_labels = [
        str(tip.name)
        for tip in tree.get_terminals(order="preorder")
    ] + [
        str(node.name)
        for node in tree.get_nonterminals(order="preorder")
    ]
    assert expected["label"].tolist() == expected_labels
    assert len(result.states) == len(expected)

    np.testing.assert_allclose(
        result.states,
        expected["Attribute_value"].to_numpy(dtype=float),
        rtol=1e-10,
        atol=1e-9,
        equal_nan=False,
    )


def test_hsp_preserves_observed_tip_states_and_returns_finite_objective(
    parity_data: tuple[object, pd.DataFrame, pd.DataFrame],
) -> None:
    tree, input_values, _ = parity_data
    input_array = input_values["Attribute_value"].to_numpy(dtype=float)

    result = hsp_squared_change_parsimony(
        tree,
        input_array,
        weighted=True,
    )

    observed = np.isfinite(input_array)
    np.testing.assert_array_equal(result.states[: len(input_array)][observed], input_array[observed])
    assert np.isfinite(result.total_sum_of_squared_changes)
    assert result.total_sum_of_squared_changes >= 0


def test_hsp_rejects_an_all_hidden_vector(
    parity_data: tuple[object, pd.DataFrame, pd.DataFrame],
) -> None:
    tree, input_values, _ = parity_data

    with pytest.raises(ValueError, match="All tip states are hidden"):
        hsp_squared_change_parsimony(
            tree,
            np.full(len(input_values), np.nan),
        )
