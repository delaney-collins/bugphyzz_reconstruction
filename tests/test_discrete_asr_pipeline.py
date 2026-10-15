"""Checkpoint parity tests for the discrete-specific ASR pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bugphyzz.trees.asr_shared import (
    deduplicate_taxonomy_predictions,
    join_reconstruction_to_taxonomy,
    make_tree_taxonomy_table,
)
from bugphyzz.trees.discrete_pipeline import (
    make_discrete_long_predictions,
    prepare_discrete_asr_input,
    select_discrete_mappable_unannotated,
)
from bugphyzz.trees.ltp import ltp


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PACKAGE_DIR / "asr_reference_2023"
RESOURCE_DIR = PACKAGE_DIR / "resources"


@pytest.fixture(scope="module")
def resources():
    return ltp(
        resource_dir=RESOURCE_DIR,
        remove_gn_nodes=True,
        node_names=True,
    )


@pytest.mark.parametrize(
    ("case_name", "annotated_count", "coverage_percent"),
    [
        ("motility", 8040, 34),
        ("shape", 2050, 8),
    ],
)
def test_discrete_input_matrix_matches_r(
    resources,
    case_name: str,
    annotated_count: int,
    coverage_percent: int,
) -> None:
    case_dir = REFERENCE_DIR / case_name
    prepared = pd.read_csv(
        case_dir / "00_prepared_input.tsv",
        sep="\t",
        dtype={"NCBI_ID": "string", "taxid": "string"},
        na_values=["NA"],
    )
    result = prepare_discrete_asr_input(
        resources.tree,
        resources.tip_data,
        prepared,
        trait_name=case_name,
    )
    expected = pd.read_csv(
        case_dir / "08_input_matrix_tree_order.tsv",
        sep="\t",
        index_col="tip_label",
    )

    assert result.input_tree_order.index.tolist() == expected.index.tolist()
    assert result.input_tree_order.columns.tolist() == expected.columns.tolist()
    np.testing.assert_allclose(
        result.input_tree_order.to_numpy(),
        expected.to_numpy(),
        rtol=1e-15,
        atol=1e-15,
    )
    assert result.coverage.iloc[0].to_dict() == {
        "trait": case_name,
        "annotated_tips": annotated_count,
        "total_tips": 23501,
        "coverage_percent": coverage_percent,
        "rounding": "floor",
        "asr_run": True,
    }


@pytest.mark.parametrize("case_name", ["motility", "shape"])
def test_discrete_postprocessing_matches_r(
    resources,
    case_name: str,
) -> None:
    case_dir = REFERENCE_DIR / case_name
    prepared = pd.read_csv(
        case_dir / "00_prepared_input.tsv",
        sep="\t",
        dtype={"NCBI_ID": "string", "taxid": "string"},
        na_values=["NA"],
    )
    prepared_input = prepare_discrete_asr_input(
        resources.tree,
        resources.tip_data,
        prepared,
        trait_name=case_name,
    )
    reconstruction = pd.read_csv(
        case_dir / "20_ancr_all_tip_node_probabilities.tsv",
        sep="\t",
        index_col="label",
    )
    tree_taxonomy = make_tree_taxonomy_table(
        resources.tip_data,
        resources.node_data,
    )

    mappable = select_discrete_mappable_unannotated(
        reconstruction,
        mappable_labels=tree_taxonomy["label"],
        annotated_tip_labels=prepared_input.annotated_tip_labels,
    )
    joined = join_reconstruction_to_taxonomy(
        mappable,
        tree_taxonomy,
    )
    deduplicated = deduplicate_taxonomy_predictions(
        joined,
        order_by="label",
    )
    predictions = make_discrete_long_predictions(
        deduplicated,
        prepared,
        state_columns=list(prepared_input.input_tree_order.columns),
    )

    expected_mappable = pd.read_csv(
        case_dir / "22_reconstructed_mappable_unannotated.tsv",
        sep="\t",
    )
    expected_joined = pd.read_csv(
        case_dir / "23_reconstructed_joined_to_taxonomy.tsv",
        sep="\t",
        dtype={"NCBI_ID": "string", "taxid": "string"},
    )
    expected_deduplicated = pd.read_csv(
        case_dir / "24_reconstructed_deduplicated_by_ncbi.tsv",
        sep="\t",
        dtype={"NCBI_ID": "string", "taxid": "string"},
    )
    expected_predictions = pd.read_csv(
        case_dir / "25_asr_predictions_long.tsv",
        sep="\t",
        dtype={"NCBI_ID": "string", "taxid": "string"},
        na_values=["NA"],
    )

    assert mappable["label"].tolist() == expected_mappable["label"].tolist()
    assert joined["label"].tolist() == expected_joined["label"].tolist()
    assert deduplicated["label"].tolist() == expected_deduplicated[
        "label"
    ].tolist()
    assert predictions["NCBI_ID"].tolist() == expected_predictions[
        "NCBI_ID"
    ].tolist()
    assert predictions["Attribute"].tolist() == expected_predictions[
        "Attribute"
    ].tolist()
    assert predictions["Frequency"].tolist() == expected_predictions[
        "Frequency"
    ].tolist()
    np.testing.assert_allclose(
        predictions["Score"],
        expected_predictions["Score"],
        rtol=1e-12,
        atol=5e-12,
    )
