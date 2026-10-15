"""Checkpoint parity tests for continuous-specific ASR stages."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bugphyzz.trees.continuous import (
    add_continuous_prediction_evidence,
    attach_continuous_nsti,
    calculate_continuous_nsti,
    prepare_continuous_asr_input,
    reconstruct_continuous_states,
    select_mappable_unannotated_states,
)
from bugphyzz.trees.ltp import ltp


PACKAGE_DIR = Path(__file__).resolve().parents[1]
RESOURCE_DIR = PACKAGE_DIR / "resources"
REFERENCE_DIR = PACKAGE_DIR / "asr_reference_2023"
CASE_DIR = REFERENCE_DIR / "growth_temperature"
TREE_REFERENCE_DIR = REFERENCE_DIR / "_tree_reference"


@pytest.fixture(scope="module")
def continuous_stages():
    resources = ltp(
        resource_dir=RESOURCE_DIR,
        remove_gn_nodes=True,
        node_names=True,
    )
    prepared = pd.read_csv(
        CASE_DIR / "00_prepared_input.tsv",
        sep="\t",
        na_values=["NA"],
        dtype={"NCBI_ID": "string", "taxid": "string"},
    )
    inputs = prepare_continuous_asr_input(
        resources.tree,
        resources.tip_data,
        prepared,
        trait_name="growth_temperature",
    )
    reconstruction = reconstruct_continuous_states(
        resources.tree,
        inputs,
    )
    tree_data = pd.read_csv(
        TREE_REFERENCE_DIR / "tree_data_combined.tsv",
        sep="\t",
        dtype="string",
    )
    selected = select_mappable_unannotated_states(
        reconstruction.states,
        mappable_labels=tree_data["label"],
        annotated_tip_labels=inputs.annotated_tip_labels,
    )
    with (
        TREE_REFERENCE_DIR / "nodes_with_taxid.txt"
    ).open(encoding="utf-8") as handle:
        nodes = [line.rstrip("\r\n") for line in handle if line.rstrip("\r\n")]

    nsti = calculate_continuous_nsti(
        resources.tree,
        annotated_tip_labels=inputs.annotated_tip_labels,
        node_labels=nodes,
    )
    attached = attach_continuous_nsti(selected, nsti)
    evidence = add_continuous_prediction_evidence(attached)

    return inputs, reconstruction, selected, nsti, attached, evidence


def test_continuous_tree_ordered_input_matches_r(continuous_stages) -> None:
    inputs, *_ = continuous_stages
    expected = pd.read_csv(
        CASE_DIR / "07_input_vector_tree_order.tsv",
        sep="\t",
        na_values=["NA"],
    )

    assert inputs.input_tree_order["tip_label"].tolist() == expected[
        "tip_label"
    ].tolist()
    np.testing.assert_allclose(
        inputs.input_tree_order["Attribute_value"],
        expected["Attribute_value"],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    assert inputs.coverage.iloc[0].to_dict() == {
        "trait": "growth_temperature",
        "annotated_tips": 14460,
        "total_tips": 23501,
        "coverage_percent": 62,
        "rounding": "round",
        "asr_run": True,
    }


def test_all_continuous_reconstructed_states_match_r(
    continuous_stages,
) -> None:
    _, reconstruction, *_ = continuous_stages
    expected = pd.read_csv(
        CASE_DIR / "09_all_reconstructed_states.tsv",
        sep="\t",
    )

    assert reconstruction.states["label"].tolist() == expected["label"].tolist()
    assert reconstruction.states["location"].tolist() == expected[
        "location"
    ].tolist()
    np.testing.assert_allclose(
        reconstruction.states["Attribute_value"],
        expected["Attribute_value"],
        rtol=1e-10,
        atol=1e-9,
    )


def test_mappable_continuous_states_match_r(continuous_stages) -> None:
    _, _, selected, *_ = continuous_stages
    expected = pd.read_csv(
        CASE_DIR / "10_mappable_unannotated_states.tsv",
        sep="\t",
    )

    assert selected["label"].tolist() == expected["label"].tolist()
    assert selected["location"].tolist() == expected["location"].tolist()
    np.testing.assert_allclose(
        selected["Attribute_value"],
        expected["Attribute_value"],
        rtol=1e-10,
        atol=1e-9,
    )


def test_continuous_nsti_and_attachment_match_r(
    continuous_stages,
) -> None:
    _, _, _, nsti, attached, _ = continuous_stages
    expected_nsti = pd.read_csv(CASE_DIR / "11_nsti.tsv", sep="\t")
    expected_attached = pd.read_csv(
        CASE_DIR / "12_reconstructed_states_with_nsti.tsv",
        sep="\t",
    )

    assert nsti["label"].tolist() == expected_nsti["label"].tolist()
    assert nsti["location"].tolist() == expected_nsti["location"].tolist()
    np.testing.assert_allclose(
        nsti["nsti"],
        expected_nsti["nsti"],
        rtol=1e-12,
        atol=1e-15,
    )

    assert attached["label"].tolist() == expected_attached["label"].tolist()
    np.testing.assert_allclose(
        attached["Attribute_value"],
        expected_attached["Attribute_value"],
        rtol=1e-10,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        attached["nsti"],
        expected_attached["nsti"],
        rtol=1e-12,
        atol=1e-15,
    )


def test_continuous_evidence_fields_are_complete(continuous_stages) -> None:
    *_, evidence = continuous_stages

    assert evidence["Evidence"].eq("asr").all()
    assert evidence["Frequency"].notna().all()
    assert evidence["Score"].notna().all()
    assert evidence["Attribute_source"].isna().all()
    assert evidence["Confidence_in_curation"].isna().all()
