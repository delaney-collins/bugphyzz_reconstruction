"""Parity tests for taxonomy mapping and shared ASR assembly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bugphyzz.trees.asr_shared import (
    combine_observed_and_predictions,
    deduplicate_taxonomy_predictions,
    exclude_observed_taxa,
    format_final_asr_output,
    join_reconstruction_to_taxonomy,
    make_tree_taxonomy_table,
)
from bugphyzz.trees.continuous import add_continuous_prediction_evidence
from bugphyzz.trees.ltp import ltp


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PACKAGE_DIR / "asr_reference_2023"
RESOURCE_DIR = PACKAGE_DIR / "resources"


@pytest.fixture(scope="module")
def tree_taxonomy() -> pd.DataFrame:
    resources = ltp(
        resource_dir=RESOURCE_DIR,
        remove_gn_nodes=True,
        node_names=True,
    )
    return make_tree_taxonomy_table(
        resources.tip_data,
        resources.node_data,
    )


def test_continuous_taxonomy_mapping_and_deduplication_match_r(
    tree_taxonomy: pd.DataFrame,
) -> None:
    case_dir = REFERENCE_DIR / "growth_temperature"
    reconstructed = pd.read_csv(
        case_dir / "12_reconstructed_states_with_nsti.tsv",
        sep="\t",
    )
    observed = pd.read_csv(
        case_dir / "15_observed_rows_retained.tsv",
        sep="\t",
        dtype={"NCBI_ID": "string", "taxid": "string"},
        na_values=["NA"],
    )
    expected = pd.read_csv(
        case_dir / "14_final_asr_predictions.tsv",
        sep="\t",
        dtype={"NCBI_ID": "string", "taxid": "string"},
        na_values=["NA"],
    )

    evidence = add_continuous_prediction_evidence(reconstructed)
    joined = join_reconstruction_to_taxonomy(evidence, tree_taxonomy)
    unobserved = exclude_observed_taxa(joined, observed)
    predictions = deduplicate_taxonomy_predictions(
        unobserved,
        order_by="Attribute_value",
    )

    assert predictions["NCBI_ID"].tolist() == expected["NCBI_ID"].tolist()
    assert predictions["Taxon_name"].tolist() == expected["Taxon_name"].tolist()
    assert predictions["Rank"].tolist() == expected["Rank"].tolist()
    assert predictions["Frequency"].tolist() == expected["Frequency"].tolist()
    assert predictions["Evidence"].eq("asr").all()
    np.testing.assert_allclose(
        predictions["Attribute_value"],
        expected["Attribute_value"],
        rtol=1e-10,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        predictions["nsti"],
        expected["nsti"],
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        predictions["Score"],
        expected["Score"],
        rtol=0,
        atol=0,
    )


def test_continuous_final_assembly_matches_r(
    tree_taxonomy: pd.DataFrame,
) -> None:
    case_dir = REFERENCE_DIR / "growth_temperature"
    reconstructed = add_continuous_prediction_evidence(
        pd.read_csv(
            case_dir / "12_reconstructed_states_with_nsti.tsv",
            sep="\t",
        )
    )
    observed = pd.read_csv(
        case_dir / "15_observed_rows_retained.tsv",
        sep="\t",
        dtype={"NCBI_ID": "string", "taxid": "string"},
        na_values=["NA"],
    )
    predictions = deduplicate_taxonomy_predictions(
        exclude_observed_taxa(
            join_reconstruction_to_taxonomy(
                reconstructed,
                tree_taxonomy,
            ),
            observed,
        ),
        order_by="Attribute_value",
    )
    combined = combine_observed_and_predictions(
        observed,
        predictions,
        attribute_group="growth temperature",
        attribute_type="range",
    )
    result = format_final_asr_output(
        combined,
        attribute_group="growth temperature",
        attribute_type="range",
    )
    expected = pd.read_csv(
        case_dir / "99_final_propagated.tsv",
        sep="\t",
        dtype={
            "NCBI_ID": "string",
            "taxid": "string",
            "Attribute": "string",
        },
        na_values=["NA"],
    )

    assert result.columns.tolist() == expected.columns.tolist()
    assert result["NCBI_ID"].tolist() == expected["NCBI_ID"].tolist()
    assert result["Attribute"].tolist() == expected["Attribute"].tolist()
    np.testing.assert_allclose(
        pd.to_numeric(result["Attribute_value"]),
        pd.to_numeric(expected["Attribute_value"]),
        rtol=1e-10,
        atol=1e-9,
    )


def test_discrete_taxonomy_mapping_and_deduplication_match_r(
    tree_taxonomy: pd.DataFrame,
) -> None:
    case_dir = REFERENCE_DIR / "motility"
    reconstructed = pd.read_csv(
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

    joined = join_reconstruction_to_taxonomy(
        reconstructed,
        tree_taxonomy,
    )
    deduplicated = deduplicate_taxonomy_predictions(
        joined,
        order_by="label",
    )

    assert joined["label"].tolist() == expected_joined["label"].tolist()
    assert joined["NCBI_ID"].tolist() == expected_joined["NCBI_ID"].tolist()
    assert deduplicated["label"].tolist() == expected_deduplicated[
        "label"
    ].tolist()
    assert deduplicated["NCBI_ID"].tolist() == expected_deduplicated[
        "NCBI_ID"
    ].tolist()


def test_discrete_final_assembly_matches_r() -> None:
    case_dir = REFERENCE_DIR / "motility"
    predictions = pd.read_csv(
        case_dir / "25_asr_predictions_long.tsv",
        sep="\t",
        dtype={"NCBI_ID": "string", "taxid": "string"},
        na_values=["NA"],
    )
    observed = pd.read_csv(
        case_dir / "26_observed_rows_retained.tsv",
        sep="\t",
        dtype={"NCBI_ID": "string", "taxid": "string"},
        na_values=["NA"],
    )
    combined = combine_observed_and_predictions(
        observed,
        predictions,
        attribute_group="motility",
        attribute_type="binary",
        predictions_first=True,
    )
    result = format_final_asr_output(
        combined,
        attribute_group="motility",
        attribute_type="binary",
    )
    expected = pd.read_csv(
        case_dir / "99_final_propagated.tsv",
        sep="\t",
        dtype={
            "NCBI_ID": "string",
            "taxid": "string",
            "Attribute": "string",
            "Attribute_value": "string",
        },
        na_values=["NA"],
    )

    assert result.columns.tolist() == expected.columns.tolist()
    assert result["NCBI_ID"].tolist() == expected["NCBI_ID"].tolist()
    assert result["Attribute"].astype(str).tolist() == expected[
        "Attribute"
    ].astype(str).tolist()
    assert result["Attribute_value"].astype(str).tolist() == expected[
        "Attribute_value"
    ].astype(str).tolist()
    np.testing.assert_allclose(
        result["Score"],
        expected["Score"],
        rtol=1e-12,
        atol=1e-14,
    )
