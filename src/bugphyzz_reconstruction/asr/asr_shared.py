"""Taxonomy mapping and final assembly shared by continuous and discrete ASR."""

from __future__ import annotations

from typing import Literal

import pandas as pd


class ASRAssemblyError(ValueError):
    """Raised when reconstructed predictions cannot be assembled safely."""


TAXONOMY_COLUMNS = [
    "label",
    "NCBI_ID",
    "Taxon_name",
    "Rank",
    "taxid",
]


def make_tree_taxonomy_table(
    tip_data: pd.DataFrame,
    node_data: pd.DataFrame,
) -> pd.DataFrame:
    """Combine LTP tip and internal-node metadata into one mapping table."""
    tip_required = {
        "tip_label",
        "NCBI_ID",
        "Taxon_name",
        "Rank",
        "taxid",
    }
    node_required = {
        "node_label",
        "NCBI_ID",
        "Taxon_name",
        "Rank",
        "taxid",
    }
    missing_tips = tip_required - set(tip_data.columns)
    missing_nodes = node_required - set(node_data.columns)
    if missing_tips:
        raise ASRAssemblyError(
            f"tip_data is missing columns: {sorted(missing_tips)}."
        )
    if missing_nodes:
        raise ASRAssemblyError(
            f"node_data is missing columns: {sorted(missing_nodes)}."
        )

    tips = (
        tip_data.reset_index(drop=True)
        .rename(columns={"tip_label": "label"})
        .loc[:, TAXONOMY_COLUMNS]
    )
    nodes = (
        node_data.reset_index(drop=True)
        .rename(columns={"node_label": "label"})
        .loc[:, TAXONOMY_COLUMNS]
    )
    return pd.concat([tips, nodes], ignore_index=True)


def join_reconstruction_to_taxonomy(
    reconstructed: pd.DataFrame,
    tree_taxonomy: pd.DataFrame,
) -> pd.DataFrame:
    """Map reconstructed tree labels to tip/node taxonomy metadata."""
    if "label" not in reconstructed.columns:
        raise ASRAssemblyError(
            "reconstructed is missing required column: label"
        )
    missing = set(TAXONOMY_COLUMNS) - set(tree_taxonomy.columns)
    if missing:
        raise ASRAssemblyError(
            f"tree_taxonomy is missing columns: {sorted(missing)}."
        )

    return reconstructed.merge(
        tree_taxonomy.loc[:, TAXONOMY_COLUMNS],
        how="left",
        on="label",
        sort=False,
        validate="many_to_many",
    )


def exclude_observed_taxa(
    predictions: pd.DataFrame,
    observed: pd.DataFrame,
) -> pd.DataFrame:
    """Remove missing taxonomy IDs and IDs already carrying observations."""
    if "NCBI_ID" not in predictions.columns:
        raise ASRAssemblyError(
            "predictions is missing required column: NCBI_ID"
        )
    if "NCBI_ID" not in observed.columns:
        raise ASRAssemblyError(
            "observed is missing required column: NCBI_ID"
        )

    observed_ids = {
        str(value)
        for value in observed["NCBI_ID"]
        if not pd.isna(value)
    }
    ids = predictions["NCBI_ID"].astype("string")
    keep = ids.notna() & ~ids.astype(str).isin(observed_ids)
    return predictions.loc[keep].reset_index(drop=True)


def deduplicate_taxonomy_predictions(
    predictions: pd.DataFrame,
    *,
    order_by: str,
) -> pd.DataFrame:
    """Keep the maximum-ranked tree mapping for every taxonomy ID.

    This reproduces ``group_by(NCBI_ID) |> slice_max(order_by=..., n=1,
    with_ties=FALSE)``. Groups are emitted in R/dplyr's sorted key order,
    while ties retain the first input row.
    """
    required = {"NCBI_ID", order_by}
    missing = required - set(predictions.columns)
    if missing:
        raise ASRAssemblyError(
            f"predictions is missing columns: {sorted(missing)}."
        )
    if predictions.empty:
        return predictions.reset_index(drop=True)

    working = predictions.copy()
    working["_asr_input_order"] = range(len(working))
    working = working.sort_values(
        ["NCBI_ID", order_by, "_asr_input_order"],
        ascending=[True, False, True],
        kind="stable",
        na_position="last",
    )
    output = (
        working.groupby("NCBI_ID", sort=True, dropna=False)
        .head(1)
        .drop(columns="_asr_input_order")
        .reset_index(drop=True)
    )
    return output


def prepare_mapped_predictions(
    reconstructed: pd.DataFrame,
    tree_taxonomy: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    deduplicate_by: str,
) -> pd.DataFrame:
    """Join taxonomy, remove observed IDs, and select one mapping per ID."""
    joined = join_reconstruction_to_taxonomy(
        reconstructed,
        tree_taxonomy,
    )
    unobserved = exclude_observed_taxa(joined, observed)
    return deduplicate_taxonomy_predictions(
        unobserved,
        order_by=deduplicate_by,
    )


def combine_observed_and_predictions(
    observed: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    attribute_group: str,
    attribute_type: str,
    predictions_first: bool = False,
) -> pd.DataFrame:
    """Row-bind observations and predictions and add shared trait metadata."""
    if not attribute_group:
        raise ASRAssemblyError("attribute_group must be a nonempty string.")
    if not attribute_type:
        raise ASRAssemblyError("attribute_type must be a nonempty string.")

    tables = (
        [predictions, observed]
        if predictions_first
        else [observed, predictions]
    )
    combined = pd.concat(
        [table.reset_index(drop=True) for table in tables],
        ignore_index=True,
        sort=False,
    )
    combined["Attribute_group"] = attribute_group
    combined["Attribute_type"] = attribute_type
    return combined


def format_final_asr_output(
    combined: pd.DataFrame,
    *,
    attribute_group: str,
    attribute_type: Literal[
        "range",
        "binary",
        "multistate-intersection",
    ],
) -> pd.DataFrame:
    """Format a combined observed/predicted table for Bugphyzz output."""
    result = combined.copy()

    if attribute_type == "range":
        result["Attribute"] = attribute_group
        ordered = [
            "NCBI_ID",
            "Taxon_name",
            "Rank",
            "Attribute",
            "Attribute_value",
            "Evidence",
            "Frequency",
            "Attribute_source",
            "Confidence_in_curation",
            "Score",
            "Attribute_type",
            "Attribute_group",
            "taxid",
        ]
    elif attribute_type in {"binary", "multistate-intersection"}:
        if "Attribute" not in result.columns:
            raise ASRAssemblyError(
                "Discrete combined data is missing required column: Attribute"
            )

        if attribute_type == "binary":
            split = result["Attribute"].astype("string").str.split(
                "--",
                n=1,
                expand=True,
            )
            if split.shape[1] != 2 or split.iloc[:, 1].isna().any():
                raise ASRAssemblyError(
                    "Binary Attribute values must use 'trait--state' format."
                )
            result["Attribute"] = split.iloc[:, 0]
            result["Attribute_value"] = split.iloc[:, 1]
        else:
            result["Attribute_value"] = result["Attribute"]
            result["Attribute"] = attribute_group

        ordered = [
            "NCBI_ID",
            "Taxon_name",
            "Rank",
            "Attribute",
            "Attribute_value",
            "Frequency",
            "Score",
            "Attribute_source",
            "Confidence_in_curation",
            "taxid",
            "Evidence",
            "Attribute_group",
            "Attribute_type",
        ]
    else:
        raise NotImplementedError(
            f"Final ASR formatting is not implemented for {attribute_type!r}."
        )

    missing = [column for column in ordered if column not in result.columns]
    if missing:
        raise ASRAssemblyError(
            f"Combined ASR data is missing final columns: {missing}."
        )

    trailing = [
        column
        for column in result.columns
        if column not in ordered and column != "label"
    ]
    return result.loc[:, ordered + trailing].reset_index(drop=True)
