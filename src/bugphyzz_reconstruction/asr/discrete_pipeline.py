"""Discrete-specific preparation and ASR postprocessing for Bugphyzz."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from Bio.Phylo.BaseTree import Tree

from bugphyzz_reconstruction.asr.asr_shared import (
    deduplicate_taxonomy_predictions,
    exclude_observed_taxa,
    join_reconstruction_to_taxonomy,
)
from bugphyzz_reconstruction.asr.discrete import (
    AncrResult,
    FitMkResult,
    ancr,
    fit_mk,
)


class DiscreteASRError(ValueError):
    """Raised when discrete-ASR inputs are invalid or inconsistent."""


def _asr_score_to_frequency(values: pd.Series) -> pd.Series:
    """Apply R ``scores2Freq`` thresholds without near-one coercion."""
    scores = pd.to_numeric(values, errors="coerce")
    conditions = [
        scores.eq(1.0),
        scores.ge(0.9) & scores.lt(1.0),
        scores.ge(0.5) & scores.lt(0.9),
        scores.gt(0.0) & scores.lt(0.5),
        scores.eq(0.0),
    ]
    labels = ["always", "usually", "sometimes", "rarely", "never"]
    return pd.Series(
        pd.array(
            np.select(conditions, labels, default=None),
            dtype="string",
        ),
        index=values.index,
        name=values.name,
    )


@dataclass(frozen=True)
class DiscreteASRInput:
    """Prepared discrete inputs corresponding to R checkpoints 01-08."""

    joined_tip_data: pd.DataFrame
    annotated_tips_long: pd.DataFrame
    annotated_tip_state_matrix: pd.DataFrame
    unannotated_uniform_state_matrix: pd.DataFrame
    input_before_tree_order: pd.DataFrame
    input_tree_order: pd.DataFrame
    coverage: pd.DataFrame

    @property
    def annotated_tip_labels(self) -> list[str]:
        return self.annotated_tip_state_matrix.index.astype(str).tolist()


@dataclass(frozen=True)
class DiscreteASRStages:
    """Complete discrete-specific result prior to shared final assembly."""

    prepared_input: DiscreteASRInput
    fit: FitMkResult
    reconstruction: AncrResult
    mappable_unannotated: pd.DataFrame
    joined_to_taxonomy: pd.DataFrame
    deduplicated_by_taxonomy: pd.DataFrame
    predictions_long: pd.DataFrame


def _tree_tip_labels(tree: Tree) -> list[str]:
    labels: list[str] = []
    for tip in tree.get_terminals(order="preorder"):
        if tip.name is None:
            raise DiscreteASRError("The tree contains an unnamed tip.")
        labels.append(str(tip.name))
    return labels


def join_discrete_tip_data(
    tip_data: pd.DataFrame,
    prepared_data: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join prepared state distributions to LTP tips by NCBI ID."""
    required_tips = {"tip_label", "NCBI_ID"}
    required_prepared = {"NCBI_ID", "Attribute", "Score"}
    missing_tips = required_tips - set(tip_data.columns)
    missing_prepared = required_prepared - set(prepared_data.columns)
    if missing_tips:
        raise DiscreteASRError(
            f"tip_data is missing columns: {sorted(missing_tips)}."
        )
    if missing_prepared:
        raise DiscreteASRError(
            f"prepared_data is missing columns: {sorted(missing_prepared)}."
        )

    return tip_data.reset_index(drop=True).merge(
        prepared_data.reset_index(drop=True),
        how="left",
        on="NCBI_ID",
        sort=False,
        suffixes=(".tree", ".trait"),
        validate="many_to_many",
    )


def prepare_discrete_asr_input(
    tree: Tree,
    tip_data: pd.DataFrame,
    prepared_data: pd.DataFrame,
    *,
    trait_name: str | None = None,
) -> DiscreteASRInput:
    """Construct observed and uniform hidden-tip state matrices."""
    joined = join_discrete_tip_data(tip_data, prepared_data)
    tip_labels = _tree_tip_labels(tree)

    annotated_long = (
        joined.loc[
            joined["Attribute"].notna(),
            ["tip_label", "Attribute", "Score"],
        ]
        .copy()
        .reset_index(drop=True)
    )
    if annotated_long.empty:
        raise DiscreteASRError("All tree-tip states are hidden.")

    annotated_long["Score"] = pd.to_numeric(
        annotated_long["Score"],
        errors="coerce",
    )
    if annotated_long["Score"].isna().any():
        raise DiscreteASRError(
            "Observed discrete tip scores must be numeric and nonmissing."
        )
    if annotated_long.duplicated(["tip_label", "Attribute"]).any():
        duplicates = (
            annotated_long.loc[
                annotated_long.duplicated(
                    ["tip_label", "Attribute"],
                    keep=False,
                ),
                ["tip_label", "Attribute"],
            ]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise DiscreteASRError(
            "More than one score was supplied for a tip/state pair. "
            f"First duplicates: {duplicates}."
        )

    state_order = annotated_long["Attribute"].drop_duplicates().astype(str)
    annotated_matrix = (
        annotated_long.pivot(
            index="tip_label",
            columns="Attribute",
            values="Score",
        )
        .reindex(columns=state_order)
        .fillna(0.0)
    )
    annotated_matrix.index = annotated_matrix.index.astype(str)
    annotated_matrix.index.name = "tip_label"
    annotated_matrix.columns.name = None

    unannotated_labels = (
        joined.loc[joined["Attribute"].isna(), "tip_label"]
        .astype(str)
        .tolist()
    )
    if len(unannotated_labels) != len(set(unannotated_labels)):
        raise DiscreteASRError(
            "Joining prepared data produced duplicate unannotated tip rows."
        )

    state_count = annotated_matrix.shape[1]
    uniform_matrix = pd.DataFrame(
        1.0 / state_count,
        index=pd.Index(unannotated_labels, name="tip_label"),
        columns=annotated_matrix.columns,
    )
    before_order = pd.concat(
        [annotated_matrix, uniform_matrix],
        axis=0,
    )

    duplicated_labels = before_order.index[
        before_order.index.duplicated(keep=False)
    ].unique()
    if len(duplicated_labels):
        raise DiscreteASRError(
            "Discrete input contains duplicated tip labels. "
            f"First values: {duplicated_labels[:10].tolist()}."
        )

    tree_label_set = set(tip_labels)
    input_label_set = set(before_order.index.astype(str))
    missing_from_input = [
        label for label in tip_labels if label not in input_label_set
    ]
    absent_from_tree = [
        label for label in before_order.index if label not in tree_label_set
    ]
    if missing_from_input or absent_from_tree:
        raise DiscreteASRError(
            "Discrete input and tree tips do not agree. "
            f"Missing from input: {missing_from_input[:10]}; "
            f"absent from tree: {absent_from_tree[:10]}."
        )

    ordered = before_order.loc[tip_labels].copy()
    annotated_count = len(annotated_matrix)
    coverage_percent = int(
        np.floor(annotated_count / len(tip_labels) * 100.0)
    )

    if trait_name is None:
        if "Attribute_group" in prepared_data.columns:
            groups = (
                prepared_data["Attribute_group"].dropna().astype(str).unique()
            )
            trait_name = groups[0] if len(groups) == 1 else "discrete_trait"
        else:
            trait_name = "discrete_trait"

    coverage = pd.DataFrame(
        [
            {
                "trait": trait_name,
                "annotated_tips": annotated_count,
                "total_tips": len(tip_labels),
                "coverage_percent": coverage_percent,
                "rounding": "floor",
                "asr_run": coverage_percent >= 1,
            }
        ]
    )

    return DiscreteASRInput(
        joined_tip_data=joined,
        annotated_tips_long=annotated_long,
        annotated_tip_state_matrix=annotated_matrix,
        unannotated_uniform_state_matrix=uniform_matrix,
        input_before_tree_order=before_order,
        input_tree_order=ordered,
        coverage=coverage,
    )


def select_discrete_mappable_unannotated(
    reconstruction: pd.DataFrame,
    *,
    mappable_labels: pd.Series,
    annotated_tip_labels: list[str],
) -> pd.DataFrame:
    """Keep taxonomy-mappable reconstructed rows that were not observations."""
    if reconstruction.index.name == "label":
        working = reconstruction.reset_index()
    elif "label" in reconstruction.columns:
        working = reconstruction.copy()
    else:
        raise DiscreteASRError(
            "reconstruction must have a 'label' index or column."
        )

    mappable = {
        str(value)
        for value in mappable_labels
        if not pd.isna(value)
    }
    annotated = set(map(str, annotated_tip_labels))
    labels = working["label"].astype(str)
    return working.loc[
        labels.isin(mappable) & ~labels.isin(annotated)
    ].reset_index(drop=True)


def make_discrete_long_predictions(
    deduplicated: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    state_columns: list[str],
) -> pd.DataFrame:
    """Reshape marginal state probabilities and attach ASR evidence fields."""
    identifier_columns = [
        "label",
        "NCBI_ID",
        "Taxon_name",
        "Rank",
        "taxid",
    ]
    missing = (
        set(identifier_columns)
        | set(state_columns)
        | {"NCBI_ID"}
    ) - set(deduplicated.columns)
    if missing:
        raise DiscreteASRError(
            f"deduplicated data is missing columns: {sorted(missing)}."
        )

    working = deduplicated.copy()
    working["_asr_row_order"] = range(len(working))
    long = working.melt(
        id_vars=identifier_columns,
        value_vars=state_columns,
        var_name="Attribute",
        value_name="Score",
    )
    state_order = {
        state: index
        for index, state in enumerate(state_columns)
    }
    # pandas.melt emits one complete state column at a time, whereas
    # tidyr::pivot_longer emits all states for row 1, then all states for
    # row 2. Restore the R ordering explicitly.
    long["_asr_row_order"] = np.tile(
        np.arange(len(working)),
        len(state_columns),
    )
    long["_asr_state_order"] = long["Attribute"].map(state_order)
    long = (
        long.sort_values(
            ["_asr_row_order", "_asr_state_order"],
            kind="stable",
        )
        .drop(columns=["_asr_row_order", "_asr_state_order"])
        .reset_index(drop=True)
    )
    long = exclude_observed_taxa(long, observed)
    long["Evidence"] = "asr"
    long["Confidence_in_curation"] = pd.NA
    long["Attribute_source"] = pd.NA
    long["Frequency"] = _asr_score_to_frequency(long["Score"])

    ordered = identifier_columns + [
        "Attribute",
        "Score",
        "Evidence",
        "Confidence_in_curation",
        "Attribute_source",
        "Frequency",
    ]
    return long.loc[:, ordered].reset_index(drop=True)


def run_discrete_asr_pipeline(
    tree: Tree,
    tip_data: pd.DataFrame,
    tree_taxonomy: pd.DataFrame,
    prepared_data: pd.DataFrame,
    *,
    trait_name: str | None = None,
) -> DiscreteASRStages:
    """Run the discrete branch through the long-form ASR prediction table."""
    prepared_input = prepare_discrete_asr_input(
        tree,
        tip_data,
        prepared_data,
        trait_name=trait_name,
    )
    if not bool(prepared_input.coverage["asr_run"].iloc[0]):
        raise DiscreteASRError(
            "Discrete ASR coverage is below the one-percent threshold."
        )

    fitted = fit_mk(
        tree,
        prepared_input.input_tree_order,
        model="ER",
        pi="equal",
    )
    reconstruction = ancr(fitted, tips=True)
    mappable = select_discrete_mappable_unannotated(
        reconstruction.ace,
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
        prepared_data,
        state_columns=list(prepared_input.input_tree_order.columns),
    )

    return DiscreteASRStages(
        prepared_input=prepared_input,
        fit=fitted,
        reconstruction=reconstruction,
        mappable_unannotated=mappable,
        joined_to_taxonomy=joined,
        deduplicated_by_taxonomy=deduplicated,
        predictions_long=predictions,
    )
