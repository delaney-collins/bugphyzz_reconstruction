"""Continuous-trait preparation and post-reconstruction stages.

These functions translate the continuous-specific portion of
``export_bugphyzz.R``. Taxonomy joins, observed-taxon exclusion, and final
output assembly deliberately remain outside this module so they can be shared
with the discrete branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

import numpy as np
import pandas as pd
from Bio.Phylo.BaseTree import Tree

from bugphyzz_reconstruction.preprocessing.utils import (
    frequency_to_post_asr_score,
    nsti_to_frequency,
)
from bugphyzz_reconstruction.asr.ltp import (
    HSPResult,
    get_nsti,
    hsp_squared_change_parsimony,
)


class ContinuousASRError(ValueError):
    """Raised when continuous-ASR inputs are invalid or inconsistent."""


@dataclass(frozen=True)
class ContinuousASRInput:
    """Prepared continuous inputs corresponding to R checkpoints 01-07."""

    joined_tip_data: pd.DataFrame
    annotated_tip_values: pd.DataFrame
    unannotated_tip_values: pd.DataFrame
    input_before_tree_order: pd.DataFrame
    input_tree_order: pd.DataFrame
    coverage: pd.DataFrame

    @property
    def annotated_tip_labels(self) -> list[str]:
        return self.annotated_tip_values["tip_label"].astype(str).tolist()

    @property
    def state_vector(self) -> pd.Series:
        """Return the named vector passed to squared-change parsimony."""
        return pd.Series(
            self.input_tree_order["Attribute_value"].to_numpy(dtype=float),
            index=pd.Index(
                self.input_tree_order["tip_label"].astype(str),
                name="tip_label",
            ),
            name="Attribute_value",
            dtype=float,
        )


@dataclass(frozen=True)
class ContinuousASRReconstruction:
    """Continuous reconstruction plus its castor-compatible fit metadata."""

    states: pd.DataFrame
    fit: HSPResult


def _tree_tip_labels(tree: Tree) -> list[str]:
    labels: list[str] = []
    for tip in tree.get_terminals(order="preorder"):
        if tip.name is None:
            raise ContinuousASRError("The tree contains an unnamed tip.")
        labels.append(str(tip.name))
    return labels


def _internal_node_labels(tree: Tree) -> list[str]:
    labels: list[str] = []
    for node in tree.get_nonterminals(order="preorder"):
        if node.name is None:
            raise ContinuousASRError(
                "The tree contains an unnamed internal node."
            )
        labels.append(str(node.name))
    return labels


def join_continuous_tip_data(
    tip_data: pd.DataFrame,
    prepared_data: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join prepared range observations to LTP tips by ``NCBI_ID``."""
    required_tip = {"tip_label", "NCBI_ID"}
    required_prepared = {"NCBI_ID", "Attribute_value"}
    missing_tip = required_tip - set(tip_data.columns)
    missing_prepared = required_prepared - set(prepared_data.columns)

    if missing_tip:
        raise ContinuousASRError(
            f"tip_data is missing columns: {sorted(missing_tip)}."
        )
    if missing_prepared:
        raise ContinuousASRError(
            f"prepared_data is missing columns: {sorted(missing_prepared)}."
        )

    tips = tip_data.reset_index(drop=True).copy()
    prepared = prepared_data.reset_index(drop=True).copy()

    return tips.merge(
        prepared,
        how="left",
        on="NCBI_ID",
        sort=False,
        suffixes=(".tree", ".trait"),
        validate="many_to_many",
    )


def prepare_continuous_asr_input(
    tree: Tree,
    tip_data: pd.DataFrame,
    prepared_data: pd.DataFrame,
    *,
    trait_name: str | None = None,
) -> ContinuousASRInput:
    """Build the exact observed/hidden vector consumed by continuous ASR."""
    joined = join_continuous_tip_data(tip_data, prepared_data)
    tip_labels = _tree_tip_labels(tree)

    joined_labels = joined["tip_label"].astype("string")
    duplicated = joined_labels[joined_labels.duplicated(keep=False)]
    if not duplicated.empty:
        preview = duplicated.drop_duplicates().head(10).tolist()
        raise ContinuousASRError(
            "Joining prepared data produced duplicate tree-tip rows. "
            f"First duplicated labels: {preview}."
        )

    values = pd.to_numeric(joined["Attribute_value"], errors="coerce")
    invalid = joined["Attribute_value"].notna() & values.isna()
    if invalid.any():
        raise ContinuousASRError(
            "Attribute_value contains nonnumeric observed values."
        )

    tip_values = pd.DataFrame(
        {
            "tip_label": joined_labels,
            "Attribute_value": values,
        }
    )
    annotated = tip_values.loc[
        tip_values["Attribute_value"].notna()
    ].reset_index(drop=True)
    unannotated = tip_values.loc[
        tip_values["Attribute_value"].isna()
    ].reset_index(drop=True)

    if annotated.empty:
        raise ContinuousASRError("All tree-tip values are hidden.")

    before_order = pd.concat(
        [annotated, unannotated],
        ignore_index=True,
    )
    labels_before = set(before_order["tip_label"].astype(str))
    tree_label_set = set(tip_labels)
    missing_from_input = [
        label for label in tip_labels if label not in labels_before
    ]
    absent_from_tree = [
        label
        for label in before_order["tip_label"].astype(str)
        if label not in tree_label_set
    ]
    if missing_from_input or absent_from_tree:
        raise ContinuousASRError(
            "Continuous input and tree tips do not agree. "
            f"Missing from input: {missing_from_input[:10]}; "
            f"absent from tree: {absent_from_tree[:10]}."
        )

    ordered = (
        before_order
        .set_index("tip_label")
        .loc[tip_labels]
        .reset_index()
    )

    coverage_percent = int(
        np.rint(len(annotated) / len(tip_labels) * 100.0)
    )
    if trait_name is None:
        if "Attribute_group" in prepared_data.columns:
            nonmissing_groups = (
                prepared_data["Attribute_group"].dropna().astype(str).unique()
            )
            trait_name = (
                nonmissing_groups[0]
                if len(nonmissing_groups) == 1
                else "continuous_trait"
            )
        else:
            trait_name = "continuous_trait"

    coverage = pd.DataFrame(
        [
            {
                "trait": trait_name,
                "annotated_tips": len(annotated),
                "total_tips": len(tip_labels),
                "coverage_percent": coverage_percent,
                "rounding": "round",
                "asr_run": coverage_percent >= 1,
            }
        ]
    )

    return ContinuousASRInput(
        joined_tip_data=joined,
        annotated_tip_values=annotated,
        unannotated_tip_values=unannotated,
        input_before_tree_order=before_order,
        input_tree_order=ordered,
        coverage=coverage,
    )


def reconstruct_continuous_states(
    tree: Tree,
    prepared_input: ContinuousASRInput,
    *,
    weighted: bool = True,
) -> ContinuousASRReconstruction:
    """Run HSP and label every reconstructed tip and internal-node state."""
    if not bool(prepared_input.coverage["asr_run"].iloc[0]):
        raise ContinuousASRError(
            "Continuous ASR coverage is below the one-percent threshold."
        )

    result = hsp_squared_change_parsimony(
        tree,
        prepared_input.state_vector,
        weighted=weighted,
        check_input=True,
    )
    if not result.success:
        raise ContinuousASRError(
            "Squared-change parsimony reconstruction was unsuccessful."
        )

    tip_labels = _tree_tip_labels(tree)
    node_labels = _internal_node_labels(tree)
    labels = tip_labels + node_labels
    locations = (
        ["tip"] * len(tip_labels)
        + ["internal_node"] * len(node_labels)
    )

    states = pd.DataFrame(
        {
            "label": labels,
            "Attribute_value": result.states,
            "location": locations,
        }
    )
    return ContinuousASRReconstruction(states=states, fit=result)


def select_mappable_unannotated_states(
    reconstruction: pd.DataFrame,
    *,
    mappable_labels: Collection[str],
    annotated_tip_labels: Collection[str],
) -> pd.DataFrame:
    """Keep reconstructed tree positions that map to taxonomy and were hidden."""
    required = {"label", "Attribute_value"}
    missing = required - set(reconstruction.columns)
    if missing:
        raise ContinuousASRError(
            f"reconstruction is missing columns: {sorted(missing)}."
        )

    mappable = {
        str(label)
        for label in mappable_labels
        if not pd.isna(label)
    }
    annotated = {
        str(label)
        for label in annotated_tip_labels
        if not pd.isna(label)
    }
    labels = reconstruction["label"].astype(str)

    return reconstruction.loc[
        labels.isin(mappable) & ~labels.isin(annotated)
    ].reset_index(drop=True)


def calculate_continuous_nsti(
    tree: Tree,
    *,
    annotated_tip_labels: Collection[str],
    node_labels: Collection[str],
) -> pd.DataFrame:
    """Calculate NSTI and annotate whether each row is a tip or node."""
    result = get_nsti(
        tree,
        annotated_tip_labels=annotated_tip_labels,
        nodes=node_labels,
    )
    tip_label_set = set(_tree_tip_labels(tree))
    result["location"] = np.where(
        result["label"].astype(str).isin(tip_label_set),
        "unannotated_tip",
        "internal_node",
    )
    return result


def attach_continuous_nsti(
    reconstructed_states: pd.DataFrame,
    nsti: pd.DataFrame,
) -> pd.DataFrame:
    """Attach NSTI values to continuous reconstructed states by tree label."""
    required_states = {"label", "Attribute_value"}
    required_nsti = {"label", "nsti"}
    if missing := required_states - set(reconstructed_states.columns):
        raise ContinuousASRError(
            f"reconstructed_states is missing columns: {sorted(missing)}."
        )
    if missing := required_nsti - set(nsti.columns):
        raise ContinuousASRError(
            f"nsti is missing columns: {sorted(missing)}."
        )

    return reconstructed_states.merge(
        nsti,
        how="left",
        on="label",
        sort=False,
        suffixes=(".x", ".y"),
        validate="one_to_one",
    )


def add_continuous_prediction_evidence(
    reconstructed_with_nsti: pd.DataFrame,
) -> pd.DataFrame:
    """Convert NSTI into Bugphyzz evidence fields before taxonomy mapping."""
    if "nsti" not in reconstructed_with_nsti.columns:
        raise ContinuousASRError(
            "reconstructed_with_nsti is missing required column: nsti"
        )

    result = reconstructed_with_nsti.copy()
    result["Attribute_source"] = pd.NA
    result["Confidence_in_curation"] = pd.NA
    result["Frequency"] = nsti_to_frequency(result["nsti"])
    result["Score"] = frequency_to_post_asr_score(result["Frequency"])
    result["Evidence"] = "asr"
    return result
