"""Taxonomy-aware trait preparation translated from taxPPro.R.

This module implements:

- get_set_with_ids
- get_set_without_ids
- ordered_columns

The functions resolve direct and parent-based taxonomy records using the
frozen 2023 taxonomy resources.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from bugphyzz_reconstruction.preprocessing.utils import (
    add_rank_prefix,
    confidence_to_categorical,
    score_to_frequency,
)
from bugphyzz_reconstruction.taxonomy.frozen import FrozenTaxonomyProvider


VALID_RANKS: frozenset[str] = frozenset(
    {
        "genus",
        "species",
        "strain",
    }
)

ORDERED_COLUMNS: tuple[str, ...] = (
    "NCBI_ID",
    "Taxon_name",
    "Rank",
    "Attribute",
    "Attribute_source",
    "Confidence_in_curation",
    "Evidence",
    "Frequency",
    "Score",
    "Attribute_group",
    "Attribute_type",
    "taxid",
)


class TaxonomySetError(ValueError):
    """Raised when taxonomy-set preparation cannot be completed consistently."""


def ordered_columns() -> list[str]:
    """Return the R-compatible final column order."""
    return list(ORDERED_COLUMNS)


def _require_columns(
    table: pd.DataFrame,
    required: Sequence[str],
    *,
    function_name: str,
) -> None:
    """Validate required input columns."""
    missing = sorted(set(required).difference(table.columns))

    if missing:
        raise TaxonomySetError(
            f"{function_name} is missing required columns: "
            + ", ".join(missing)
        )

def _clip_probability_scores(values: pd.Series) -> pd.Series:
    scores = pd.to_numeric(values, errors="coerce").astype("float64")

    tolerance = 1e-12
    invalid = scores.lt(-tolerance) | scores.gt(1.0 + tolerance)
    if invalid.any():
        raise TaxonomySetError(
            "Calculated probability scores fall outside [0, 1]."
        )

    return scores.clip(lower=0.0, upper=1.0)


def _get_single_nonmissing_value(
    values: pd.Series,
    *,
    field_name: str,
) -> str:
    """Return exactly one unique non-missing string value."""
    unique_values = (
        values.astype("string")
        .dropna()
        .str.strip()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if len(unique_values) != 1:
        raise TaxonomySetError(
            f"Expected exactly one non-missing {field_name}, "
            f"but found {unique_values.tolist()}."
        )

    return str(unique_values.iloc[0])


def _normalize_attribute_type(attribute_type: str) -> str:
    """Normalize supported attribute-type names."""
    normalized = attribute_type.strip().lower()

    aliases = {
        "mltistate-union": "multistate-union",
    }

    return aliases.get(normalized, normalized)


def _normalize_taxid(values: pd.Series) -> pd.Series:
    """Normalize taxids to nullable strings."""
    result = values.astype("string").str.strip()

    invalid = (
        result.isna()
        | result.eq("")
        | result.str.lower().eq("unknown")
    )

    return result.mask(invalid)


def _merge_lookup(
    table: pd.DataFrame,
    taxonomy: FrozenTaxonomyProvider,
    *,
    replace_taxon_name: bool,
) -> pd.DataFrame:
    """Merge frozen taxonomy names and ranks onto a table."""
    lookup = taxonomy.lookup_many(
        table["NCBI_ID"],
        include_unresolved=True,
    ).drop(columns="taxonomy_status")

    lookup = lookup.rename(
        columns={
            "Taxon_name": "_lookup_taxon_name",
            "Rank": "_lookup_rank",
        }
    )

    result = table.merge(
        lookup,
        on="NCBI_ID",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    result["Rank"] = result["_lookup_rank"].astype("string").str.lower()

    if replace_taxon_name:
        result["Taxon_name"] = result["_lookup_taxon_name"]
    elif "Taxon_name" not in result.columns:
        result["Taxon_name"] = result["_lookup_taxon_name"]

    return result.drop(
        columns=["_lookup_taxon_name", "_lookup_rank"]
    )


def _drop_exact_duplicates(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """R-compatible distinct() preserving first-occurrence order."""
    return table.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)


def _retain_highest_confidence(
    table: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    with_ties: bool,
) -> pd.DataFrame:
    """Reproduce dplyr::slice_max on ordered confidence factors.

    Confidence is converted to the ordered categorical produced by
    confidence_to_categorical(). Stable sorting preserves original row order
    among tied records.
    """
    result = table.copy()

    result["Confidence_in_curation"] = confidence_to_categorical(
        result["Confidence_in_curation"]
    )

    result["_input_order"] = np.arange(len(result))

    result = result.sort_values(
        [*group_columns, "Confidence_in_curation", "_input_order"],
        ascending=[
            *([True] * len(group_columns)),
            False,
            True,
        ],
        kind="stable",
        na_position="last",
    )

    if with_ties:
        maximum_confidence = result.groupby(
            list(group_columns),
            dropna=False,
            observed=True,
        )["Confidence_in_curation"].transform("max")

        is_maximum = (
            result["Confidence_in_curation"].eq(maximum_confidence)
            |
            (
                result["Confidence_in_curation"].isna()
                & maximum_confidence.isna()
            )
        )

        result = result.loc[
            is_maximum.fillna(False)
        ]
    else:
        result = result.drop_duplicates(
            subset=list(group_columns),
            keep="first",
        )

    return (
        result.sort_values("_input_order", kind="stable")
        .drop(columns="_input_order")
        .reset_index(drop=True)
    )


def _select_lexicographic_maximum_source(
    table: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    with_ties: bool,
) -> pd.DataFrame:
    """Reproduce slice_max(Attribute_source) behavior.

    Character values are ordered lexicographically, matching R's character
    ordering under the active locale. Python uses standard Unicode
    lexicographic ordering.

    For an ordered pandas categorical, the category ordering is used instead.
    For numeric source columns, numeric ordering is retained.
    """
    result = table.copy()
    result["_input_order"] = np.arange(len(result))

    source = result["Attribute_source"]

    if (
        isinstance(source.dtype, pd.CategoricalDtype)
        and source.dtype.ordered
    ):
        # Pandas categorical codes represent missing values as -1.
        # Convert them back to missing values so they behave like R's NA.
        result["_source_order"] = result["_source_order"].mask(
            result["_source_order"].eq(-1)
        )
    else:
        result["_source_order"] = source

    result = result.sort_values(
        [
            *group_columns,
            "_source_order",
            "_input_order",
        ],
        ascending=[
            *([True] * len(group_columns)),
            False,
            True,
        ],
        kind="stable",
        na_position="last",
    )

    if with_ties:
        maximum_source = result.groupby(
            list(group_columns),
            dropna=False,
            observed=True,
        )["_source_order"].transform("max")

        is_maximum = (
            result["_source_order"].eq(maximum_source)
            |
            (
                result["_source_order"].isna()
                & maximum_source.isna()
            )
        )

        result = result.loc[
            is_maximum.fillna(False)
        ]
    else:
        result = result.drop_duplicates(
            subset=list(group_columns),
            keep="first",
        )

    return (
        result.sort_values(
            "_input_order",
            kind="stable",
        )
        .drop(
            columns=[
                "_input_order",
                "_source_order",
            ]
        )
        .reset_index(drop=True)
    )


def _single_range_mean_by_taxid(
    table: pd.DataFrame,
) -> pd.Series:
    """Compute one mean across all retained min/max values per taxid.

    This reproduces the R expression:

        mean(cbind(Attribute_value_min, Attribute_value_max))

    within each NCBI_ID group.

    It is equivalent to flattening both columns together and calculating one
    scalar mean for the entire group. It is not a row-wise midpoint.
    """
    minimum = pd.to_numeric(
        table["Attribute_value_min"],
        errors="coerce",
    )
    maximum = pd.to_numeric(
        table["Attribute_value_max"],
        errors="coerce",
    )

    working = pd.DataFrame(
        {
            "NCBI_ID": table["NCBI_ID"],
            "Attribute_value_min": minimum,
            "Attribute_value_max": maximum,
        },
        index=table.index,
    )

    def group_mean(group: pd.DataFrame) -> float:
        values = group[
            ["Attribute_value_min", "Attribute_value_max"]
        ].to_numpy(dtype=float)

        return float(np.mean(values))

    means = working.groupby(
        "NCBI_ID",
        dropna=False,
        sort=False,
    ).apply(
        group_mean,
        include_groups=False,
    )

    return table["NCBI_ID"].map(means)

def _r_style_sum(
    values: pd.Series,
    *,
    na_rm: bool = False,
) -> float:
    """Approximate R's sum() using extended-precision accumulation."""
    total = np.longdouble(0.0)

    for value in values:
        if pd.isna(value):
            if na_rm:
                continue
            return np.nan

        total += np.longdouble(value)

    return float(total)

def _normalize_scores_by_taxid(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize retained scores within each taxid."""
    result = table.copy()

    result["Score"] = pd.to_numeric(
        result["Score"],
        errors="coerce",
    ).astype("float64")

    result["Total_score"] = result.groupby(
        "NCBI_ID",
        dropna=False,
        sort=False,
    )["Score"].transform(
        lambda values: values.sum(skipna=False)
    )

    result["Score"] = (
        result["Score"] / result["Total_score"]
    )

    return result

def _add_prefixed_taxid(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve raw taxid and replace NCBI_ID with rank-prefixed ID."""
    result = table.copy()

    result["taxid"] = result["NCBI_ID"].astype("string")

    result["NCBI_ID"] = add_rank_prefix(
        result["NCBI_ID"],
        result["Rank"],
    )

    return result.loc[
        result["NCBI_ID"].notna()
    ].copy()


def _relocate_ordered_columns(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Move known output columns first while preserving any extras."""
    present_ordered = [
        column
        for column in ORDERED_COLUMNS
        if column in table.columns
    ]

    remaining = [
        column
        for column in table.columns
        if column not in present_ordered
    ]

    return table.loc[
        :,
        [*present_ordered, *remaining],
    ]


def get_set_with_ids(
    table: pd.DataFrame,
    taxonomy: FrozenTaxonomyProvider,
) -> pd.DataFrame | None:
    """Prepare records that already contain direct NCBI taxonomy IDs.

    Equivalent to R ``getSetWithIDs``.

    Parameters
    ----------
    table
        Filtered phenotype table containing exactly one Attribute_type.
    taxonomy
        Frozen taxonomy provider used for taxon-name and rank lookup.

    Returns
    -------
    pandas.DataFrame or None
        Directly observed records prepared for propagation, or ``None`` when
        no valid directly identified records remain.
    """
    if table is None or table.empty:
        return None

    _require_columns(
        table,
        (
            "NCBI_ID",
            "Parent_NCBI_ID",
            "Taxon_name",
            "Attribute_source",
            "Confidence_in_curation",
            "Score",
            "Frequency",
            "Evidence",
            "Attribute_type",
            "Attribute_group",
        ),
        function_name="get_set_with_ids",
    )

    attribute_type = _normalize_attribute_type(
        _get_single_nonmissing_value(
            table["Attribute_type"],
            field_name="Attribute_type",
        )
    )

    result = table.copy()
    result["NCBI_ID"] = _normalize_taxid(result["NCBI_ID"])

    result = result.loc[
        result["NCBI_ID"].notna()
    ].copy()

    if result.empty:
        return None

    result = _merge_lookup(
        result,
        taxonomy,
        replace_taxon_name=True,
    )

    result = result.loc[
        result["Rank"].isin(VALID_RANKS)
    ].copy()

    if result.empty:
        return None

    result = _drop_exact_duplicates(result)

    if attribute_type == "range":
        _require_columns(
            result,
            (
                "Attribute_value_min",
                "Attribute_value_max",
            ),
            function_name="get_set_with_ids range input",
        )

        # R behavior:
        # 1. retain all highest-confidence ties by NCBI_ID;
        # 2. calculate one scalar mean across all min/max values in that group;
        # 3. slice_max confidence again with_ties = FALSE.
        result = _retain_highest_confidence(
            result,
            group_columns=("NCBI_ID",),
            with_ties=True,
        )

        result["Attribute_value"] = (
            _single_range_mean_by_taxid(result)
        )

        result = _retain_highest_confidence(
            result,
            group_columns=("NCBI_ID",),
            with_ties=False,
        )

        result = result.drop(
            columns=[
                "Attribute_value_min",
                "Attribute_value_max",
                "Parent_NCBI_ID",
            ],
            errors="ignore",
        )

        result = _drop_exact_duplicates(result)
        result = _add_prefixed_taxid(result)
        result = _drop_exact_duplicates(result)

        return result.reset_index(drop=True)

    if attribute_type not in {
        "binary",
        "multistate-intersection",
        "multistate-union",
    }:
        raise TaxonomySetError(
            "get_set_with_ids received unsupported Attribute_type "
            f"{attribute_type!r}."
        )

    _require_columns(
        result,
        ("Attribute",),
        function_name="get_set_with_ids discrete input",
    )

    # Keep all rows tied for maximum confidence per taxid.
    result = _retain_highest_confidence(
        result,
        group_columns=("NCBI_ID",),
        with_ties=True,
    )

    # Direct records use with_ties = FALSE for Attribute_source.
    result = _select_lexicographic_maximum_source(
        result,
        group_columns=("NCBI_ID", "Attribute"),
        with_ties=False,
    )

    result = _normalize_scores_by_taxid(result)

    result["Score"] = _clip_probability_scores(result["Score"])

    result["Frequency"] = score_to_frequency(
        result["Score"]
    )

    result = result.drop(
        columns=[
            "Parent_NCBI_ID",
            "Total_score",
        ],
        errors="ignore",
    )

    result = _add_prefixed_taxid(result)
    result = _drop_exact_duplicates(result)
    result = _relocate_ordered_columns(result)

    return result.reset_index(drop=True)


def get_set_without_ids(
    table: pd.DataFrame,
    taxonomy: FrozenTaxonomyProvider,
    set_with_ids: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Prepare records propagated from Parent_NCBI_ID.

    Equivalent to R ``getSetWithoutIDs``.

    Rows are considered here only when direct ``NCBI_ID`` is missing or
    ``"unknown"``. Their ``Parent_NCBI_ID`` becomes the working taxid.

    Parameters
    ----------
    table
        Filtered phenotype table containing exactly one Attribute_type and one
        Attribute_group.
    taxonomy
        Frozen taxonomy provider used for name and rank lookup.
    set_with_ids
        Prepared direct-ID records. Parent-derived records whose prefixed
        NCBI_ID already appears here are excluded.

    Returns
    -------
    pandas.DataFrame or None
        Parent-derived records, or ``None`` when the input contains no rows
        lacking direct taxids.
    """
    if table is None or table.empty:
        return None

    _require_columns(
        table,
        (
            "NCBI_ID",
            "Parent_NCBI_ID",
            "Taxon_name",
            "Attribute_source",
            "Confidence_in_curation",
            "Score",
            "Frequency",
            "Evidence",
            "Attribute_type",
            "Attribute_group",
        ),
        function_name="get_set_without_ids",
    )

    attribute_type = _normalize_attribute_type(
        _get_single_nonmissing_value(
            table["Attribute_type"],
            field_name="Attribute_type",
        )
    )

    attribute_group = _get_single_nonmissing_value(
        table["Attribute_group"],
        field_name="Attribute_group",
    )

    direct_taxid = _normalize_taxid(table["NCBI_ID"])
    missing_direct_id = direct_taxid.isna()

    if not missing_direct_id.any():
        return None

    if set_with_ids is None:
        direct_ids: set[str] = set()
    else:
        direct_ids = set(
            set_with_ids.get(
                "NCBI_ID",
                pd.Series(dtype="string"),
            )
            .astype("string")
            .dropna()
            .tolist()
        )

    result = table.loc[
        missing_direct_id
    ].copy()

    # Corrected interpretation of the R condition:
    # unique(tbl$Attribute_type) == "range"
    if attribute_type == "range":
        _require_columns(
            result,
            (
                "Attribute_value_min",
                "Attribute_value_max",
            ),
            function_name="get_set_without_ids range input",
        )

        result = result.drop(
            columns=[
                "NCBI_ID",
                "Taxon_name",
                "Frequency",
            ],
            errors="ignore",
        )

        result = result.rename(
            columns={
                "Parent_NCBI_ID": "NCBI_ID",
            }
        )

        result["NCBI_ID"] = _normalize_taxid(result["NCBI_ID"])
        result = _drop_exact_duplicates(result)

        result = _merge_lookup(
            result,
            taxonomy,
            replace_taxon_name=True,
        )

        result = result.loc[
            result["Rank"].isin(VALID_RANKS)
        ].copy()

        if result.empty:
            return None

        result = _retain_highest_confidence(
            result,
            group_columns=("NCBI_ID",),
            with_ties=True,
        )

        result["Attribute_value"] = (
            _single_range_mean_by_taxid(result)
        )

        result = _retain_highest_confidence(
            result,
            group_columns=("NCBI_ID",),
            with_ties=False,
        )

        result = result.drop(
            columns=[
                "Attribute_value_min",
                "Attribute_value_max",
            ],
            errors="ignore",
        )

        result = _drop_exact_duplicates(result)

        result["Score"] = _clip_probability_scores(result["Score"])

        result["Frequency"] = score_to_frequency(
            result["Score"]
        )

        result["Evidence"] = "tax"
        result["Attribute_group"] = attribute_group
        result["Attribute_type"] = attribute_type
        result["Attribute_source"] = pd.NA
        result["Confidence_in_curation"] = pd.NA

        result = _add_prefixed_taxid(result)

        result = result.loc[
            ~result["NCBI_ID"].isin(direct_ids)
        ].copy()

        result = _drop_exact_duplicates(result)

        return (
            result.sort_values(
                "NCBI_ID",
                kind="stable",
                na_position="last",
            )
            .reset_index(drop=True)
        )

    if attribute_type not in {
        "binary",
        "multistate-intersection",
        "multistate-union",
    }:
        raise TaxonomySetError(
            "get_set_without_ids received unsupported Attribute_type "
            f"{attribute_type!r}."
        )

    _require_columns(
        result,
        ("Attribute",),
        function_name="get_set_without_ids discrete input",
    )

    result = result.drop(
        columns=[
            "NCBI_ID",
            "Frequency",
        ],
        errors="ignore",
    )

    result = result.rename(
        columns={
            "Parent_NCBI_ID": "NCBI_ID",
        }
    )

    result["NCBI_ID"] = _normalize_taxid(result["NCBI_ID"])
    result = _drop_exact_duplicates(result)

    result = _merge_lookup(
        result,
        taxonomy,
        replace_taxon_name=False,
    )

    result = result.loc[
        result["Rank"].isin(VALID_RANKS)
    ].copy()

    if result.empty:
        return None

    # Parent-derived records preserve all maximum-confidence ties.
    result = _retain_highest_confidence(
        result,
        group_columns=("NCBI_ID",),
        with_ties=True,
    )

    # Parent-derived records also preserve all ties for maximum source.
    result = _select_lexicographic_maximum_source(
        result,
        group_columns=("NCBI_ID", "Attribute"),
        with_ties=True,
    )

    # Normalize before duplicate parent/attribute records are summed, matching
    # the R pipeline order.
    result = _normalize_scores_by_taxid(result)

    result["Score"] = result.groupby(
        ["NCBI_ID", "Attribute"],
        dropna=False,
        sort=False,
    )["Score"].transform(
        lambda values: values.sum(skipna=True)
    )

    result = result.drop(
        columns=["Taxon_name"],
        errors="ignore",
    )

    # Parent taxon name should come from the frozen lookup after aggregation.
    lookup = taxonomy.lookup_many(
        result["NCBI_ID"],
        include_unresolved=True,
    )[
        ["NCBI_ID", "Taxon_name"]
    ]

    result = result.merge(
        lookup,
        on="NCBI_ID",
        how="left",
        sort=False,
        validate="many_to_one",
    )

    result = _drop_exact_duplicates(result)

    result["Score"] = _clip_probability_scores(result["Score"])

    result["Frequency"] = score_to_frequency(
        result["Score"]
    )

    result["Evidence"] = "tax"
    result["Attribute_group"] = attribute_group
    result["Attribute_type"] = attribute_type
    result["Attribute_source"] = pd.NA
    result["Confidence_in_curation"] = pd.NA

    result = result.drop(
        columns=["Total_score"],
        errors="ignore",
    )

    result = _add_prefixed_taxid(result)

    result = result.loc[
        ~result["NCBI_ID"].isin(direct_ids)
    ].copy()

    result = _drop_exact_duplicates(result)

    result = result.sort_values(
        ["NCBI_ID", "Attribute"],
        kind="stable",
        na_position="last",
    )

    result = _relocate_ordered_columns(result)

    return result.reset_index(drop=True)


# R-compatible aliases
getSetWithIDs = get_set_with_ids
getSetWithoutIDs = get_set_without_ids
