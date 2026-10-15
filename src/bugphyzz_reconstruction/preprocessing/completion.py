"""Trait-state completion functions translated from taxPPro.R.

This module implements the data-completion portions of the original R
preprocessing workflow:

- complete_binary_data
- complete_multistate_union

Binary completion creates all observed NCBI_ID × Attribute combinations and
fills missing scores with zero.

Multistate-union completion splits a trait table into its underlying binary
attributes, prepares each subset independently, and applies binary completion
to each resulting dataset.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from bugphyzz_reconstruction.preprocessing.taxonomy_sets import (
    get_set_with_ids,
    get_set_without_ids,
)
from bugphyzz_reconstruction.taxonomy.frozen import FrozenTaxonomyProvider


class TraitCompletionError(ValueError):
    """Raised when trait-state completion cannot be performed consistently."""


def _normalize_attribute_type(value: str) -> str:
    """Normalize supported attribute-type names."""
    normalized = value.strip().lower()

    aliases = {
        "mltistate-union": "multistate-union",
    }

    return aliases.get(normalized, normalized)


def _get_single_attribute_type(
    table: pd.DataFrame,
    *,
    expected: str | None = None,
) -> str:
    """Return and validate one unique non-missing Attribute_type."""
    if "Attribute_type" not in table.columns:
        raise TraitCompletionError(
            "Input table is missing required column: Attribute_type"
        )

    values = (
        table["Attribute_type"]
        .astype("string")
        .dropna()
        .str.strip()
        .map(_normalize_attribute_type)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if len(values) != 1:
        raise TraitCompletionError(
            "Expected exactly one non-missing Attribute_type, "
            f"but found {values.tolist()}."
        )

    attribute_type = str(values.iloc[0])

    if expected is not None and attribute_type != expected:
        raise TraitCompletionError(
            f"Expected Attribute_type {expected!r}, "
            f"but found {attribute_type!r}."
        )

    return attribute_type


def _combine_optional_tables(
    *tables: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine non-empty DataFrames while preserving their row order.

    This reproduces the effective behavior of dplyr::bind_rows() when one or
    more inputs are NULL.
    """
    usable_tables = [
        table
        for table in tables
        if table is not None and not table.empty
    ]

    if not usable_tables:
        return pd.DataFrame()

    return pd.concat(
        usable_tables,
        axis=0,
        ignore_index=True,
        sort=False,
    )


def _ordered_unique(
    values: Iterable[object],
) -> list[object]:
    """Return unique non-missing values in first-occurrence order."""
    return (
        pd.Series(list(values))
        .dropna()
        .drop_duplicates(keep="first")
        .tolist()
    )


def _complete_id_attribute_grid(
    table: pd.DataFrame,
    *,
    fill_score: float = 0.0,
    attribute_levels: list[object] | None = None,
) -> pd.DataFrame:
    """Create all NCBI_ID × Attribute combinations.

    This is the pandas equivalent of:

        tidyr::complete(NCBI_ID, Attribute, fill = list(Score = 0))

    Values from existing rows are preserved. Newly created combinations have
    ``Score`` filled with ``fill_score`` and other metadata columns left
    missing.

    Parameters
    ----------
    table
        Input trait table.
    fill_score
        Score assigned to newly created ID/attribute combinations.
    attribute_levels
        Optional explicit Attribute ordering. When omitted, first-occurrence
        ordering from the table is used.

    Returns
    -------
    pandas.DataFrame
        Completed and sorted trait table.
    """
    required = {"NCBI_ID", "Attribute", "Score"}
    missing = sorted(required.difference(table.columns))

    if missing:
        raise TraitCompletionError(
            "Cannot complete ID/attribute grid; missing columns: "
            + ", ".join(missing)
        )

    if table.empty:
        return table.copy().reset_index(drop=True)

    result = table.copy()

    ncbi_ids = _ordered_unique(result["NCBI_ID"])

    if attribute_levels is None:
        attributes = _ordered_unique(result["Attribute"])
    else:
        attributes = [
            value
            for value in attribute_levels
            if pd.notna(value)
        ]

    if not ncbi_ids or not attributes:
        return result.reset_index(drop=True)

    grid = pd.MultiIndex.from_product(
        [ncbi_ids, attributes],
        names=["NCBI_ID", "Attribute"],
    ).to_frame(index=False)

    # tidyr::complete() expects each NCBI_ID/Attribute combination to identify
    # at most one existing record. The upstream taxonomy-set functions should
    # already guarantee this.
    duplicated = result.duplicated(
        subset=["NCBI_ID", "Attribute"],
        keep=False,
    )

    if duplicated.any():
        examples = (
            result.loc[
                duplicated,
                ["NCBI_ID", "Attribute"],
            ]
            .drop_duplicates()
            .head(10)
        )

        raise TraitCompletionError(
            "Completion requires unique NCBI_ID/Attribute combinations. "
            "Duplicate combinations include:\n"
            + examples.to_string(index=False)
        )

    completed = grid.merge(
        result,
        on=["NCBI_ID", "Attribute"],
        how="left",
        sort=False,
        validate="one_to_one",
    )

    completed["Score"] = pd.to_numeric(
        completed["Score"],
        errors="coerce",
    ).fillna(fill_score)

    # Reproduce arrange(NCBI_ID, Attribute), while retaining explicit binary
    # factor ordering when supplied.
    completed["_id_order"] = pd.Categorical(
        completed["NCBI_ID"],
        categories=ncbi_ids,
        ordered=True,
    )

    completed["_attribute_order"] = pd.Categorical(
        completed["Attribute"],
        categories=attributes,
        ordered=True,
    )

    completed = (
        completed.sort_values(
            ["_id_order", "_attribute_order"],
            kind="stable",
        )
        .drop(
            columns=[
                "_id_order",
                "_attribute_order",
            ]
        )
        .reset_index(drop=True)
    )

    return completed


def complete_binary_data(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Complete states and scores for a binary trait dataset.

    Equivalent to the R function ``completeBinaryData``.

    All observed NCBI IDs receive a row for every observed binary state.
    Missing combinations receive ``Score = 0``.

    The original R function contains one special case: when the only observed
    attribute ends in ``--TRUE``, it creates the corresponding ``--FALSE``
    factor level before completing the table. This behavior is reproduced
    explicitly.

    Parameters
    ----------
    table
        Prepared binary trait records.

    Returns
    -------
    pandas.DataFrame
        Completed table sorted by NCBI_ID and binary attribute state.
    """
    if table is None:
        raise TypeError(
            "table must be a pandas DataFrame, not None."
        )

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            "table must be a pandas DataFrame."
        )

    if table.empty:
        return table.copy().reset_index(drop=True)

    required = {"NCBI_ID", "Attribute", "Score"}
    missing = sorted(required.difference(table.columns))

    if missing:
        raise TraitCompletionError(
            "complete_binary_data is missing required columns: "
            + ", ".join(missing)
        )

    result = table.copy()

    current_attributes = _ordered_unique(
        result["Attribute"]
    )

    attribute_levels: list[object] | None = None

    if (
        len(current_attributes) == 1
        and str(current_attributes[0]).endswith("--TRUE")
    ):
        true_level = str(current_attributes[0])
        false_level = true_level.removesuffix("--TRUE") + "--FALSE"

        # Reproduce:
        #
        # factor(
        #     Attribute,
        #     levels = c(extra_level, current_attr)
        # )
        #
        # FALSE must sort before TRUE in this special case.
        attribute_levels = [
            false_level,
            true_level,
        ]

    return _complete_id_attribute_grid(
        result,
        fill_score=0.0,
        attribute_levels=attribute_levels,
    )


def complete_multistate_union(
    table: pd.DataFrame,
    taxonomy: FrozenTaxonomyProvider,
) -> dict[str, pd.DataFrame]:
    """Prepare and complete a multistate-union trait table.

    This extracts the ``multistate-union`` branch from the original
    ``getDataReady`` function.

    During discrete filtering, multistate-union attributes are represented as:

        attribute_name--TRUE
        attribute_name--FALSE

    The R implementation removes the Boolean suffix to identify the underlying
    trait, splits the table by that trait, prepares each subset independently,
    and then treats each subset as binary data.

    Parameters
    ----------
    table
        A filtered multistate-union phenotype table.
    taxonomy
        Frozen taxonomy provider used by the direct- and parent-taxonomy
        preparation functions.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from the base attribute name to its completed binary table.
        Attributes that produce no usable records are omitted.
    """
    if table is None:
        raise TypeError(
            "table must be a pandas DataFrame, not None."
        )

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            "table must be a pandas DataFrame."
        )

    if table.empty:
        return {}

    _get_single_attribute_type(
        table,
        expected="multistate-union",
    )

    required = {
        "Attribute",
        "Attribute_type",
    }

    missing = sorted(required.difference(table.columns))

    if missing:
        raise TraitCompletionError(
            "complete_multistate_union is missing required columns: "
            + ", ".join(missing)
        )

    result = table.copy()

    # Equivalent to:
    #
    # tbl$Attribute_group_2 <- sub(
    #     '--(TRUE|FALSE)',
    #     '',
    #     tbl$Attribute
    # )
    #
    # Using an end anchor prevents accidental removal from the middle of an
    # attribute name.
    result["Attribute_group_2"] = (
        result["Attribute"]
        .astype("string")
        .str.replace(
            r"--(?:TRUE|FALSE)$",
            "",
            regex=True,
        )
    )

    missing_group = result["Attribute_group_2"].isna()

    if missing_group.any():
        raise TraitCompletionError(
            "Some multistate-union rows have missing Attribute values."
        )

    base_attributes = (
        result["Attribute_group_2"]
        .drop_duplicates(keep="first")
        .tolist()
    )

    output: dict[str, pd.DataFrame] = {}

    for base_attribute in base_attributes:
        subset = result.loc[
            result["Attribute_group_2"].eq(base_attribute)
        ].drop(
            columns="Attribute_group_2"
        ).copy()

        set_with_ids = get_set_with_ids(
            subset,
            taxonomy,
        )

        set_without_ids = get_set_without_ids(
            subset,
            taxonomy,
            set_with_ids=set_with_ids,
        )

        dataset = _combine_optional_tables(
            set_with_ids,
            set_without_ids,
        )

        # Equivalent to the R loop's:
        #
        # if (!nrow(dataset))
        #     next
        if dataset.empty:
            continue

        output[str(base_attribute)] = complete_binary_data(
            dataset
        )

    return output


# ---------------------------------------------------------------------------
# Optional R-compatible aliases
# ---------------------------------------------------------------------------

completeBinaryData = complete_binary_data
completeMultistateUnion = complete_multistate_union

