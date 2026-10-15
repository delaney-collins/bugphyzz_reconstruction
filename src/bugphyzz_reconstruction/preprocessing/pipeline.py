"""Trait-data preparation pipeline translated from taxPPro.R.

This module implements the Python equivalent of the R function:

    getDataReady()

It orchestrates:

- direct-taxonomy preparation with get_set_with_ids();
- parent-taxonomy preparation with get_set_without_ids();
- binary trait completion;
- multistate-intersection completion;
- range trait assembly;
- multistate-union splitting and completion.

Filtering remains in filtering.py, taxonomy conflict resolution remains in
taxonomy_sets.py, and completion helpers remain in completion.py.
"""

from __future__ import annotations

import pandas as pd

from bugphyzz_reconstruction.preprocessing.completion import (
    complete_binary_data,
    complete_multistate_union,
)
from bugphyzz_reconstruction.preprocessing.taxonomy_sets import (
    get_set_with_ids,
    get_set_without_ids,
)
from bugphyzz_reconstruction.taxonomy.frozen import FrozenTaxonomyProvider


SUPPORTED_ATTRIBUTE_TYPES: frozenset[str] = frozenset(
    {
        "binary",
        "multistate-intersection",
        "multistate-union",
        "range",
    }
)


class TraitPreparationError(ValueError):
    """Raised when trait data cannot be prepared consistently."""


def _normalize_attribute_type(attribute_type: str) -> str:
    """Normalize supported attribute-type spellings."""
    normalized = attribute_type.strip().lower()

    aliases = {
        # Retained only as defensive compatibility with historical input.
        # The Python filtering dispatcher emits the corrected spelling.
        "mltistate-union": "multistate-union",
    }

    return aliases.get(normalized, normalized)


def _get_single_attribute_type(
    table: pd.DataFrame,
) -> str:
    """Return exactly one non-missing Attribute_type.

    The R implementation assumes that unique(tbl$Attribute_type) returns one
    value. Python checks that assumption explicitly.
    """
    if "Attribute_type" not in table.columns:
        raise TraitPreparationError(
            "Input table is missing required column: Attribute_type"
        )

    attribute_types = (
        table["Attribute_type"]
        .astype("string")
        .dropna()
        .str.strip()
        .map(_normalize_attribute_type)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if len(attribute_types) != 1:
        raise TraitPreparationError(
            "Expected exactly one non-missing Attribute_type, "
            f"but found {attribute_types.tolist()}."
        )

    attribute_type = str(attribute_types.iloc[0])

    if attribute_type not in SUPPORTED_ATTRIBUTE_TYPES:
        raise TraitPreparationError(
            f"Unsupported Attribute_type: {attribute_type!r}. "
            "Supported values are: "
            + ", ".join(sorted(SUPPORTED_ATTRIBUTE_TYPES))
        )

    return attribute_type


def _drop_all_missing_columns(
    table: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Drop columns whose values are all missing.

    This expresses the intended behavior of the R pipeline:

        purrr::discard(~ all(is.na(.x)))

    when applied to a data frame.
    """
    if table is None:
        return None

    table = table.dropna(
        axis="columns",
        how="all",
    )

    return table


def _combine_optional_tables(
    *tables: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine non-empty tables like dplyr::bind_rows().

    None and empty tables are ignored. Columns are unioned, and missing values
    are inserted where one input lacks a column.
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


def _complete_multistate_intersection(
    table: pd.DataFrame,
) -> pd.DataFrame:
    """Complete NCBI_ID × Attribute combinations with Score = 0.

    This extracts the R expression:

        tidyr::complete(
            NCBI_ID,
            Attribute,
            fill = list(Score = 0)
        ) |>
        dplyr::arrange(NCBI_ID, Attribute)

    used for multistate-intersection traits.
    """
    required_columns = {
        "NCBI_ID",
        "Attribute",
        "Score",
    }

    missing_columns = sorted(
        required_columns.difference(table.columns)
    )

    if missing_columns:
        raise TraitPreparationError(
            "Cannot complete multistate-intersection data; "
            "missing required columns: "
            + ", ".join(missing_columns)
        )

    if table.empty:
        return table.copy().reset_index(drop=True)

    duplicated = table.duplicated(
        subset=["NCBI_ID", "Attribute"],
        keep=False,
    )

    if duplicated.any():
        examples = (
            table.loc[
                duplicated,
                ["NCBI_ID", "Attribute"],
            ]
            .drop_duplicates()
            .head(10)
        )

        raise TraitPreparationError(
            "Multistate-intersection completion requires unique "
            "NCBI_ID/Attribute combinations. Duplicate combinations include:\n"
            + examples.to_string(index=False)
        )

    ncbi_ids = (
        table["NCBI_ID"]
        .dropna()
        .drop_duplicates(keep="first")
        .tolist()
    )

    attributes = (
        table["Attribute"]
        .dropna()
        .drop_duplicates(keep="first")
        .tolist()
    )

    if not ncbi_ids or not attributes:
        return table.copy().reset_index(drop=True)

    complete_grid = pd.MultiIndex.from_product(
        [ncbi_ids, attributes],
        names=["NCBI_ID", "Attribute"],
    ).to_frame(index=False)

    output = complete_grid.merge(
        table,
        on=["NCBI_ID", "Attribute"],
        how="left",
        sort=False,
        validate="one_to_one",
    )

    output["Score"] = pd.to_numeric(
        output["Score"],
        errors="coerce",
    ).fillna(0.0)

    # R's arrange(NCBI_ID, Attribute) uses the natural ordering of the
    # completed columns. Stable sorting makes tie behavior deterministic.
    output = output.sort_values(
        ["NCBI_ID", "Attribute"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    return output


def _prepare_direct_and_parent_sets(
    table: pd.DataFrame,
    taxonomy: FrozenTaxonomyProvider,
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame,
]:
    """Prepare, clean, and combine direct- and parent-taxonomy records."""
    set_with_ids = get_set_with_ids(
        table,
        taxonomy,
    )

    set_with_ids = _drop_all_missing_columns(
        set_with_ids
    )

    set_without_ids = get_set_without_ids(
        table,
        taxonomy,
        set_with_ids=set_with_ids,
    )

    set_without_ids = _drop_all_missing_columns(
        set_without_ids
    )

    dataset = _combine_optional_tables(
        set_with_ids,
        set_without_ids,
    )

    return (
        set_with_ids,
        set_without_ids,
        dataset,
    )


def prepare_trait_data(
    table: pd.DataFrame | None,
    taxonomy: FrozenTaxonomyProvider,
) -> pd.DataFrame | dict[str, pd.DataFrame] | None:
    """Prepare a filtered phenotype table for propagation or ASR.

    Equivalent to the R function ``getDataReady``.

    Parameters
    ----------
    table
        A table previously processed by ``filter_data``. It must contain
        exactly one supported ``Attribute_type``.
    taxonomy
        Frozen taxonomy provider used by the direct- and parent-taxonomy
        preparation functions.

    Returns
    -------
    pandas.DataFrame, dict[str, pandas.DataFrame], or None
        Binary
            Completed NCBI_ID × Attribute table.

        Multistate-intersection
            Completed NCBI_ID × Attribute table with absent states assigned
            Score = 0.

        Range
            Direct and parent-derived numeric records sorted by NCBI_ID.

        Multistate-union
            Dictionary mapping each base attribute to an independently
            prepared binary table.

        None
            Returned when the input is None, empty, or produces no usable
            direct or parent-derived records.
    """
    if table is None:
        return None

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            "table must be a pandas DataFrame or None."
        )

    if table.empty:
        return None

    attribute_type = _get_single_attribute_type(
        table
    )

    # Ensure downstream modules see the normalized spelling.
    working = table.copy()
    working["Attribute_type"] = attribute_type

    if attribute_type == "multistate-union":
        output = complete_multistate_union(
            working,
            taxonomy,
        )

        return output or None

    (
        set_with_ids,
        set_without_ids,
        dataset,
    ) = _prepare_direct_and_parent_sets(
        working,
        taxonomy,
    )

    # This is the effective intent of:
    #
    # if (is.null(set_with_ids) && is.null(set_without_ids))
    #     return(NULL)
    #
    # Checking the combined dataset also handles empty DataFrames cleanly.
    if dataset.empty:
        return None

    if attribute_type == "binary":
        return complete_binary_data(
            dataset
        )

    if attribute_type == "multistate-intersection":
        return _complete_multistate_intersection(
            dataset
        )

    if attribute_type == "range":
        if "NCBI_ID" not in dataset.columns:
            raise TraitPreparationError(
                "Prepared range data is missing required column: NCBI_ID"
            )

        return (
            dataset.sort_values(
                "NCBI_ID",
                kind="stable",
                na_position="last",
            )
            .reset_index(drop=True)
        )

    # This branch is unreachable because _get_single_attribute_type validates
    # supported types, but it provides a defensive failure if that changes.
    raise TraitPreparationError(
        f"Unhandled Attribute_type: {attribute_type!r}"
    )


# ---------------------------------------------------------------------------
# R-compatible alias
# ---------------------------------------------------------------------------

getDataReady = prepare_trait_data
