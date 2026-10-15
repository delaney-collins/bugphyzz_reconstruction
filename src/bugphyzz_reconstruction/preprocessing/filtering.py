"""Filtering functions translated from taxPPro.R.

This module implements the first stage of the taxPPro propagation workflow:

- filter_data
- filter_data_numeric
- filter_data_discrete

The functions retain only rows that can be used for taxonomic propagation or
ancestral-state reconstruction.

The implementation uses vectorized pandas operations and frozen Bugphyzz
resources rather than live NCBI or R package lookups.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from bugphyzz_reconstruction.preprocessing.utils import frequency_to_score
from bugphyzz_reconstruction.taxonomy.frozen import FrozenTaxonomyProvider


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCRETE_ATTRIBUTE_TYPES: frozenset[str] = frozenset(
    {
        "binary",
        "multistate-intersection",
        "multistate-union",
    }
)

NUMERIC_ATTRIBUTE_TYPES: frozenset[str] = frozenset(
    {
        "range",
    }
)

NUMERIC_OUTPUT_COLUMNS: tuple[str, ...] = (
    "NCBI_ID",
    "Taxon_name",
    "Parent_NCBI_ID",
    "Attribute_value_min",
    "Attribute_value_max",
    "Attribute_source",
    "Confidence_in_curation",
    "Frequency",
    "Score",
    "Evidence",
    "Attribute_type",
    "Attribute_group",
)

DISCRETE_OUTPUT_COLUMNS: tuple[str, ...] = (
    "NCBI_ID",
    "Taxon_name",
    "Parent_NCBI_ID",
    "Attribute",
    "Attribute_source",
    "Confidence_in_curation",
    "Frequency",
    "Score",
    "Evidence",
    "Attribute_type",
    "Attribute_group",
)


class TraitFilteringError(ValueError):
    """Raised when trait input cannot be filtered consistently."""


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _require_columns(
    table: pd.DataFrame,
    required_columns: Sequence[str],
    *,
    function_name: str,
) -> None:
    """Validate that a table contains the required columns."""
    missing = sorted(set(required_columns).difference(table.columns))

    if missing:
        raise TraitFilteringError(
            f"{function_name} is missing required columns: "
            + ", ".join(missing)
        )


def _ensure_taxonomy_id_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Restore optional direct/parent ID columns removed when entirely empty."""
    result = table.copy()
    for column in ("NCBI_ID", "Parent_NCBI_ID"):
        if column not in result:
            result[column] = pd.NA
    return result


def _get_single_nonmissing_value(
    values: pd.Series,
    *,
    field_name: str,
) -> str:
    """Return one unique non-missing value from a Series.

    The R implementation assumes that each input table contains exactly one
    Attribute_type and one Attribute_group. Python checks this explicitly so
    mixed inputs fail clearly rather than producing ambiguous behavior.
    """
    unique_values = (
        values.astype("string")
        .dropna()
        .str.strip()
        .drop_duplicates()
    )

    if len(unique_values) != 1:
        raise TraitFilteringError(
            f"Expected exactly one non-missing {field_name}, "
            f"but found {unique_values.tolist()}."
        )

    return str(unique_values.iloc[0])


def _normalize_attribute_type(attribute_type: str) -> str:
    """Normalize supported attribute-type spellings.

    The original R dispatcher used the typo ``mltistate-union``. The Python
    implementation uses the corrected spelling ``multistate-union``.
    """
    normalized = attribute_type.strip().lower()

    aliases = {
        "mltistate-union": "multistate-union",
    }

    return aliases.get(normalized, normalized)


def _normalize_taxonomy_id_column(
    values: pd.Series,
) -> pd.Series:
    """Normalize NCBI taxonomy identifiers while preserving row index."""
    normalized = values.astype("string").str.strip()

    invalid = (
        normalized.isna()
        | normalized.eq("")
        | normalized.str.lower().eq("unknown")
    )

    return normalized.mask(invalid)


def _normalize_string_column(
    values: pd.Series,
) -> pd.Series:
    """Strip surrounding whitespace from a nullable string column."""
    return values.astype("string").str.strip()


def _ensure_output_columns(
    table: pd.DataFrame,
    columns: Sequence[str],
    *,
    function_name: str,
) -> pd.DataFrame:
    """Return a table containing output columns in a fixed order."""
    _require_columns(
        table,
        columns,
        function_name=function_name,
    )

    return table.loc[:, list(columns)].copy()


def _format_count(value: int) -> str:
    """Format an integer using the comma grouping used by R messages."""
    return f"{value:,}"


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def filter_data(
    table: pd.DataFrame,
    *,
    taxonomy: FrozenTaxonomyProvider | None = None,
) -> pd.DataFrame | None:
    """Filter a Bugphyzz trait table for propagation.

    Equivalent to the R function ``filterData``.

    Parameters
    ----------
    table
        DataFrame returned from the Bugphyzz physiology import layer.
        The table must contain exactly one non-missing ``Attribute_type``.
    taxonomy
        Frozen taxonomy provider. Required for discrete attributes because
        ``attributes.tsv`` is used to validate allowed attribute terms.

    Returns
    -------
    pandas.DataFrame or None
        Filtered trait data, or ``None`` for unsupported attribute types or
        when numeric filtering leaves no usable rows.
    """
    if table is None:
        return None

    if not isinstance(table, pd.DataFrame):
        raise TypeError(
            "table must be a pandas DataFrame."
        )

    if table.empty:
        return None

    _require_columns(
        table,
        ("Attribute_type",),
        function_name="filter_data",
    )

    raw_attribute_type = _get_single_nonmissing_value(
        table["Attribute_type"],
        field_name="Attribute_type",
    )

    attribute_type = _normalize_attribute_type(raw_attribute_type)

    # Apply the normalized attribute type to a copy so downstream functions
    # always receive the corrected spelling.
    normalized_table = table.copy()
    normalized_table["Attribute_type"] = attribute_type

    if attribute_type in DISCRETE_ATTRIBUTE_TYPES:
        if taxonomy is None:
            raise TraitFilteringError(
                "taxonomy must be supplied when filtering discrete traits "
                "because the valid attribute vocabulary is loaded from "
                "attributes.tsv."
            )

        if taxonomy.attributes is None:
            raise TraitFilteringError(
                "The FrozenTaxonomyProvider was loaded without attributes.tsv. "
                "Reload it with load_attributes=True."
            )

        return filter_data_discrete(
            normalized_table,
            valid_attributes_table=taxonomy.attributes,
        )

    if attribute_type in NUMERIC_ATTRIBUTE_TYPES:
        return filter_data_numeric(normalized_table)

    return None


# ---------------------------------------------------------------------------
# Numeric filtering
# ---------------------------------------------------------------------------

def filter_data_numeric(
    table: pd.DataFrame,
) -> pd.DataFrame | None:
    """Filter numeric/range trait records for propagation.

    Equivalent to the R function ``filterDataNumeric``.

    Rows are retained only when:

    - ``Attribute_value_min`` is present and finite;
    - ``Attribute_value_max`` is present and finite;
    - either ``NCBI_ID`` or ``Parent_NCBI_ID`` is available;
    - ``Attribute_source`` is present;
    - ``Frequency`` is present.

    A numeric ``Score`` is derived from ``Frequency``.

    Parameters
    ----------
    table
        A single numeric/range phenotype table.

    Returns
    -------
    pandas.DataFrame or None
        Filtered table with the same output-column order as the R function.
        Returns ``None`` if no rows remain.
    """
    if table is None or table.empty:
        return None

    # physiologies() drops columns that are entirely missing. Direct and
    # parent IDs are alternatives, so either column may legitimately vanish.
    table = _ensure_taxonomy_id_columns(table)

    required_columns = (
        "NCBI_ID",
        "Taxon_name",
        "Parent_NCBI_ID",
        "Attribute_value_min",
        "Attribute_value_max",
        "Attribute_source",
        "Confidence_in_curation",
        "Frequency",
        "Evidence",
        "Attribute_type",
        "Attribute_group",
    )

    _require_columns(
        table,
        required_columns,
        function_name="filter_data_numeric",
    )

    working = table.copy()
    original_row_count = len(working)

    # Normalize taxonomy identifiers before evaluating whether a usable direct
    # or parent taxid is present.
    working["NCBI_ID"] = _normalize_taxonomy_id_column(
        working["NCBI_ID"]
    )
    working["Parent_NCBI_ID"] = _normalize_taxonomy_id_column(
        working["Parent_NCBI_ID"]
    )

    # Reproduce R's numeric coercion assumptions while allowing invalid text
    # values to become missing and therefore be removed.
    minimum = pd.to_numeric(
        working["Attribute_value_min"],
        errors="coerce",
    )
    maximum = pd.to_numeric(
        working["Attribute_value_max"],
        errors="coerce",
    )

    usable_numeric_values = (
        minimum.notna()
        & maximum.notna()
        & np.isfinite(minimum)
        & np.isfinite(maximum)
    )

    usable_taxonomy = (
        working["NCBI_ID"].notna()
        | working["Parent_NCBI_ID"].notna()
    )

    usable_metadata = (
        working["Attribute_source"].notna()
        & working["Frequency"].notna()
    )

    keep_mask = (
        usable_numeric_values
        & usable_taxonomy
        & usable_metadata
    )

    filtered = working.loc[keep_mask].copy()

    filtered["Attribute_value_min"] = minimum.loc[keep_mask]
    filtered["Attribute_value_max"] = maximum.loc[keep_mask]

    filtered["Score"] = frequency_to_score(
        filtered["Frequency"]
    )

    filtered = _ensure_output_columns(
        filtered,
        NUMERIC_OUTPUT_COLUMNS,
        function_name="filter_data_numeric",
    )

    # pandas drop_duplicates matches dplyr::distinct() for complete rows.
    filtered = filtered.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)

    rows_dropped = original_row_count - len(filtered)

    if filtered.empty:
        print(
            "Not enough data. Dropping :"
            f"{_format_count(rows_dropped)} rows."
        )
        return None

    print(
        f"{_format_count(rows_dropped)} rows were dropped."
    )

    return filtered


# ---------------------------------------------------------------------------
# Discrete filtering
# ---------------------------------------------------------------------------

def filter_data_discrete(
    table: pd.DataFrame,
    *,
    valid_attributes_table: pd.DataFrame,
) -> pd.DataFrame:
    """Filter discrete trait records for propagation.

    Equivalent to the R function ``filterDataDiscrete``.

    The function:

    - identifies valid attributes for the current phenotype group;
    - removes rows lacking ``Attribute`` or ``Attribute_value``;
    - warns about invalid attribute terms;
    - handles binary, multistate-intersection, and multistate-union traits;
    - requires a direct or parent NCBI ID;
    - requires ``Attribute_source`` and ``Frequency``;
    - converts frequency labels to numeric scores;
    - removes duplicate rows.

    Parameters
    ----------
    table
        A single discrete phenotype table.
    valid_attributes_table
        Frozen ``attributes.tsv`` table.

    Returns
    -------
    pandas.DataFrame
        Filtered discrete trait data.
    """
    if table is None or table.empty:
        return pd.DataFrame(
            columns=list(DISCRETE_OUTPUT_COLUMNS)
        )

    table = _ensure_taxonomy_id_columns(table)

    required_columns = (
        "NCBI_ID",
        "Taxon_name",
        "Parent_NCBI_ID",
        "Attribute",
        "Attribute_value",
        "Attribute_source",
        "Confidence_in_curation",
        "Frequency",
        "Evidence",
        "Attribute_type",
        "Attribute_group",
    )

    _require_columns(
        table,
        required_columns,
        function_name="filter_data_discrete",
    )

    _require_columns(
        valid_attributes_table,
        ("attribute_group", "attribute"),
        function_name="filter_data_discrete attributes table",
    )

    working = table.copy()
    original_row_count = len(working)

    physiology_name = _get_single_nonmissing_value(
        working["Attribute_group"],
        field_name="Attribute_group",
    )

    raw_attribute_type = _get_single_nonmissing_value(
        working["Attribute_type"],
        field_name="Attribute_type",
    )

    attribute_type = _normalize_attribute_type(
        raw_attribute_type
    )

    working["Attribute_type"] = attribute_type

    # Match the R grepl("\\b<phys_name>\\b", attribute_group) behavior.
    #
    # re.escape prevents punctuation in physiology names from being treated
    # as regular-expression syntax.
    physiology_pattern = re.compile(
        rf"\b{re.escape(physiology_name)}\b",
        flags=re.IGNORECASE,
    )

    attribute_groups = (
        valid_attributes_table["attribute_group"]
        .astype("string")
        .fillna("")
    )

    group_match = attribute_groups.str.contains(
        physiology_pattern,
        regex=True,
        na=False,
    )

    valid_attributes = (
        valid_attributes_table.loc[
            group_match,
            "attribute",
        ]
        .astype("string")
        .dropna()
        .drop_duplicates()
    )

    valid_attribute_set = set(valid_attributes.tolist())

    # R removes missing Attribute and Attribute_value before checking invalid
    # attributes.
    working["Attribute"] = _normalize_string_column(
        working["Attribute"]
    )

    working = working.loc[
        working["Attribute"].notna()
        & working["Attribute_value"].notna()
    ].copy()

    observed_attributes = (
        working["Attribute"]
        .dropna()
        .drop_duplicates()
    )

    invalid_attributes = [
        attribute
        for attribute in observed_attributes.tolist()
        if attribute not in valid_attribute_set
    ]

    if invalid_attributes:
        invalid_string = "---".join(
            str(attribute)
            for attribute in invalid_attributes
        )

        warnings.warn(
            f"Invalid attributes in {physiology_name}: "
            f"{invalid_string}",
            category=UserWarning,
            stacklevel=2,
        )

    # Keep only valid attributes before applying type-specific reshaping.
    working = working.loc[
        working["Attribute"].isin(valid_attribute_set)
    ].copy()

    if attribute_type == "multistate-intersection":
        # R uses Attribute_value == TRUE. Convert common representations of
        # logical values without applying row-wise parsing.
        attribute_values = working["Attribute_value"]

        if pd.api.types.is_bool_dtype(attribute_values):
            true_mask = attribute_values.fillna(False)
        else:
            normalized_values = (
                attribute_values.astype("string")
                .str.strip()
                .str.lower()
            )

            true_mask = normalized_values.isin(
                {
                    "true",
                    "t",
                    "1",
                }
            )

        working = working.loc[true_mask].copy()

    elif attribute_type in {
        "binary",
        "multistate-union",
    }:
        # Reproduce paste0(Attribute, "--", Attribute_value).
        #
        # R renders logical values as TRUE/FALSE. Normalize common boolean
        # representations so the Python output follows that convention.
        attribute_values = working["Attribute_value"]

        if pd.api.types.is_bool_dtype(attribute_values):
            formatted_values = attribute_values.map(
                {
                    True: "TRUE",
                    False: "FALSE",
                }
            ).astype("string")
        else:
            normalized_values = (
                attribute_values.astype("string")
                .str.strip()
            )

            lowercase_values = normalized_values.str.lower()

            formatted_values = normalized_values.mask(
                lowercase_values.isin({"true", "t", "1"}),
                "TRUE",
            )

            formatted_values = formatted_values.mask(
                lowercase_values.isin({"false", "f", "0"}),
                "FALSE",
            )

        working["Attribute"] = (
            working["Attribute"].astype("string")
            + "--"
            + formatted_values.astype("string")
        )

    else:
        raise TraitFilteringError(
            "filter_data_discrete received an unsupported "
            f"Attribute_type: {attribute_type!r}"
        )

    working["NCBI_ID"] = _normalize_taxonomy_id_column(
        working["NCBI_ID"]
    )
    working["Parent_NCBI_ID"] = _normalize_taxonomy_id_column(
        working["Parent_NCBI_ID"]
    )

    usable_taxonomy = (
        working["NCBI_ID"].notna()
        | working["Parent_NCBI_ID"].notna()
    )

    usable_metadata = (
        working["Attribute_source"].notna()
        & working["Frequency"].notna()
    )

    filtered = working.loc[
        usable_taxonomy & usable_metadata
    ].copy()

    filtered["Score"] = frequency_to_score(
        filtered["Frequency"]
    )

    filtered = _ensure_output_columns(
        filtered,
        DISCRETE_OUTPUT_COLUMNS,
        function_name="filter_data_discrete",
    )

    filtered = filtered.drop_duplicates(
        keep="first"
    ).reset_index(drop=True)

    rows_dropped = original_row_count - len(filtered)

    print(
        f"{_format_count(rows_dropped)} rows were dropped."
    )

    return filtered


# ---------------------------------------------------------------------------
# R-compatible aliases
# ---------------------------------------------------------------------------

filterData = filter_data
filterDataNumeric = filter_data_numeric
filterDataDiscrete = filter_data_discrete

