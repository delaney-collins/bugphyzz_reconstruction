"""Parity tests for bugphyzz.preprocessing.taxonomy_sets.

This script compares the Python taxonomy-set preparation functions against
frozen R outputs:

    01_filtered.tsv
        Input to taxonomy-set preparation.

    02_with_ids.tsv
        Expected output from getSetWithIDs().

    03_without_ids.tsv
        Expected output from getSetWithoutIDs().

    04_combined_before_completion.tsv
        Expected direct- and parent-derived rows before trait completion.

    05_prepared.tsv
        Expected output from the complete getDataReady() preparation stage.

Three representative attribute types are tested:

- binary
- multistate-intersection
- range

Run directly:

    python tests/parity/test_taxonomy_sets_parity.py

or with pytest:

    pytest tests/parity/test_taxonomy_sets_parity.py -v
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import traceback
from pandas.testing import assert_frame_equal

# Permit direct execution from a src-layout checkout without requiring an
# editable installation in the selected interpreter.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bugphyzz_reconstruction.preprocessing.taxonomy_sets import (
    get_set_with_ids,
    get_set_without_ids,
)
from bugphyzz_reconstruction.preprocessing.pipeline import prepare_trait_data
from bugphyzz_reconstruction.taxonomy.frozen import FrozenTaxonomyProvider


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = REPOSITORY_ROOT

RESOURCE_DIR = (
    PROJECT_ROOT
    / "src"
    / "bugphyzz_reconstruction"
    / "resources"
    / "taxonomy_2023"
)

FROZEN_OUTPUT_DIR = (
    PROJECT_ROOT.parent
    / "asr_python"
    / "frozen_r_outputs"
)


AttributeType = Literal[
    "binary",
    "multistate-intersection",
    "range",
]


@dataclass(frozen=True)
class TaxonomyParityCase:
    """Frozen files for one taxonomy-set parity case."""

    name: str
    attribute_type: AttributeType
    filtered_file: Path
    with_ids_file: Path
    without_ids_file: Path
    combined_file: Path
    prepared_file: Path


TEST_CASES: tuple[TaxonomyParityCase, ...] = (
    TaxonomyParityCase(
        name="binary",
        attribute_type="binary",
        filtered_file=(
            FROZEN_OUTPUT_DIR
            / "binary_example"
            / "01_filtered.tsv"
        ),
        with_ids_file=(
            FROZEN_OUTPUT_DIR
            / "binary_example"
            / "02_with_ids.tsv"
        ),
        without_ids_file=(
            FROZEN_OUTPUT_DIR
            / "binary_example"
            / "03_without_ids.tsv"
        ),
        combined_file=(
            FROZEN_OUTPUT_DIR
            / "binary_example"
            / "04_combined_before_completion.tsv"
        ),
        prepared_file=(
            FROZEN_OUTPUT_DIR
            / "binary_example"
            / "05_prepared.tsv"
        ),
    ),
    TaxonomyParityCase(
        name="multistate_intersection",
        attribute_type="multistate-intersection",
        filtered_file=(
            FROZEN_OUTPUT_DIR
            / "intersection_example"
            / "01_filtered.tsv"
        ),
        with_ids_file=(
            FROZEN_OUTPUT_DIR
            / "intersection_example"
            / "02_with_ids.tsv"
        ),
        without_ids_file=(
            FROZEN_OUTPUT_DIR
            / "intersection_example"
            / "03_without_ids.tsv"
        ),
        combined_file=(
            FROZEN_OUTPUT_DIR
            / "intersection_example"
            / "04_combined_before_completion.tsv"
        ),
        prepared_file=(
            FROZEN_OUTPUT_DIR
            / "intersection_example"
            / "05_prepared.tsv"
        ),
    ),
    TaxonomyParityCase(
        name="range",
        attribute_type="range",
        filtered_file=(
            FROZEN_OUTPUT_DIR
            / "range_example"
            / "01_filtered.tsv"
        ),
        with_ids_file=(
            FROZEN_OUTPUT_DIR
            / "range_example"
            / "02_with_ids.tsv"
        ),
        without_ids_file=(
            FROZEN_OUTPUT_DIR
            / "range_example"
            / "03_without_ids.tsv"
        ),
        combined_file=(
            FROZEN_OUTPUT_DIR
            / "range_example"
            / "04_combined_before_completion.tsv"
        ),
        prepared_file=(
            FROZEN_OUTPUT_DIR
            / "range_example"
            / "05_prepared.tsv"
        ),
    ),
)


# ---------------------------------------------------------------------------
# TSV loading
# ---------------------------------------------------------------------------

NA_VALUES: tuple[str, ...] = (
    "NA",
    "NaN",
    "nan",
    "<NA>",
    "NULL",
)


def read_parity_tsv(
    path: Path,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame | None:
    """Read a frozen R parity TSV while preserving taxonomy IDs as strings.

    Parameters
    ----------
    path
        TSV file to read.
    allow_missing
        When True, a missing file is interpreted as an R NULL result.

    Returns
    -------
    pandas.DataFrame or None
        Loaded table, or None when the file is absent and allow_missing=True.
    """
    if not path.exists():
        if allow_missing:
            return None

        raise FileNotFoundError(
            f"Parity file does not exist: {path}"
        )

    try:
        table = pd.read_csv(
            path,
            sep="\t",
            dtype="string",
            na_values=list(NA_VALUES),
            keep_default_na=True,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    # Remove possible row-name columns written by R.
    row_name_columns = [
        column
        for column in table.columns
        if str(column).startswith("Unnamed:")
        or str(column) in {"row.names", "X"}
    ]

    if row_name_columns:
        table = table.drop(
            columns=row_name_columns
        )

    return table


def drop_all_missing_columns(
    table: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Apply the all-missing-column cleanup used by prepare_trait_data()."""
    if table is None:
        return None
    return table.dropna(axis="columns", how="all")


def combine_optional_tables(
    *tables: pd.DataFrame | None,
) -> pd.DataFrame:
    """Combine direct and parent tables using bind-rows semantics."""
    usable = [
        table
        for table in tables
        if table is not None and not table.empty
    ]
    if not usable:
        return pd.DataFrame()
    return pd.concat(
        usable,
        axis=0,
        ignore_index=True,
        sort=False,
    )


# ---------------------------------------------------------------------------
# Comparison normalization
# ---------------------------------------------------------------------------

TAXID_COLUMNS: frozenset[str] = frozenset(
    {
        "NCBI_ID",
        "Parent_NCBI_ID",
        "taxid",
    }
)

NUMERIC_COLUMNS: frozenset[str] = frozenset(
    {
        "Attribute_value_min",
        "Attribute_value_max",
        "Attribute_value",
        "Score",
        "Total_score",
    }
)

STRING_COLUMNS: frozenset[str] = frozenset(
    {
        "Taxon_name",
        "Rank",
        "Attribute",
        "Attribute_source",
        "Confidence_in_curation",
        "Frequency",
        "Evidence",
        "Attribute_group",
        "Attribute_type",
    }
)


def normalize_taxid_series(
    series: pd.Series,
) -> pd.Series:
    """Normalize raw and rank-prefixed taxonomy identifiers."""
    normalized = (
        series.astype("string")
        .str.strip()
    )

    # Normalize numeric taxids that may have been written as 123.0.
    normalized = normalized.str.replace(
        r"^(\d+)\.0$",
        r"\1",
        regex=True,
    )

    # Also handle rank-prefixed IDs such as s__123.0.
    normalized = normalized.str.replace(
        r"^([a-z]__\d+)\.0$",
        r"\1",
        regex=True,
    )

    return normalized


def normalize_attribute_labels(
    series: pd.Series,
) -> pd.Series:
    """Normalize binary TRUE/FALSE suffix representations."""
    normalized = (
        series.astype("string")
        .str.strip()
    )

    normalized = normalized.str.replace(
        r"--(?:true|t|1)$",
        "--TRUE",
        regex=True,
        case=False,
    )

    normalized = normalized.str.replace(
        r"--(?:false|f|0)$",
        "--FALSE",
        regex=True,
        case=False,
    )

    return normalized


def normalize_for_comparison(
    table: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Normalize R/Python representation differences.

    This function does not:

    - sort rows;
    - sort columns;
    - remove duplicates;
    - drop all-missing columns;
    - aggregate records;
    - alter scores beyond numeric conversion.

    Substantive differences therefore remain visible.
    """
    if table is None:
        return None

    normalized = table.copy()

    normalized.columns = [
        str(column).strip()
        for column in normalized.columns
    ]

    for column in normalized.columns:
        if column in TAXID_COLUMNS:
            normalized[column] = normalize_taxid_series(
                normalized[column]
            )

        elif column in NUMERIC_COLUMNS:
            normalized[column] = pd.to_numeric(
                normalized[column],
                errors="coerce",
            )

        elif column == "Attribute":
            normalized[column] = normalize_attribute_labels(
                normalized[column]
            )

        elif column in STRING_COLUMNS:
            normalized[column] = (
                normalized[column]
                .astype("string")
                .str.strip()
            )

        elif pd.api.types.is_categorical_dtype(
            normalized[column].dtype
        ):
            normalized[column] = (
                normalized[column]
                .astype("string")
                .str.strip()
            )

    return normalized.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def values_equal(
    left: object,
    right: object,
    *,
    numeric: bool,
) -> bool:
    """Compare two normalized scalar cell values safely."""

    left_missing = pd.isna(left)
    right_missing = pd.isna(right)

    # Both missing values are equivalent for parity purposes.
    if left_missing and right_missing:
        return True

    # One missing and one non-missing value are different.
    if left_missing or right_missing:
        return False

    if numeric:
        try:
            return bool(
                np.isclose(
                    float(left),
                    float(right),
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )
            )
        except (TypeError, ValueError):
            return False

    comparison = left == right

    # Defensive fallback in case an extension scalar still returns pd.NA.
    if pd.isna(comparison):
        return False

    return bool(comparison)


def describe_difference(
    actual: pd.DataFrame | None,
    expected: pd.DataFrame | None,
    *,
    case_name: str,
    stage_name: str,
) -> str:
    """Build a readable report for a failed parity comparison."""
    lines: list[str] = [
        "",
        "=" * 78,
        f"TAXONOMY PARITY FAILURE: {case_name}",
        f"Stage: {stage_name}",
        "=" * 78,
    ]

    if actual is None or expected is None:
        lines.append(
            f"Python result: {'None' if actual is None else actual.shape}"
        )
        lines.append(
            f"R result:      {'None' if expected is None else expected.shape}"
        )
        return "\n".join(lines)

    lines.extend(
        [
            f"Python shape: {actual.shape}",
            f"R shape:      {expected.shape}",
        ]
    )

    actual_columns = list(actual.columns)
    expected_columns = list(expected.columns)

    if actual_columns != expected_columns:
        lines.extend(
            [
                "",
                "Column comparison:",
                f"Python: {actual_columns}",
                f"R:      {expected_columns}",
            ]
        )

        missing_columns = [
            column
            for column in expected_columns
            if column not in actual_columns
        ]

        extra_columns = [
            column
            for column in actual_columns
            if column not in expected_columns
        ]

        if missing_columns:
            lines.append(
                "Missing from Python: "
                + ", ".join(missing_columns)
            )

        if extra_columns:
            lines.append(
                "Extra in Python: "
                + ", ".join(extra_columns)
            )

        return "\n".join(lines)

    common_rows = min(
        len(actual),
        len(expected),
    )

    differing_rows: list[int] = []

    for row_index in range(common_rows):
        row_differs = False

        for column in actual_columns:
            if not values_equal(
                actual.iloc[row_index][column],
                expected.iloc[row_index][column],
                numeric=column in NUMERIC_COLUMNS,
            ):
                row_differs = True
                break

        if row_differs:
            differing_rows.append(row_index)

    if differing_rows:
        lines.extend(
            [
                "",
                "First differing row indices: "
                + ", ".join(
                    str(index)
                    for index in differing_rows[:10]
                ),
            ]
        )

        first_index = differing_rows[0]

        comparison_rows: list[dict[str, object]] = []

        for column in actual_columns:
            python_value = actual.iloc[first_index][column]
            r_value = expected.iloc[first_index][column]

            comparison_rows.append(
                {
                    "column": column,
                    "Python": python_value,
                    "R": r_value,
                    "equal": values_equal(
                        python_value,
                        r_value,
                        numeric=column in NUMERIC_COLUMNS,
                    ),
                }
            )

        comparison = pd.DataFrame(
            comparison_rows
        ).set_index("column")

        lines.extend(
            [
                "",
                f"First differing row: {first_index}",
                comparison.to_string(),
            ]
        )

    if len(actual) != len(expected):
        lines.extend(
            [
                "",
                "Row counts differ.",
                f"Python rows: {len(actual):,}",
                f"R rows:      {len(expected):,}",
            ]
        )

        if len(actual) > common_rows:
            lines.extend(
                [
                    "",
                    "First extra Python rows:",
                    actual.iloc[
                        common_rows:common_rows + 5
                    ].to_string(index=False),
                ]
            )

        if len(expected) > common_rows:
            lines.extend(
                [
                    "",
                    "First extra R rows:",
                    expected.iloc[
                        common_rows:common_rows + 5
                    ].to_string(index=False),
                ]
            )
    lines.append(
    describe_taxid_differences(
        actual,
        expected,
    )
)
    return "\n".join(lines)

def diagnose_taxonomy_assignments(
    filtered: pd.DataFrame,
    actual: pd.DataFrame | None,
    expected: pd.DataFrame | None,
    taxonomy: FrozenTaxonomyProvider,
    *,
    output_file: Path,
) -> pd.DataFrame:
    """Compare differing output taxids with the frozen taxonomy lookup."""
    if actual is None:
        actual_ids: set[str] = set()
    else:
        actual_ids = set(
            normalize_taxid_series(actual["taxid"])
            .dropna()
            .tolist()
        )

    if expected is None:
        expected_ids: set[str] = set()
    else:
        expected_ids = set(
            normalize_taxid_series(expected["taxid"])
            .dropna()
            .tolist()
        )

    differing_ids = sorted(
        actual_ids.symmetric_difference(expected_ids),
        key=lambda value: int(value)
        if str(value).isdigit()
        else str(value),
    )

    if not differing_ids:
        diagnostics = pd.DataFrame()
        diagnostics.to_csv(
            output_file,
            sep="\t",
            index=False,
        )
        return diagnostics

    lookup = taxonomy.lookup_many(
        differing_ids,
        include_unresolved=True,
    ).copy()

    lookup["present_in_python"] = (
        lookup["NCBI_ID"]
        .astype("string")
        .isin(actual_ids)
    )

    lookup["present_in_r"] = (
        lookup["NCBI_ID"]
        .astype("string")
        .isin(expected_ids)
    )

    filtered_ids = normalize_taxid_series(
        filtered["NCBI_ID"]
    )

    filtered_rows = filtered.copy()
    filtered_rows["_normalized_taxid"] = filtered_ids

    input_summary = (
        filtered_rows.loc[
            filtered_rows["_normalized_taxid"].isin(differing_ids)
        ]
        .groupby(
            "_normalized_taxid",
            dropna=False,
        )
        .agg(
            input_rows=("NCBI_ID", "size"),
            input_taxon_name=(
                "Taxon_name",
                lambda values: " | ".join(
                    values.astype("string")
                    .dropna()
                    .drop_duplicates()
                    .tolist()[:5]
                ),
            ),
            input_confidence=(
                "Confidence_in_curation",
                lambda values: " | ".join(
                    values.astype("string")
                    .dropna()
                    .drop_duplicates()
                    .tolist()
                ),
            ),
        )
        .reset_index()
        .rename(
            columns={
                "_normalized_taxid": "NCBI_ID",
            }
        )
    )

    diagnostics = lookup.merge(
        input_summary,
        on="NCBI_ID",
        how="left",
        sort=False,
    )

    diagnostics.to_csv(
        output_file,
        sep="\t",
        index=False,
    )

    return diagnostics


def sort_taxonomy_output(
    table: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Sort taxonomy-stage output for parity comparison."""
    if table is None:
        return None

    if table.empty:
        return table.reset_index(drop=True)

    sort_columns = [
        column
        for column in (
            "NCBI_ID",
            "Attribute",
            "Attribute_source",
        )
        if column in table.columns
    ]

    if not sort_columns:
        return table.reset_index(drop=True)

    return (
        table.sort_values(
            sort_columns,
            kind="stable",
            na_position="last",
        )
        .reset_index(drop=True)
    )

def describe_taxid_differences(
    actual: pd.DataFrame | None,
    expected: pd.DataFrame | None,
) -> str:
    """Report taxids present in only one taxonomy-stage output."""
    if actual is None or expected is None:
        return ""

    if "taxid" not in actual.columns or "taxid" not in expected.columns:
        return ""

    actual_ids = set(
        normalize_taxid_series(actual["taxid"])
        .dropna()
        .tolist()
    )

    expected_ids = set(
        normalize_taxid_series(expected["taxid"])
        .dropna()
        .tolist()
    )

    missing_from_python = sorted(
        expected_ids - actual_ids,
        key=lambda value: int(value)
        if str(value).isdigit()
        else str(value),
    )

    extra_in_python = sorted(
        actual_ids - expected_ids,
        key=lambda value: int(value)
        if str(value).isdigit()
        else str(value),
    )

    lines = [
        "",
        "Taxid-set comparison:",
        f"Unique Python taxids: {len(actual_ids):,}",
        f"Unique R taxids:      {len(expected_ids):,}",
        f"Missing from Python:   {len(missing_from_python):,}",
        f"Extra in Python:       {len(extra_in_python):,}",
    ]

    if missing_from_python:
        lines.append(
            "First taxids missing from Python: "
            + ", ".join(missing_from_python[:25])
        )

    if extra_in_python:
        lines.append(
            "First taxids extra in Python: "
            + ", ".join(extra_in_python[:25])
        )

    return "\n".join(lines)


def assert_tables_match(
    actual: pd.DataFrame | None,
    expected: pd.DataFrame | None,
    *,
    case_name: str,
    stage_name: str,
) -> None:
    """Assert parity between one Python result and one frozen R result."""
    actual_normalized = normalize_for_comparison(actual)
    expected_normalized = normalize_for_comparison(expected)

    actual_normalized = sort_taxonomy_output(actual_normalized)
    expected_normalized = sort_taxonomy_output(expected_normalized)

    if actual_normalized is None or expected_normalized is None:
        assert actual_normalized is None and expected_normalized is None, (
            describe_difference(
                actual_normalized,
                expected_normalized,
                case_name=case_name,
                stage_name=stage_name,
            )
        )
        return

    try:
        assert_frame_equal(
            actual_normalized,
            expected_normalized,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
            check_like=False,
            check_categorical=False,
        )
    except AssertionError as error:
        diagnostics = describe_difference(
            actual_normalized,
            expected_normalized,
            case_name=case_name,
            stage_name=stage_name,
        )

        raise AssertionError(
            f"{error}\n{diagnostics}"
        ) from error


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def taxonomy() -> FrozenTaxonomyProvider:
    """Load the frozen taxonomy provider once for the test session."""
    if not RESOURCE_DIR.exists():
        pytest.fail(
            f"RESOURCE_DIR does not exist: {RESOURCE_DIR}"
        )

    return FrozenTaxonomyProvider.from_tsv(
        resource_dir=RESOURCE_DIR,
        load_attributes=True,
    )


@pytest.fixture(
    params=TEST_CASES,
    ids=lambda case: case.name,
)
def prepared_case(
    request: pytest.FixtureRequest,
    taxonomy: FrozenTaxonomyProvider,
) -> dict[str, object]:
    """Run taxonomy, combination, and preparation stages for one fixture."""
    case: TaxonomyParityCase = request.param

    filtered = read_parity_tsv(
        case.filtered_file
    )

    assert filtered is not None

    actual_with_ids = get_set_with_ids(
        filtered,
        taxonomy,
    )

    # Use the Python direct-ID result here because this is how the translated
    # pipeline actually runs. It also tests that direct IDs are correctly
    # excluded from the parent-derived result.
    actual_without_ids = get_set_without_ids(
        filtered,
        taxonomy,
        set_with_ids=actual_with_ids,
    )

    actual_combined = combine_optional_tables(
        drop_all_missing_columns(actual_with_ids),
        drop_all_missing_columns(actual_without_ids),
    )
    actual_prepared = prepare_trait_data(
        filtered,
        taxonomy,
    )
    if isinstance(actual_prepared, dict):
        raise TypeError(
            "These parity fixtures must produce one prepared DataFrame, "
            "not multistate-union component tables."
        )

    expected_with_ids = read_parity_tsv(
        case.with_ids_file,
        allow_missing=True,
    )

    expected_without_ids = read_parity_tsv(
        case.without_ids_file,
        allow_missing=True,
    )
    expected_combined = read_parity_tsv(
        case.combined_file,
        allow_missing=True,
    )
    expected_prepared = read_parity_tsv(
        case.prepared_file,
        allow_missing=True,
    )

    return {
        "case": case,
        "filtered": filtered,
        "actual_with_ids": actual_with_ids,
        "actual_without_ids": actual_without_ids,
        "actual_combined": actual_combined,
        "actual_prepared": actual_prepared,
        "expected_with_ids": expected_with_ids,
        "expected_without_ids": expected_without_ids,
        "expected_combined": expected_combined,
        "expected_prepared": expected_prepared,
    }


# ---------------------------------------------------------------------------
# Main parity comparisons
# ---------------------------------------------------------------------------

def test_with_ids_matches_frozen_r_output(
    prepared_case: dict[str, object],
) -> None:
    """Compare get_set_with_ids() against 02_with_ids.tsv."""
    case = prepared_case["case"]

    assert isinstance(case, TaxonomyParityCase)

    assert_tables_match(
        prepared_case["actual_with_ids"],
        prepared_case["expected_with_ids"],
        case_name=case.name,
        stage_name="02_with_ids",
    )


def test_without_ids_matches_frozen_r_output(
    prepared_case: dict[str, object],
) -> None:
    """Compare get_set_without_ids() against 03_without_ids.tsv."""
    case = prepared_case["case"]

    assert isinstance(case, TaxonomyParityCase)

    assert_tables_match(
        prepared_case["actual_without_ids"],
        prepared_case["expected_without_ids"],
        case_name=case.name,
        stage_name="03_without_ids",
    )


def test_combined_before_completion_matches_frozen_r_output(
    prepared_case: dict[str, object],
) -> None:
    """Compare combined taxonomy rows against stage 04."""
    case = prepared_case["case"]

    assert isinstance(case, TaxonomyParityCase)

    assert_tables_match(
        prepared_case["actual_combined"],
        prepared_case["expected_combined"],
        case_name=case.name,
        stage_name="04_combined_before_completion",
    )


def test_prepared_matches_frozen_r_output(
    prepared_case: dict[str, object],
) -> None:
    """Compare complete trait preparation against stage 05."""
    case = prepared_case["case"]

    assert isinstance(case, TaxonomyParityCase)

    assert_tables_match(
        prepared_case["actual_prepared"],
        prepared_case["expected_prepared"],
        case_name=case.name,
        stage_name="05_prepared",
    )


# ---------------------------------------------------------------------------
# Focused behavioral checks
# ---------------------------------------------------------------------------

def test_direct_and_parent_sets_do_not_overlap(
    prepared_case: dict[str, object],
) -> None:
    """Parent-propagated taxids must not duplicate directly observed taxids."""
    with_ids = prepared_case["actual_with_ids"]
    without_ids = prepared_case["actual_without_ids"]

    if (
        with_ids is None
        or without_ids is None
        or with_ids.empty
        or without_ids.empty
    ):
        return

    direct_ids = set(
        with_ids["NCBI_ID"]
        .astype("string")
        .dropna()
    )

    parent_ids = set(
        without_ids["NCBI_ID"]
        .astype("string")
        .dropna()
    )

    overlap = sorted(
        direct_ids.intersection(parent_ids)
    )

    assert not overlap, (
        "Direct and parent-derived taxonomy sets overlap. "
        f"Overlapping IDs include: {overlap[:10]}"
    )


def test_only_supported_ranks_are_retained(
    prepared_case: dict[str, object],
) -> None:
    """Both stages should retain only genus, species, and strain records."""
    allowed_ranks = {
        "genus",
        "species",
        "strain",
    }

    for stage in (
        "actual_with_ids",
        "actual_without_ids",
    ):
        table = prepared_case[stage]

        if (
            table is None
            or table.empty
            or "Rank" not in table.columns
        ):
            continue

        observed_ranks = set(
            table["Rank"]
            .astype("string")
            .str.lower()
            .dropna()
        )

        unexpected = sorted(
            observed_ranks.difference(allowed_ranks)
        )

        assert not unexpected, (
            f"{stage} contains unsupported ranks: {unexpected}"
        )


def test_rank_prefixed_ids_match_rank(
    prepared_case: dict[str, object],
) -> None:
    """Rank-prefixed NCBI_ID values should agree with Rank."""
    expected_prefixes = {
        "genus": "g__",
        "species": "s__",
        "strain": "t__",
    }

    for stage in (
        "actual_with_ids",
        "actual_without_ids",
    ):
        table = prepared_case[stage]

        if table is None or table.empty:
            continue

        required = {"NCBI_ID", "Rank"}

        if not required.issubset(table.columns):
            continue

        for rank, prefix in expected_prefixes.items():
            rank_rows = table.loc[
                table["Rank"]
                .astype("string")
                .str.lower()
                .eq(rank)
            ]

            if rank_rows.empty:
                continue

            invalid = rank_rows.loc[
                ~rank_rows["NCBI_ID"]
                .astype("string")
                .str.startswith(prefix, na=False),
                ["NCBI_ID", "Rank"],
            ]

            assert invalid.empty, (
                f"{stage} contains IDs with prefixes inconsistent with "
                f"rank {rank!r}:\n{invalid.head(10).to_string(index=False)}"
            )


def test_raw_taxid_matches_prefixed_id(
    prepared_case: dict[str, object],
) -> None:
    """The taxid column should match the numeric part of NCBI_ID."""
    for stage in (
        "actual_with_ids",
        "actual_without_ids",
    ):
        table = prepared_case[stage]

        if (
            table is None
            or table.empty
            or not {"NCBI_ID", "taxid"}.issubset(table.columns)
        ):
            continue

        prefixed_taxids = (
            table["NCBI_ID"]
            .astype("string")
            .str.replace(
                r"^[a-z]__",
                "",
                regex=True,
            )
        )

        raw_taxids = normalize_taxid_series(
            table["taxid"]
        )

        mismatched = ~(
            prefixed_taxids.eq(raw_taxids)
            | (
                prefixed_taxids.isna()
                & raw_taxids.isna()
            )
        )

        assert not mismatched.any(), (
            f"{stage} contains mismatches between NCBI_ID and taxid:\n"
            + table.loc[
                mismatched,
                ["NCBI_ID", "taxid"],
            ]
            .head(10)
            .to_string(index=False)
        )


def test_parent_records_are_taxonomic_evidence(
    prepared_case: dict[str, object],
) -> None:
    """Records from get_set_without_ids() should use Evidence = tax."""
    table = prepared_case["actual_without_ids"]

    if (
        table is None
        or table.empty
        or "Evidence" not in table.columns
    ):
        return

    evidence_values = set(
        table["Evidence"]
        .astype("string")
        .dropna()
    )

    assert evidence_values == {"tax"}, (
        "Parent-derived records should contain only Evidence='tax', "
        f"but found {sorted(evidence_values)}."
    )


def test_discrete_scores_sum_to_one_per_taxid(
    prepared_case: dict[str, object],
) -> None:
    """Discrete scores should be normalized within each retained taxid."""
    case = prepared_case["case"]

    assert isinstance(case, TaxonomyParityCase)

    if case.attribute_type == "range":
        return

    for stage in (
        "actual_with_ids",
        "actual_without_ids",
    ):
        table = prepared_case[stage]

        if (
            table is None
            or table.empty
            or not {"NCBI_ID", "Score"}.issubset(table.columns)
        ):
            continue

        scores = table.copy()

        scores["Score"] = pd.to_numeric(
            scores["Score"],
            errors="coerce",
        )

        score_sums = scores.groupby(
            "NCBI_ID",
            dropna=False,
            sort=False,
        )["Score"].sum()

        invalid = score_sums.loc[
            ~np.isclose(
                score_sums.to_numpy(dtype=float),
                1.0,
                rtol=1e-12,
                atol=1e-12,
            )
        ]

        assert invalid.empty, (
            f"{stage} contains taxids whose scores do not sum to one:\n"
            + invalid.head(10).to_string()
        )


def test_range_has_one_record_per_taxid(
    prepared_case: dict[str, object],
) -> None:
    """Range processing should reduce each source type to one row per taxid."""
    case = prepared_case["case"]

    assert isinstance(case, TaxonomyParityCase)

    if case.attribute_type != "range":
        return

    for stage in (
        "actual_with_ids",
        "actual_without_ids",
    ):
        table = prepared_case[stage]

        if table is None or table.empty:
            continue

        duplicated = table.duplicated(
            subset=["NCBI_ID"],
            keep=False,
        )

        assert not duplicated.any(), (
            f"{stage} contains duplicate range records per NCBI_ID:\n"
            + table.loc[
                duplicated
            ].head(10).to_string(index=False)
        )


def test_output_contains_no_exact_duplicates(
    prepared_case: dict[str, object],
) -> None:
    """Both taxonomy stages should end without exact duplicate rows."""
    for stage in (
        "actual_with_ids",
        "actual_without_ids",
    ):
        table = prepared_case[stage]

        if table is None or table.empty:
            continue

        duplicates = table.duplicated(
            keep=False
        )

        assert not duplicates.any(), (
            f"{stage} contains exact duplicate rows:\n"
            + table.loc[
                duplicates
            ].head(10).to_string(index=False)
        )


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

def run_parity_tests_directly() -> None:
    """Run the primary stage comparisons without invoking pytest."""
    print("Loading frozen taxonomy resources...")

    taxonomy = FrozenTaxonomyProvider.from_tsv(
        resource_dir=RESOURCE_DIR,
        load_attributes=True,
    )

    failures: list[str] = []

    for case in TEST_CASES:
        print()
        print("-" * 78)
        print(
            f"Testing {case.name} "
            f"({case.attribute_type})"
        )
        print("-" * 78)

        try:
            filtered = read_parity_tsv(
                case.filtered_file
            )

            assert filtered is not None

            expected_with_ids = read_parity_tsv(
                case.with_ids_file,
                allow_missing=True,
            )

            expected_without_ids = read_parity_tsv(
                case.without_ids_file,
                allow_missing=True,
            )
            expected_combined = read_parity_tsv(
                case.combined_file,
                allow_missing=True,
            )
            expected_prepared = read_parity_tsv(
                case.prepared_file,
                allow_missing=True,
            )

            actual_with_ids = get_set_with_ids(
                filtered,
                taxonomy,
            )

            actual_without_ids = get_set_without_ids(
                filtered,
                taxonomy,
                set_with_ids=actual_with_ids,
            )
            actual_combined = combine_optional_tables(
                drop_all_missing_columns(actual_with_ids),
                drop_all_missing_columns(actual_without_ids),
            )
            actual_prepared = prepare_trait_data(
                filtered,
                taxonomy,
            )
            if isinstance(actual_prepared, dict):
                raise TypeError(
                    "Parity fixture unexpectedly produced union components."
                )

            assert_tables_match(
                actual_with_ids,
                expected_with_ids,
                case_name=case.name,
                stage_name="02_with_ids",
            )

            print(
                "[PASS] 02_with_ids: "
                + (
                    "None"
                    if actual_with_ids is None
                    else (
                        f"{len(actual_with_ids):,} rows × "
                        f"{len(actual_with_ids.columns):,} columns"
                    )
                )
            )

            assert_tables_match(
                actual_without_ids,
                expected_without_ids,
                case_name=case.name,
                stage_name="03_without_ids",
            )

            print(
                "[PASS] 03_without_ids: "
                + (
                    "None"
                    if actual_without_ids is None
                    else (
                        f"{len(actual_without_ids):,} rows × "
                        f"{len(actual_without_ids.columns):,} columns"
                    )
                )
            )

            assert_tables_match(
                actual_combined,
                expected_combined,
                case_name=case.name,
                stage_name="04_combined_before_completion",
            )
            print(
                "[PASS] 04_combined_before_completion: "
                f"{len(actual_combined):,} rows × "
                f"{len(actual_combined.columns):,} columns"
            )

            assert_tables_match(
                actual_prepared,
                expected_prepared,
                case_name=case.name,
                stage_name="05_prepared",
            )
            print(
                "[PASS] 05_prepared: "
                + (
                    "None"
                    if actual_prepared is None
                    else (
                        f"{len(actual_prepared):,} rows × "
                        f"{len(actual_prepared.columns):,} columns"
                    )
                )
            )
        
        except Exception as error:
            failures.append(case.name)

            print(f"[FAIL] {case.name}")
            traceback.print_exc()
    print()
    print("=" * 78)
    diagnostic_file = (
    PROJECT_ROOT
    / "tests"
    / "parity"
    / "diagnostics"
    / f"{case.name}_02_taxonomy_differences.tsv"
)


    if failures:
        print(
            "Taxonomy-set parity failed for: "
            + ", ".join(failures)
        )
        raise SystemExit(1)

    print(
        "All taxonomy-set parity tests passed."
    )


if __name__ == "__main__":
    run_parity_tests_directly()
