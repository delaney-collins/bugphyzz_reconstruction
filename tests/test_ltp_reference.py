"""Parity tests for the Python LTP resource loader.

These tests compare the Python output from bugphyzz.ltp.ltp() against
the frozen R reference files written to:

    asr_reference_2023/_tree_reference/

Expected reference files
------------------------
- node_data_loaded.tsv
- nodes_with_taxid.tsv
- tip_data_loaded.tsv
- tree_data_combined.tsv
- tree_edges.tsv
- tree_node_labels.tsv
- tree_summary.tsv
- tree_tip_labels.tsv

Adjust the paths in the USER SETTINGS section if needed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest
from Bio.Phylo.BaseTree import Clade, Tree

from bugphyzz.asr.ltp import LTPResources, ltp


# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------

PROJECT_DIR = Path("C:/Users/delan/OneDrive/Desktop/bugphyzz_package/src")

RESOURCE_DIR = PROJECT_DIR / "bugphyzz" / "resources"

REFERENCE_DIR = (
    PROJECT_DIR
    / "bugphyzz"
    / "asr_reference_2023"
    / "_tree_reference"
)

# Change this if the frozen script used the misspelled filename.
TREE_SUMMARY_CANDIDATES = (
    "tree_summary.tsv",
    "tree_sumary.tsv",
)

FLOAT_RTOL = 1e-10
FLOAT_ATOL = 1e-12


# ---------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------


@pytest.fixture(scope="session")
def resources() -> LTPResources:
    """Load the Python LTP resources once for all tests."""
    return ltp(
        resource_dir=RESOURCE_DIR,
        remove_gn_nodes=True,
        node_names=True,
    )


@pytest.fixture(scope="session")
def tree(resources: LTPResources) -> Tree:
    return resources.tree


# ---------------------------------------------------------------------
# GENERAL HELPERS
# ---------------------------------------------------------------------


def read_reference(filename: str) -> pd.DataFrame:
    """Read a frozen R TSV as nullable strings."""
    path = REFERENCE_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(f"Reference file not found: {path}")

    return pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=True,
        na_values=["NA", "NaN", "nan", ""],
    )

def read_reference_lines(filename: str) -> list[str]:
    """Read a headerless reference text file with one value per line."""
    path = REFERENCE_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(f"Reference file not found: {path}")

    values: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.rstrip("\r\n")

            if value == "":
                continue

            values.append(value)

    return values


def read_tree_summary() -> pd.DataFrame:
    """Read tree_summary.tsv, accepting the tree_sumary.tsv typo."""
    for filename in TREE_SUMMARY_CANDIDATES:
        path = REFERENCE_DIR / filename
        if path.is_file():
            return read_reference(filename)

    expected = ", ".join(TREE_SUMMARY_CANDIDATES)
    raise FileNotFoundError(
        f"No tree summary file found in {REFERENCE_DIR}. "
        f"Expected one of: {expected}"
    )


def drop_exported_row_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove common row-name columns produced by R exports."""
    result = df.copy()

    removable = {
        "",
        "x",
        "x.1",
        "row.names",
        "rownames",
        "...1",
        "unnamed: 0",
    }

    for column in list(result.columns):
        if str(column).strip().lower() in removable:
            values = pd.to_numeric(result[column], errors="coerce")

            # Only discard it automatically if it looks like a row counter.
            if values.notna().all():
                expected_zero = np.arange(len(result))
                expected_one = np.arange(1, len(result) + 1)

                if (
                    np.array_equal(values.to_numpy(), expected_zero)
                    or np.array_equal(values.to_numpy(), expected_one)
                ):
                    result = result.drop(columns=column)

    return result


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for R/Python comparisons."""
    result = df.copy()

    result.columns = [
        re.sub(r"\.+", "_", str(column).strip())
        .replace(" ", "_")
        .lower()
        for column in result.columns
    ]

    return result


def normalize_scalar(value: object) -> object:
    """Normalize a scalar without discarding meaningful text."""
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()

    if text in {"", "NA", "NaN", "nan", "None", "<NA>"}:
        return pd.NA

    # Normalize integer-like floating-point strings:
    # "123.0" -> "123"
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return text.split(".", maxsplit=1)[0]

    return text


def normalize_frame(
    df: pd.DataFrame,
    *,
    sort_columns: bool = True,
) -> pd.DataFrame:
    """Normalize a data frame for strict parity comparisons."""
    result = drop_exported_row_names(df)
    result = normalize_column_names(result)

    for column in result.columns:
        result[column] = result[column].map(normalize_scalar).astype("string")

    if sort_columns:
        result = result.reindex(sorted(result.columns), axis="columns")

    return result.reset_index(drop=True)


def assert_same_columns(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    context: str,
) -> None:
    """Raise a useful assertion when column sets differ."""
    observed_columns = set(observed.columns)
    expected_columns = set(expected.columns)

    missing = expected_columns - observed_columns
    extra = observed_columns - expected_columns

    assert not missing and not extra, (
        f"{context} column mismatch.\n"
        f"Missing from Python: {sorted(missing)}\n"
        f"Extra in Python: {sorted(extra)}"
    )


def assert_frames_equal_as_strings(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    context: str,
    check_row_order: bool = True,
    sort_by: Iterable[str] | None = None,
) -> None:
    """Compare normalized data frames with useful diagnostics."""
    observed_norm = normalize_frame(observed)
    expected_norm = normalize_frame(expected)

    assert_same_columns(
        observed_norm,
        expected_norm,
        context=context,
    )

    observed_norm = observed_norm[expected_norm.columns]

    if sort_by is not None:
        sort_columns = [
            column
            for column in sort_by
            if column in observed_norm.columns
        ]

        if sort_columns:
            observed_norm = (
                observed_norm
                .sort_values(sort_columns, na_position="last")
                .reset_index(drop=True)
            )
            expected_norm = (
                expected_norm
                .sort_values(sort_columns, na_position="last")
                .reset_index(drop=True)
            )

    elif not check_row_order:
        all_columns = list(expected_norm.columns)

        observed_norm = (
            observed_norm
            .sort_values(all_columns, na_position="last")
            .reset_index(drop=True)
        )
        expected_norm = (
            expected_norm
            .sort_values(all_columns, na_position="last")
            .reset_index(drop=True)
        )

    try:
        pd.testing.assert_frame_equal(
            observed_norm,
            expected_norm,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        mismatch = observed_norm.ne(expected_norm)
        mismatch &= ~(
            observed_norm.isna()
            & expected_norm.isna()
        )

        locations = np.argwhere(mismatch.to_numpy())

        details: list[str] = []

        for row_index, column_index in locations[:10]:
            column = observed_norm.columns[column_index]

            details.append(
                f"row={row_index}, column={column!r}, "
                f"Python={observed_norm.iloc[row_index, column_index]!r}, "
                f"R={expected_norm.iloc[row_index, column_index]!r}"
            )

        detail_text = "\n".join(details)

        raise AssertionError(
            f"{context} does not match the frozen R output.\n"
            f"First mismatches:\n{detail_text}"
        ) from exc


def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    *,
    required: bool = True,
) -> str | None:
    """Find the first matching normalized column name."""
    normalized = normalize_column_names(df)

    candidate_set = {
        candidate.lower()
        for candidate in candidates
    }

    for column in normalized.columns:
        if column in candidate_set:
            return column

    if required:
        raise AssertionError(
            "Could not find any expected column. "
            f"Candidates: {sorted(candidate_set)}; "
            f"available: {list(normalized.columns)}"
        )

    return None


# ---------------------------------------------------------------------
# TREE EXTRACTION HELPERS
# ---------------------------------------------------------------------


def get_tip_labels(tree: Tree) -> list[str]:
    """Return tip labels in Newick preorder."""
    labels: list[str] = []

    for tip in tree.get_terminals(order="preorder"):
        if tip.name is None:
            raise AssertionError("Python tree contains an unnamed tip.")

        labels.append(str(tip.name))

    return labels


def get_internal_nodes(tree: Tree) -> list[Clade]:
    """Return internal nodes in the order used by the Python loader."""
    return list(tree.get_nonterminals(order="preorder"))


def get_node_labels(tree: Tree) -> list[str | pd._libs.missing.NAType]:
    """Return internal-node labels in preorder."""
    labels: list[str | pd._libs.missing.NAType] = []

    for node in get_internal_nodes(tree):
        if node.name is None:
            labels.append(pd.NA)
        else:
            labels.append(str(node.name))

    return labels


def make_python_edge_table(tree: Tree) -> pd.DataFrame:
    """Construct an ape-style edge table from the Python tree.

    ape numbering convention:
    - terminal tips: 1 through Ntip
    - internal nodes: Ntip + 1 through Ntip + Nnode

    Internal nodes are assigned in root-first preorder, matching the
    node-naming logic in bugphyzz.ltp.
    """
    tips = list(tree.get_terminals(order="preorder"))
    nodes = get_internal_nodes(tree)

    tip_numbers = {
        clade: index
        for index, clade in enumerate(tips, start=1)
    }

    node_numbers = {
        clade: len(tips) + index
        for index, clade in enumerate(nodes, start=1)
    }

    def number_for(clade: Clade) -> int:
        if clade.is_terminal():
            return tip_numbers[clade]

        return node_numbers[clade]

    rows: list[dict[str, object]] = []

    def visit(parent: Clade) -> None:
        for child in parent.clades:
            rows.append(
                {
                    "parent": number_for(parent),
                    "child": number_for(child),
                    "branch_length": child.branch_length,
                    "parent_label": parent.name,
                    "child_label": child.name,
                }
            )
            visit(child)

    visit(tree.root)

    return pd.DataFrame(rows)


def make_python_tree_data(
    resources: LTPResources,
) -> pd.DataFrame:
    tip_data = resources.tip_data.reset_index(drop=True).copy()
    node_data = resources.node_data.reset_index(drop=True).copy()

    # Normalize column names so capitalization differences between
    # the tip and node tables do not create duplicate columns.
    tip_data.columns = [
        str(column).strip().lower()
        for column in tip_data.columns
    ]

    node_data.columns = [
        str(column).strip().lower()
        for column in node_data.columns
    ]

    if "tip_label" in tip_data.columns:
        tip_data = tip_data.rename(
            columns={"tip_label": "label"}
        )

    if "node_label" in node_data.columns:
        node_data = node_data.rename(
            columns={"node_label": "label"}
        )

    combined = pd.concat(
        [tip_data, node_data],
        ignore_index=True,
        sort=False,
    )

    wanted_columns = [
        "label",
        "ncbi_id",
        "rank",
        "taxid",
        "taxon_name",
    ]

    missing = [
        column
        for column in wanted_columns
        if column not in combined.columns
    ]

    if missing:
        raise ValueError(
            "Cannot construct tree_data_combined; missing columns: "
            f"{missing}\n"
            f"Available columns: {list(combined.columns)}"
        )

    return combined[wanted_columns]

# ---------------------------------------------------------------------
# FILE-EXISTENCE TEST
# ---------------------------------------------------------------------


def test_all_reference_files_exist() -> None:
    
    expected = [
        "node_data_loaded.tsv",
        "nodes_with_taxid.txt",
        "tip_data_loaded.tsv",
        "tree_data_combined.tsv",
        "tree_edges.tsv",
        "tree_node_labels.txt",
        "tree_tip_labels.txt",
    ]

    missing = [
        filename
        for filename in expected
        if not (REFERENCE_DIR / filename).is_file()
    ]

    assert not missing, (
        "Missing frozen R reference files:\n"
        + "\n".join(f"  - {filename}" for filename in missing)
    )

    assert any(
        (REFERENCE_DIR / filename).is_file()
        for filename in TREE_SUMMARY_CANDIDATES
    ), (
        "Missing tree summary. Expected tree_summary.tsv or "
        "tree_sumary.tsv."
    )


# ---------------------------------------------------------------------
# LABEL AND ORDER TESTS
# ---------------------------------------------------------------------


def test_tree_tip_labels_match_r_order(tree: Tree) -> None:
    TREE_TIP_LABELS_FILE = "C:/Users/delan/OneDrive/Desktop/bugphyzz_package/src/bugphyzz/asr_reference_2023/_tree_reference/tree_tip_labels.txt"
    
    expected_labels = read_reference_lines(
        TREE_TIP_LABELS_FILE
    )

    observed_labels = get_tip_labels(tree)

    assert len(observed_labels) == len(expected_labels), (
        "Tip-count mismatch.\n"
        f"Python: {len(observed_labels)}\n"
        f"R: {len(expected_labels)}"
    )

    mismatches = [
        (index, observed, expected)
        for index, (observed, expected) in enumerate(
            zip(observed_labels, expected_labels)
        )
        if observed != expected
    ]

    assert not mismatches, (
        "Python tip labels or tip order differ from R.\n"
        f"First mismatches: {mismatches[:10]}"
    )

def test_tree_node_labels_match_r_order(tree: Tree) -> None:
    TREE_NODE_LABELS_FILE = "C:/Users/delan/OneDrive/Desktop/bugphyzz_package/src/bugphyzz/asr_reference_2023/_tree_reference/tree_node_labels.txt"
    expected_labels = read_reference_lines(
        TREE_NODE_LABELS_FILE
    )

    observed_labels = [
        "" if node.name is None else str(node.name)
        for node in get_internal_nodes(tree)
    ]

    assert len(observed_labels) == len(expected_labels), (
        "Internal-node count mismatch.\n"
        f"Python: {len(observed_labels)}\n"
        f"R: {len(expected_labels)}"
    )

    mismatches = [
        (index, observed, expected)
        for index, (observed, expected) in enumerate(
            zip(observed_labels, expected_labels)
        )
        if observed != expected
    ]

    assert not mismatches, (
        "Python internal-node labels or order differ from R.\n"
        f"First mismatches: {mismatches[:10]}"
    )

# ---------------------------------------------------------------------
# METADATA TESTS
# ---------------------------------------------------------------------


def test_tip_data_matches_r(resources: LTPResources) -> None:
    expected = read_reference("tip_data_loaded.tsv")
    observed = resources.tip_data.reset_index(drop=True)

    assert_frames_equal_as_strings(
        observed,
        expected,
        context="tip_data",
        check_row_order=True,
    )


def test_node_data_matches_r(resources: LTPResources) -> None:
    expected = read_reference("node_data_loaded.tsv")
    observed = resources.node_data.reset_index(drop=True)

    assert_frames_equal_as_strings(
        observed,
        expected,
        context="node_data",
        check_row_order=True,
    )


def test_genus_tip_list_matches_tip_data(
    resources: LTPResources,
) -> None:
    expected = (
        resources.tip_data["tip_label"]
        .dropna()
        .astype(str)
    )

    expected = [
        label
        for label in expected
        if "g__" in label
    ]

    assert resources.gn_tips == expected


# ---------------------------------------------------------------------
# TREE SUMMARY TEST
# ---------------------------------------------------------------------


def test_tree_summary_matches_r(tree: Tree) -> None:
    expected = pd.read_csv(
        REFERENCE_DIR / "tree_summary.tsv",
        sep="\t",
        dtype="string",
    )

    expected.columns = [
        str(column).strip().lower()
        for column in expected.columns
    ]

    assert {"property", "value"}.issubset(expected.columns), (
        "tree_summary.tsv must contain 'property' and 'value' columns.\n"
        f"Columns present: {list(expected.columns)}"
    )

    # Convert the R summary table into a dictionary such as:
    #
    # {
    #     "ntip": "21571",
    #     "nnode": "21570",
    #     "nedge": "43140",
    # }
    expected_summary = {
        str(property_name).strip().lower(): value
        for property_name, value in zip(
            expected["property"],
            expected["value"],
        )
        if pd.notna(property_name)
    }

    python_summary = {
        "ntip": len(tree.get_terminals()),
        "nnode": len(tree.get_nonterminals()),
        "nedge": sum(
            len(clade.clades)
            for clade in tree.find_clades(order="preorder")
        ),
    }

    aliases = {
        "ntip": {
            "ntip",
            "n_tip",
            "n_tips",
            "tips",
            "number_of_tips",
        },
        "nnode": {
            "nnode",
            "n_node",
            "n_nodes",
            "nodes",
            "number_of_nodes",
        },
        "nedge": {
            "nedge",
            "n_edge",
            "n_edges",
            "edges",
            "number_of_edges",
        },
    }

    checked = 0

    for metric, candidate_names in aliases.items():
        matching_property = next(
            (
                candidate
                for candidate in candidate_names
                if candidate in expected_summary
            ),
            None,
        )

        if matching_property is None:
            continue

        expected_value = pd.to_numeric(
            pd.Series([expected_summary[matching_property]]),
            errors="coerce",
        ).iloc[0]

        assert pd.notna(expected_value), (
            f"Tree summary property {matching_property!r} "
            "does not contain a numeric value."
        )

        assert int(expected_value) == python_summary[metric], (
            f"{metric} differs.\n"
            f"Python: {python_summary[metric]}\n"
            f"R: {int(expected_value)}"
        )

        checked += 1

    assert checked > 0, (
        "No recognizable tree-count properties were found in "
        "tree_summary.tsv.\n"
        f"Properties present: {sorted(expected_summary)}"
    )

# ---------------------------------------------------------------------
# EDGE TESTS
# ---------------------------------------------------------------------


def test_tree_edges_match_r(tree: Tree) -> None:
    path = REFERENCE_DIR / "tree_edges.tsv"

    expected = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
    )

    expected = drop_exported_row_names(expected)

    assert expected.shape[1] >= 2, (
        "tree_edges.tsv must contain at least parent and child columns."
    )

    observed = make_python_edge_table(tree)

    expected_parent = pd.to_numeric(
        expected.iloc[:, 0],
        errors="raise",
    ).to_numpy()

    expected_child = pd.to_numeric(
        expected.iloc[:, 1],
        errors="raise",
    ).to_numpy()

    observed_parent = observed["parent"].to_numpy()
    observed_child = observed["child"].to_numpy()

    np.testing.assert_array_equal(
        observed_parent,
        expected_parent,
        err_msg="Parent node numbers differ from R.",
    )

    np.testing.assert_array_equal(
        observed_child,
        expected_child,
        err_msg="Child node numbers differ from R.",
    )

    if expected.shape[1] >= 3:
        expected_lengths = pd.to_numeric(
            expected.iloc[:, 2],
            errors="coerce",
        ).to_numpy(dtype=float)

        observed_lengths = observed[
            "branch_length"
        ].to_numpy(dtype=float)

        np.testing.assert_allclose(
            observed_lengths,
            expected_lengths,
            rtol=FLOAT_RTOL,
            atol=FLOAT_ATOL,
            equal_nan=True,
        )


# ---------------------------------------------------------------------
# COMBINED TREE-DATA TEST
# ---------------------------------------------------------------------


def test_tree_data_combined_matches_r(
    resources: LTPResources,
) -> None:
    expected = read_reference(
        "tree_data_combined.tsv"
    )

    observed = make_python_tree_data(resources)

    expected = expected.dropna(
        axis="columns",
        how="all",
    )

    observed = observed.dropna(
        axis="columns",
        how="all",
    )

    assert_frames_equal_as_strings(
        observed,
        expected,
        context="tree_data_combined",
        check_row_order=True,
    )


# ---------------------------------------------------------------------
# NODES-WITH-TAXID TEST
# ---------------------------------------------------------------------


# def test_nodes_with_taxid_match_r(
#     resources: LTPResources,
# ) -> None:
#     expected = normalize_column_names(
#         read_reference("nodes_with_taxid.tsv")
#     )

#     label_column = find_column(
#         expected,
#         candidates=(
#             "label",
#             "node_label",
#             "nodes_with_taxid",
#         ),
#     )

#     expected_labels = (
#         expected[label_column]
#         .dropna()
#         .map(normalize_scalar)
#         .astype("string")
#         .tolist()
#     )

#     combined = normalize_column_names(
#         make_python_tree_data(resources)
#     )

#     combined_label_column = find_column(
#         combined,
#         candidates=(
#             "label",
#             "node_label",
#             "tip_label",
#         ),
#         required=False,
#     )

#     if combined_label_column is None:
#         # If tip_data and node_data use separate label columns, combine
#         # the nonmissing values in the same row order.
#         possible_columns = [
#             column
#             for column in (
#                 "tip_label",
#                 "node_label",
#                 "label",
#             )
#             if column in combined.columns
#         ]

#         assert possible_columns, (
#             "Could not identify a label column in Python tree_data."
#         )

#         observed_series = combined[possible_columns].bfill(axis=1).iloc[:, 0]

#     else:
#         observed_series = combined[combined_label_column]

#     taxid_column = find_column(
#         combined,
#         candidates=(
#             "ncbi_id",
#             "ncbiid",
#             "taxid",
#             "tax_id",
#         ),
#     )

#     observed_labels = (
#         observed_series[
#             combined[taxid_column].notna()
#         ]
#         .dropna()
#         .map(normalize_scalar)
#         .astype("string")
#     )

#     # The original R code defined nodes_with_taxid from node-like labels.
#     # Generated unnamed-node labels begin with "n".
#     observed_labels = [
#         label
#         for label in observed_labels
#         if not str(label).startswith("n")
#     ]

#     assert observed_labels == expected_labels, (
#         "Python nodes_with_taxid differs from the frozen R reference."
#     )

def test_nodes_with_taxid_are_present_in_python_tree(
    tree: Tree,
) -> None:
    NODES_WITH_TAXID_FILE = "C:/Users/delan/OneDrive/Desktop/bugphyzz_package/src/bugphyzz/asr_reference_2023/_tree_reference/nodes_with_taxid.txt"
    expected_labels = read_reference_lines(
        NODES_WITH_TAXID_FILE
    )

    python_node_labels = {
        str(node.name)
        for node in get_internal_nodes(tree)
        if node.name is not None
    }

    missing = [
        label
        for label in expected_labels
        if label not in python_node_labels
    ]

    assert not missing, (
        f"{len(missing)} nodes_with_taxid labels are absent from "
        f"the Python tree. First missing labels: {missing[:10]}"
    )