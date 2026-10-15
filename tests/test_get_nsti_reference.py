"""Parity tests for Python ``get_nsti`` against frozen R ``getNsti`` output."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bugphyzz.trees.ltp import LTPResources, get_nsti, ltp


PACKAGE_DIR = Path(__file__).resolve().parents[1]
RESOURCE_DIR = PACKAGE_DIR / "resources"
ASR_REFERENCE_DIR = Path("C:/Users/delan/OneDrive/Desktop/bugphyzz_package/src/bugphyzz/asr_reference_2023")
TREE_FR = Path("C:/Users/delan/OneDrive/Desktop/bugphyzz_package/src/bugphyzz/asr_reference_2023")
TREE_REFERENCE_DIR = TREE_FR / "_tree_reference"

# The older pipeline checkpoint was printed with six decimal places.
PIPELINE_ATOL = 5e-7
# References from generate_get_nsti_references.R retain high precision.
GENERATED_RTOL = 1e-12
GENERATED_ATOL = 1e-14


@pytest.fixture(scope="module")
def ltp_resources() -> LTPResources:
    """Parse the LTP inputs once for all NSTI parity cases."""
    return ltp(
        resource_dir=RESOURCE_DIR,
        remove_gn_nodes=True,
        node_names=True,
    )


def _read_one_column(path: Path, expected_column: str) -> list[str]:
    """Read either a headered one-column TSV or a headerless text file."""
    if not path.is_file():
        raise FileNotFoundError(f"Frozen R reference file not found: {path}")

    if path.suffix.lower() == ".txt":
        with path.open(encoding="utf-8") as handle:
            return [
                line.rstrip("\r\n")
                for line in handle
                if line.rstrip("\r\n")
            ]

    table = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=False,
    )
    if expected_column not in table.columns:
        raise AssertionError(
            f"{path} must contain column {expected_column!r}; "
            f"found {list(table.columns)!r}."
        )

    return table[expected_column].loc[lambda values: values.ne("")].tolist()


def _read_expected(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen R reference file not found: {path}")

    expected = pd.read_csv(
        path,
        sep="\t",
        dtype={"label": "string", "nsti": "float64"},
    )
    missing = {"label", "nsti"} - set(expected.columns)
    if missing:
        raise AssertionError(
            f"{path} is missing required columns: {sorted(missing)}."
        )

    return expected.loc[:, ["label", "nsti"]]


def _assert_nsti_parity(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    case_name: str,
    rtol: float,
    atol: float,
) -> None:
    """Compare schema, row order, labels, and floating-point distances."""
    assert list(observed.columns) == ["label", "nsti"]
    assert len(observed) == len(expected), (
        f"{case_name}: row-count mismatch: "
        f"Python={len(observed):,}, R={len(expected):,}."
    )

    observed_labels = observed["label"].astype("string").reset_index(drop=True)
    expected_labels = expected["label"].astype("string").reset_index(drop=True)

    unequal = observed_labels.ne(expected_labels)
    if unequal.any():
        indices = np.flatnonzero(unequal.to_numpy())[:10]
        details = [
            (
                int(index),
                observed_labels.iloc[index],
                expected_labels.iloc[index],
            )
            for index in indices
        ]
        pytest.fail(
            f"{case_name}: Python labels or row ordering differ from R. "
            f"First mismatches (row, Python, R): {details}"
        )

    observed_nsti = pd.to_numeric(
        observed["nsti"], errors="raise"
    ).to_numpy(dtype=float)
    expected_nsti = pd.to_numeric(
        expected["nsti"], errors="raise"
    ).to_numpy(dtype=float)

    np.testing.assert_allclose(
        observed_nsti,
        expected_nsti,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
        err_msg=f"{case_name}: Python NSTI distances differ from R.",
    )


def test_growth_temperature_get_nsti_matches_r(
    ltp_resources: LTPResources,
) -> None:
    """Reproduce the NSTI call captured by the continuous-ASR R pipeline."""
    case_dir = ASR_REFERENCE_DIR / "growth_temperature"

    annotated_tip_labels = _read_one_column(
        case_dir / "02_annotated_tip_values.tsv",
        "tip_label",
    )
    nodes = _read_one_column(
        TREE_REFERENCE_DIR / "nodes_with_taxid.txt",
        "node_label",
    )
    expected = _read_expected(case_dir / "11_nsti.tsv")

    observed = get_nsti(
        tree=ltp_resources.tree,
        annotated_tip_labels=annotated_tip_labels,
        nodes=nodes,
    )

    _assert_nsti_parity(
        observed,
        expected,
        case_name="growth_temperature",
        rtol=0.0,
        atol=PIPELINE_ATOL,
    )


def _generated_reference_dir() -> Path | None:
    """Find generated fixtures, with an environment override for local runs."""
    override = os.environ.get("BUGPHYZZ_GET_NSTI_REFERENCE_DIR")
    candidates = [
        Path(override).expanduser() if override else None,
        ASR_REFERENCE_DIR / "get_nsti_reference",
        PACKAGE_DIR / "get_nsti_reference",
    ]

    for candidate in candidates:
        if candidate is not None and (candidate / "manifest.tsv").is_file():
            return candidate.resolve()

    return None


def test_generated_get_nsti_cases_match_r(
    ltp_resources: LTPResources,
) -> None:
    """Run every case produced by generate_get_nsti_references.R, if present."""
    reference_dir = _generated_reference_dir()
    if reference_dir is None:
        pytest.skip(
            "Generated getNsti fixtures were not found. Place them under "
            "asr_reference_2023/get_nsti_reference, or set "
            "BUGPHYZZ_GET_NSTI_REFERENCE_DIR."
        )

    manifest = pd.read_csv(reference_dir / "manifest.tsv", sep="\t")
    if "case" not in manifest.columns or manifest.empty:
        raise AssertionError(
            f"{reference_dir / 'manifest.tsv'} must contain nonempty "
            "'case' entries."
        )

    for case_name in manifest["case"].astype(str):
        case_dir = reference_dir / case_name
        annotated_tip_labels = _read_one_column(
            case_dir / "annotated_tip_labels.tsv",
            "annotated_tip_label",
        )
        nodes = _read_one_column(case_dir / "nodes.tsv", "node_label")
        expected = _read_expected(case_dir / "expected_get_nsti.tsv")

        observed = get_nsti(
            tree=ltp_resources.tree,
            annotated_tip_labels=annotated_tip_labels,
            nodes=nodes,
        )

        _assert_nsti_parity(
            observed,
            expected,
            case_name=case_name,
            rtol=GENERATED_RTOL,
            atol=GENERATED_ATOL,
        )
