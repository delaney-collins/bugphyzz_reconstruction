"""Tests for public ASR orchestration and resource export."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from bugphyzz.trees.asr import (
    propagate_discrete_trait,
    write_zenodo_exports,
)
from bugphyzz.trees.ltp import ltp


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def test_low_coverage_discrete_trait_skips_asr_and_retains_observation() -> None:
    resources = ltp(
        resource_dir=PACKAGE_DIR / "resources",
        remove_gn_nodes=True,
        node_names=True,
    )
    prepared = pd.read_csv(
        PACKAGE_DIR / "asr_reference_2023/motility/00_prepared_input.tsv",
        sep="\t",
        dtype={"NCBI_ID": "string", "taxid": "string"},
        na_values=["NA"],
    )
    one_taxon = prepared.loc[
        prepared["NCBI_ID"].eq(prepared["NCBI_ID"].iloc[0])
    ].copy()

    result = propagate_discrete_trait(one_taxon, resources)

    assert not result.asr_run
    assert result.coverage["coverage_percent"].iloc[0] == 0
    assert result.data["Evidence"].notna().all()
    assert len(result.data) == one_taxon["Evidence"].notna().sum()


def test_zenodo_writer_adds_r_compatible_header(tmp_path: Path) -> None:
    row = {
        "NCBI_ID": "100",
        "Taxon_name": "Example taxon",
        "Rank": "species",
        "Attribute": "motility",
        "Attribute_value": "TRUE",
        "Evidence": "tax",
        "Frequency": "always",
        "Score": 1.0,
        "Attribute_source": pd.NA,
        "Confidence_in_curation": pd.NA,
        "Attribute_type": "binary",
    }
    tables = {
        "multistate": pd.DataFrame([row]),
        "binary": pd.DataFrame([row]),
        "numeric": pd.DataFrame([row]),
    }
    paths = write_zenodo_exports(
        tables,
        tmp_path,
        build_date=date(2026, 7, 30),
    )

    first_line = paths["binary"].read_text(encoding="utf-8").splitlines()[0]
    assert first_line == (
        "# bugphyzz 2026-07-30, License: Creative Commons Attribution 4.0 "
        "International, URL: https://waldronlab.io/bugphyzz"
    )
