"""Standalone signature calculation for reconstruction release GMT files.

Consumes one finalized attribute table at a time. This implementation preserves
Bugphyzz release naming (including the legacy ``bugphyz:`` discrete prefix),
filtering, numeric threshold boundaries, and taxon ordering. It does not import
bugphyzz or download data.

Defaults include all evidence categories, exclude ``rarely`` frequencies, and
require ten unique taxa per signature. ``mixed`` includes superkingdom through
strain except kingdom, matching the existing release rules. Numeric signatures
use this package's resources/thresholds.tsv unless resource_dir is supplied;
that argument denotes the directory containing thresholds.tsv.

Keep release parity tests when changing these rules independently of the
user-facing signature API. The function also accepts the existing API's explicit
numeric bounds and filtering overrides for compatibility with release callers.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

__all__ = ["make_signatures"]

VALID_TAX_ID_TYPES = {"NCBI_ID", "Taxon_name"}


VALID_TAX_LEVELS = {
    "mixed",
    "superkingdom",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "strain",
}


MIXED_TAX_LEVELS = [
    "superkingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "strain",
]


VALID_EVIDENCE = {"exp", "igc", "tas", "nas", "tax", "asr"}


VALID_FREQUENCY = {"always", "usually", "sometimes", "rarely", "unknown"}


DISCRETE_ATTRIBUTE_TYPES = {"multistate-intersection", "binary", "multistate-union"}


NUMERIC_ATTRIBUTE_TYPES = {"range", "numeric"}


def make_signatures(
    dat: pd.DataFrame,
    tax_id_type: str = "NCBI_ID",
    tax_level: str | Iterable[str] = "mixed",
    evidence: Iterable[str] = ("exp", "igc", "tas", "nas", "tax", "asr"),
    frequency: Iterable[str] = ("always", "usually", "sometimes", "unknown"),
    min_size: int = 10,
    min_value: float | None = None,
    max_value: float | None = None,
    resource_dir: str | Path | None = None,
) -> dict[str, list[str]] | None:
    """Create release signatures from one finalized attribute table."""
    _validate_choice(tax_id_type, VALID_TAX_ID_TYPES, "tax_id_type")
    tax_levels = _as_list(tax_level)
    evidence_values = _as_list(evidence)
    frequency_values = _as_list(frequency)

    _validate_choices(tax_levels, VALID_TAX_LEVELS, "tax_level")
    _validate_choices(evidence_values, VALID_EVIDENCE, "evidence")
    _validate_choices(frequency_values, VALID_FREQUENCY, "frequency")

    attr_type = _single_unique(dat["Attribute_type"], "Attribute_type")

    if "mixed" in tax_levels:
        tax_levels = MIXED_TAX_LEVELS

    dat = dat.loc[
        dat["Rank"].isin(tax_levels)
        & dat["Evidence"].isin(evidence_values)
        & dat["Frequency"].isin(frequency_values)
    ].copy()

    if dat.empty:
        warnings.warn(
            "Not enough data for creating signatures. Try different filtering options",
            stacklevel=2,
        )
        return None

    if attr_type in DISCRETE_ATTRIBUTE_TYPES:
        signatures = _make_signatures_discrete(dat=dat, tax_id_type=tax_id_type)
    elif attr_type in NUMERIC_ATTRIBUTE_TYPES:
        signatures = _make_signatures_numeric(
            dat=dat,
            tax_id_type=tax_id_type,
            min_value=min_value,
            max_value=max_value,
            resource_dir=resource_dir,
        )
    else:
        raise ValueError(f"Unsupported Attribute_type: {attr_type!r}")

    output = {name: values for name, values in signatures.items() if len(values) >= min_size}
    if not output:
        warnings.warn(
            "Not enough data for creating signatures. Try different filtering options",
            stacklevel=2,
        )
    return output


def _make_signatures_discrete(dat: pd.DataFrame, tax_id_type: str = "NCBI_ID") -> dict[str, list[str]]:
    dat = dat.copy()
    # Mirrors the R script exactly, including the original "bugphyz" spelling.
    dat["Attribute"] = "bugphyz:" + dat["Attribute"].astype(str) + "|" + dat["Attribute_value"].astype(str)
    return _split_unique(dat, group_col="Attribute", value_col=tax_id_type)


def _make_signatures_numeric(
    dat: pd.DataFrame,
    tax_id_type: str = "NCBI_ID",
    min_value: float | None = None,
    max_value: float | None = None,
    resource_dir: str | Path | None = None,
) -> dict[str, list[str]]:
    dat = dat.copy()
    dat["Attribute_value"] = pd.to_numeric(dat["Attribute_value"], errors="coerce")

    if min_value is not None or max_value is not None:
        if min_value is None:
            min_value = float(dat["Attribute_value"].min())
            print(f"Minimum unspecified. Using {min_value}.")
        if max_value is None:
            max_value = float(dat["Attribute_value"].max())
            print(f"Maximum unspecified. Using {max_value}.")

        dat = dat.loc[
            (dat["Attribute_value"] >= min_value) & (dat["Attribute_value"] <= max_value)
        ].copy()
        dat["Attribute"] = (
            "bugphyzz:" + dat["Attribute"].astype(str) + f"| >= {min_value} & <= {max_value}"
        )
    else:
        thr = _thresholds(resource_dir=resource_dir)
        attr_group = _single_unique(dat["Attribute"], "Attribute")
        thr = thr.loc[thr["Attribute_group"] == attr_group].copy()

        dat["tmp_col"] = pd.NA
        for _, row in thr.iterrows():
            attr_name = row["Attribute"]
            lower = row["lower"]
            upper = row["upper"]

            if pd.isna(lower):
                lower = dat["Attribute_value"].min() - 0.01
            if pd.isna(upper):
                upper = dat["Attribute_value"].max()

            pos = (dat["Attribute_value"] > lower) & (dat["Attribute_value"] <= upper)
            dat.loc[pos, "tmp_col"] = attr_name
            dat.loc[pos, "Attribute"] = (
                "bugphyzz:"
                + dat.loc[pos, "Attribute"].astype(str)
                + f"|{attr_name}| > {round(float(lower), 2)} & <= {upper}"
            )

    return _split_unique(dat, group_col="Attribute", value_col=tax_id_type)


def _thresholds(resource_dir: str | Path | None = None) -> pd.DataFrame:
    path = _resource_path("thresholds.tsv", resource_dir=resource_dir)
    dat = pd.read_csv(path, sep="\t")

    def _range(row: pd.Series) -> str:
        lower = row.get("lower")
        upper = row.get("upper")
        if pd.isna(lower):
            return f"<={upper}"
        if pd.isna(upper):
            return f">={lower}"
        return f"{lower}-{upper}"

    dat["range"] = dat.apply(_range, axis=1)
    dat["unit"] = dat["unit"].fillna("") if "unit" in dat.columns else ""
    dat["Attribute_range"] = dat["range"].astype(str) + dat["unit"].astype(str)

    first_cols = ["Attribute_group", "Attribute", "Attribute_range"]
    remaining = [col for col in dat.columns if col not in first_cols]
    return dat[first_cols + remaining]


def _resource_path(filename: str, resource_dir: str | Path | None = None) -> Path:
    if resource_dir is None:
        base = Path(__file__).resolve().parents[1] / "resources"
    else:
        base = Path(resource_dir)

    path = base / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {filename!r}. Expected it at {path}. "
            "Pass resource_dir=... or place package TSV resources in resources/."
        )
    return path


def _single_unique(series: pd.Series, column_name: str) -> Any:
    values = series.dropna().unique()
    if len(values) != 1:
        raise ValueError(f"Expected exactly one unique value in {column_name}, found {values!r}.")
    return values[0]


def _as_list(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _validate_choice(value: str, valid_values: set[str], arg_name: str) -> None:
    if value not in valid_values:
        raise ValueError(f"Invalid {arg_name}: {value!r}. Valid options: {sorted(valid_values)}")


def _validate_choices(values: Iterable[str], valid_values: set[str], arg_name: str) -> None:
    bad = [value for value in values if value not in valid_values]
    if bad:
        raise ValueError(f"Invalid {arg_name}: {bad!r}. Valid options: {sorted(valid_values)}")


def _split_unique(dat: pd.DataFrame, group_col: str, value_col: str) -> dict[str, list[str]]:
    if value_col not in dat.columns:
        raise KeyError(f"Column {value_col!r} is not present in the DataFrame.")

    output: dict[str, list[str]] = {}
    for name, group in dat.groupby(group_col, dropna=False):
        values = group[value_col].dropna().astype(str).drop_duplicates().tolist()
        output[str(name)] = values
    return output
