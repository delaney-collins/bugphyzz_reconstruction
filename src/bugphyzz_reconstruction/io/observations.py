"""Load user-supplied trait observations from delimited text files.

The functions in this module provide a source-independent boundary for the
reconstruction pipeline.  They deliberately do not import the Bugphyzz Google
Sheets or BacDive adapters: a user file only needs to follow the canonical raw
observation schema consumed by :func:`preprocessing.filtering.filter_data`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd


SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".csv", ".tsv", ".txt"})
SUPPORTED_ATTRIBUTE_TYPES: frozenset[str] = frozenset(
    {"binary", "multistate-intersection", "multistate-union", "range"}
)
TAXONOMY_ID_COLUMNS: tuple[str, ...] = ("NCBI_ID", "Parent_NCBI_ID")

_COMMON_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "Taxon_name",
        "Attribute_source",
        "Confidence_in_curation",
        "Frequency",
        "Evidence",
        "Attribute_type",
        "Attribute_group",
    }
)
_DISCRETE_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"Attribute", "Attribute_value"}
)
_RANGE_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"Attribute_value_min", "Attribute_value_max"}
)


class ObservationLoadError(ValueError):
    """Raised when an observation file cannot be loaded unambiguously."""


def _normalize_attribute_type(value: str) -> str:
    normalized = value.strip().lower()
    # Defensive support for the typo used by an early R dispatcher.
    if normalized == "mltistate-union":
        return "multistate-union"
    return normalized


def _taxonomy_id(value: object) -> object:
    """Return a nullable string taxid without a spreadsheet-style ``.0``."""
    if value is None or pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "nan", "none", "null", "unknown"}:
        return pd.NA

    # CSVs previously opened in spreadsheet software often contain integral
    # identifiers serialized as 12345.0.  Remove only an all-zero decimal
    # suffix; a genuinely fractional identifier remains visible and fails
    # validation below.
    text = re.sub(r"\.0+$", "", text)
    if not text.isdecimal():
        raise ObservationLoadError(
            f"Invalid NCBI taxonomy identifier {value!r}; expected an integer."
        )
    return text


def _separator(path: Path, sep: str | None) -> str:
    if sep is not None:
        if not sep:
            raise ObservationLoadError("sep must not be empty.")
        return sep
    if path.suffix.lower() == ".csv":
        return ","
    if path.suffix.lower() in {".tsv", ".txt"}:
        return "\t"
    raise ObservationLoadError(
        f"Cannot infer the delimiter for {path.name!r}. Use a .csv, .tsv, "
        ".txt file, or supply sep explicitly."
    )


def validate_trait_observations(
    table: pd.DataFrame,
    *,
    source: str | Path | None = None,
) -> pd.DataFrame:
    """Validate and normalize one raw trait-observation table.

    A validated table contains exactly one ``Attribute_type`` and one
    ``Attribute_group``.  The returned DataFrame is a copy; the caller's table
    is not modified.
    """
    context = f" in {str(source)!r}" if source is not None else ""
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame.")
    if table.empty:
        raise ObservationLoadError(f"Observation table{context} is empty.")

    missing = sorted(_COMMON_REQUIRED_COLUMNS.difference(table.columns))
    if missing:
        raise ObservationLoadError(
            f"Observation table{context} is missing required columns: "
            + ", ".join(missing)
        )
    if not any(column in table.columns for column in TAXONOMY_ID_COLUMNS):
        raise ObservationLoadError(
            f"Observation table{context} must contain NCBI_ID, "
            "Parent_NCBI_ID, or both."
        )

    result = table.copy()
    for column in TAXONOMY_ID_COLUMNS:
        if column not in result:
            result[column] = pd.Series(pd.NA, index=result.index, dtype="string")
        else:
            result[column] = result[column].map(_taxonomy_id).astype("string")

    attribute_types = (
        result["Attribute_type"]
        .astype("string")
        .dropna()
        .str.strip()
        .map(_normalize_attribute_type)
        .drop_duplicates()
        .tolist()
    )
    if len(attribute_types) != 1:
        raise ObservationLoadError(
            f"Observation table{context} must contain exactly one non-missing "
            f"Attribute_type; found {attribute_types}."
        )
    attribute_type = attribute_types[0]
    if attribute_type not in SUPPORTED_ATTRIBUTE_TYPES:
        raise ObservationLoadError(
            f"Unsupported Attribute_type {attribute_type!r}{context}. "
            "Supported values are: " + ", ".join(sorted(SUPPORTED_ATTRIBUTE_TYPES))
        )
    result["Attribute_type"] = attribute_type

    attribute_groups = (
        result["Attribute_group"]
        .astype("string")
        .dropna()
        .str.strip()
        .loc[lambda values: values.ne("")]
        .drop_duplicates()
        .tolist()
    )
    if len(attribute_groups) != 1:
        raise ObservationLoadError(
            f"Observation table{context} must contain exactly one non-missing "
            f"Attribute_group; found {attribute_groups}."
        )
    result["Attribute_group"] = attribute_groups[0]

    type_columns = (
        _RANGE_REQUIRED_COLUMNS
        if attribute_type == "range"
        else _DISCRETE_REQUIRED_COLUMNS
    )
    missing = sorted(type_columns.difference(result.columns))
    if missing:
        raise ObservationLoadError(
            f"{attribute_type!r} observation table{context} is missing required "
            "columns: " + ", ".join(missing)
        )

    if not (result["NCBI_ID"].notna() | result["Parent_NCBI_ID"].notna()).any():
        raise ObservationLoadError(
            f"Observation table{context} has no usable NCBI taxonomy identifiers."
        )

    return result


def read_trait_observations(
    path: str | Path,
    *,
    sep: str | None = None,
    encoding: str = "utf-8-sig",
    validate: bool = True,
    **read_csv_kwargs: object,
) -> pd.DataFrame:
    """Read one raw trait table from CSV, TSV, or tab-delimited TXT.

    Taxonomy identifiers are parsed as strings at the file boundary, avoiding
    pandas' conversion of identifiers to floating-point values.  Additional
    keyword arguments are forwarded to :func:`pandas.read_csv`.
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"Observation file does not exist: {file_path}")

    conflicting = {"filepath_or_buffer", "sep", "encoding", "converters"}.intersection(
        read_csv_kwargs
    )
    if conflicting:
        raise TypeError(
            "Pass " + ", ".join(sorted(conflicting))
            + " through the named loader arguments, not read_csv_kwargs."
        )

    converters = {column: _taxonomy_id for column in TAXONOMY_ID_COLUMNS}
    try:
        table = pd.read_csv(
            file_path,
            sep=_separator(file_path, sep),
            encoding=encoding,
            converters=converters,
            **read_csv_kwargs,
        )
    except (pd.errors.ParserError, UnicodeError) as error:
        raise ObservationLoadError(
            f"Could not parse observation file {file_path}: {error}"
        ) from error

    # A converter is only applied to columns present in the input. Normalize
    # their dtype explicitly so missing IDs use pandas' nullable string value.
    for column in TAXONOMY_ID_COLUMNS:
        if column in table:
            table[column] = table[column].astype("string")

    if validate:
        return validate_trait_observations(table, source=file_path)
    return table


def read_trait_collection(
    source: str | Path | Iterable[str | Path] | Mapping[str, str | Path],
    *,
    sep: str | None = None,
    encoding: str = "utf-8-sig",
    validate: bool = True,
    **read_csv_kwargs: object,
) -> dict[str, pd.DataFrame]:
    """Load multiple raw trait files as ``{trait_name: DataFrame}``.

    ``source`` may be a directory, an iterable of paths, or an explicit mapping
    from trait names to paths.  Directory loading includes supported files in
    filename order.  For non-mapping inputs, each filename stem is the trait
    name; duplicate stems are rejected rather than silently overwritten.
    """
    if isinstance(source, Mapping):
        named_paths = [(str(name), Path(path)) for name, path in source.items()]
    elif isinstance(source, (str, Path)):
        source_path = Path(source).expanduser()
        if source_path.is_dir():
            paths = sorted(
                path
                for path in source_path.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            )
            if not paths:
                raise ObservationLoadError(
                    f"No CSV, TSV, or TXT observation files found in {source_path}."
                )
            named_paths = [(path.stem, path) for path in paths]
        else:
            named_paths = [(source_path.stem, source_path)]
    else:
        paths = [Path(path).expanduser() for path in source]
        if not paths:
            raise ObservationLoadError("No observation files were supplied.")
        named_paths = [(path.stem, path) for path in paths]

    names = [name for name, _ in named_paths]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ObservationLoadError(
            "Trait names must be unique; duplicate filename stems: "
            + ", ".join(duplicates)
        )
    if any(not name.strip() for name in names):
        raise ObservationLoadError("Trait names must not be empty.")

    return {
        name: read_trait_observations(
            path,
            sep=sep,
            encoding=encoding,
            validate=validate,
            **read_csv_kwargs,
        )
        for name, path in named_paths
    }


__all__ = [
    "ObservationLoadError",
    "read_trait_collection",
    "read_trait_observations",
    "validate_trait_observations",
]
