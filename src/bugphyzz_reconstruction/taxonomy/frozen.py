"""Frozen taxonomy interface for the 2023 Bugphyzz resources.

This module provides an offline taxonomy provider backed by version-controlled
TSV files. It reproduces the name, rank, and parent lookups that the original
R workflow obtained from taxizedb and the internal ranksParents object.

Expected resource directory
---------------------------

bugphyzz_reconstruction/
└── resources/
    └── taxonomy_2023/
        ├── attributes.tsv
        ├── ranks_parents.tsv
        ├── taxonomy_lookup.tsv
        ├── LTP_all_08_2023.newick
        ├── LTP_all_08_2023.tip_data.tsv
        └── LTP_all_08_2023.node_data.tsv

Only the following files are loaded by this module:

- taxonomy_lookup.tsv
- ranks_parents.tsv
- attributes.tsv, when requested

Tree resources should be loaded by a separate LTP/tree interface.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd


TAXONOMY_LOOKUP_FILENAME = "taxonomy_lookup.tsv"
RANKS_PARENTS_FILENAME = "ranks_parents.tsv"
ATTRIBUTES_FILENAME = "attributes.tsv"

TAXONOMY_LOOKUP_COLUMNS: tuple[str, ...] = (
    "NCBI_ID",
    "Taxon_name",
    "Rank",
)

RANKS_PARENTS_COLUMNS: tuple[str, ...] = (
    "NCBI_ID",
    "Rank",
    "Parent_NCBI_ID",
    "Parent_name",
    "Parent_rank",
)

EXPECTED_PARENT_RANK: dict[str, str] = {
    "strain": "species",
    "species": "genus",
    "genus": "family",
}


class TaxonomyResourceError(ValueError):
    """Raised when a frozen taxonomy resource is malformed or inconsistent."""


def _normalize_taxids(
    taxids: Iterable[Any] | pd.Series,
) -> pd.Series:
    """Normalize an iterable of taxonomy IDs to a unique string Series.

    Missing values, blank strings, and the literal value ``"unknown"`` are
    retained as missing values only when they occur in a lookup input. Duplicate
    non-missing IDs are removed while preserving their first occurrence.

    Parameters
    ----------
    taxids
        Taxonomy IDs to normalize.

    Returns
    -------
    pandas.Series
        Unique, non-missing taxonomy IDs with pandas ``string`` dtype.
    """
    if isinstance(taxids, pd.Series):
        series = taxids.copy()
    else:
        series = pd.Series(list(taxids))

    series = series.astype("string").str.strip()

    invalid = (
        series.isna()
        | series.eq("")
        | series.str.lower().eq("unknown")
    )

    return (
        series.mask(invalid)
        .dropna()
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
        .rename("NCBI_ID")
    )


def _normalize_id_column(
    values: pd.Series,
) -> pd.Series:
    """Normalize an ID column while retaining its original index."""
    result = values.astype("string").str.strip()

    invalid = (
        result.isna()
        | result.eq("")
        | result.str.lower().eq("unknown")
    )

    return result.mask(invalid)


def _require_columns(
    table: pd.DataFrame,
    required: Sequence[str],
    *,
    table_name: str,
) -> None:
    """Validate that a table contains all required columns."""
    missing = sorted(set(required).difference(table.columns))

    if missing:
        raise TaxonomyResourceError(
            f"{table_name} is missing required columns: "
            + ", ".join(missing)
        )


def _validate_unique_mapping(
    table: pd.DataFrame,
    *,
    key: str,
    value_columns: Sequence[str],
    table_name: str,
) -> None:
    """Ensure each key maps to only one combination of output values."""
    duplicated = table.loc[
        table[key].duplicated(keep=False),
        [key, *value_columns],
    ]

    if duplicated.empty:
        return

    conflicts = (
        duplicated.groupby(key, dropna=False)[list(value_columns)]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )

    conflicting_ids = conflicts.index[conflicts].astype(str).tolist()

    if conflicting_ids:
        preview = ", ".join(conflicting_ids[:10])

        if len(conflicting_ids) > 10:
            preview += ", ..."

        raise TaxonomyResourceError(
            f"{table_name} contains conflicting mappings for {key}: "
            f"{preview}"
        )


def _read_taxonomy_lookup(path: Path) -> pd.DataFrame:
    """Read and validate taxonomy_lookup.tsv."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen taxonomy lookup file was not found: {path}"
        )

    table = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=True,
        na_values=["NA", ""],
    )

    _require_columns(
        table,
        TAXONOMY_LOOKUP_COLUMNS,
        table_name=path.name,
    )

    table = table.loc[:, list(TAXONOMY_LOOKUP_COLUMNS)].copy()

    table["NCBI_ID"] = _normalize_id_column(table["NCBI_ID"])
    table["Taxon_name"] = table["Taxon_name"].astype("string").str.strip()
    table["Rank"] = table["Rank"].astype("string").str.strip().str.lower()

    if table["NCBI_ID"].isna().any():
        raise TaxonomyResourceError(
            f"{path.name} contains missing or invalid NCBI_ID values."
        )

    _validate_unique_mapping(
        table,
        key="NCBI_ID",
        value_columns=("Taxon_name", "Rank"),
        table_name=path.name,
    )

    # Exact duplicate rows are harmless but unnecessary.
    table = (
        table.drop_duplicates(
            subset=["NCBI_ID", "Taxon_name", "Rank"],
            keep="first",
        )
        .drop_duplicates(subset="NCBI_ID", keep="first")
        .reset_index(drop=True)
    )

    return table


def _read_ranks_parents(path: Path) -> pd.DataFrame:
    """Read and validate ranks_parents.tsv."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen ranks/parents file was not found: {path}"
        )

    table = pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=True,
        na_values=["NA", ""],
    )

    _require_columns(
        table,
        RANKS_PARENTS_COLUMNS,
        table_name=path.name,
    )

    table = table.loc[:, list(RANKS_PARENTS_COLUMNS)].copy()

    table["NCBI_ID"] = _normalize_id_column(table["NCBI_ID"])
    table["Parent_NCBI_ID"] = _normalize_id_column(
        table["Parent_NCBI_ID"]
    )
    table["Rank"] = table["Rank"].astype("string").str.strip().str.lower()
    table["Parent_rank"] = (
        table["Parent_rank"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    table["Parent_name"] = (
        table["Parent_name"]
        .astype("string")
        .str.strip()
    )

    if table["NCBI_ID"].isna().any():
        raise TaxonomyResourceError(
            f"{path.name} contains missing or invalid NCBI_ID values."
        )

    _validate_unique_mapping(
        table,
        key="NCBI_ID",
        value_columns=(
            "Rank",
            "Parent_NCBI_ID",
            "Parent_name",
            "Parent_rank",
        ),
        table_name=path.name,
    )

    table = (
        table.drop_duplicates(
            subset=list(RANKS_PARENTS_COLUMNS),
            keep="first",
        )
        .drop_duplicates(subset="NCBI_ID", keep="first")
        .reset_index(drop=True)
    )

    # Validate the parent-selection rules used by the original R resource.
    expected_parent_rank = table["Rank"].map(EXPECTED_PARENT_RANK)

    invalid_parent_rank = (
        expected_parent_rank.notna()
        & table["Parent_rank"].ne(expected_parent_rank)
    )

    if invalid_parent_rank.any():
        examples = table.loc[
            invalid_parent_rank,
            ["NCBI_ID", "Rank", "Parent_rank"],
        ].head(10)

        raise TaxonomyResourceError(
            "ranks_parents.tsv contains parent ranks that do not match "
            "the Bugphyzz parent-selection rules. Examples:\n"
            + examples.to_string(index=False)
        )

    return table


def _read_attributes(path: Path) -> pd.DataFrame:
    """Read attributes.tsv without imposing a package-specific schema."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen attributes file was not found: {path}"
        )

    return pd.read_csv(
        path,
        sep="\t",
        dtype="string",
        keep_default_na=True,
        na_values=["NA", ""],
    )


def _default_resource_directory() -> Path:
    """Return the installed taxonomy_2023 resource directory.

    This assumes that ``resources`` is a Python package directory containing
    an ``__init__.py`` file. If project resources are not packaged this way,
    pass ``resource_dir`` explicitly to ``FrozenTaxonomyProvider.from_tsv()``.
    """
    package_root = resources.files("bugphyzz_reconstruction")
    resource = package_root.joinpath("resources", "taxonomy_2023")

    # The normal installed-wheel case exposes a pathlib-compatible object.
    # Converting to Path also works directly in a source checkout.
    return Path(str(resource))


@dataclass(frozen=True, slots=True)
class FrozenTaxonomyProvider:
    """Offline taxonomy provider backed by frozen 2023 TSV resources.

    Parameters
    ----------
    taxonomy_lookup
        One row per NCBI ID with ``Taxon_name`` and ``Rank``.
    ranks_parents
        One row per NCBI ID with the selected parent ID, name, and rank.
    attributes
        Optional Bugphyzz attribute vocabulary.
    resource_dir
        Directory from which the frozen resources were loaded.
    snapshot
        Human-readable taxonomy snapshot identifier.
    """

    taxonomy_lookup: pd.DataFrame
    ranks_parents: pd.DataFrame
    attributes: pd.DataFrame | None = None
    resource_dir: Path | None = None
    snapshot: str = "taxonomy_2023"

    @classmethod
    def from_tsv(
        cls,
        resource_dir: str | Path | None = None,
        *,
        load_attributes: bool = False,
    ) -> FrozenTaxonomyProvider:
        """Load the frozen taxonomy provider from TSV files.

        Parameters
        ----------
        resource_dir
            Directory containing ``taxonomy_lookup.tsv`` and
            ``ranks_parents.tsv``. When omitted, the function looks in
            ``bugphyzz_reconstruction/resources/taxonomy_2023``.
        load_attributes
            Also load ``attributes.tsv`` when ``True``.

        Returns
        -------
        FrozenTaxonomyProvider
            Validated offline taxonomy provider.
        """
        directory = (
            Path(resource_dir)
            if resource_dir is not None
            else _default_resource_directory()
        )

        if not directory.is_dir():
            raise FileNotFoundError(
                f"Taxonomy resource directory was not found: {directory}"
            )

        taxonomy_lookup = _read_taxonomy_lookup(
            directory / TAXONOMY_LOOKUP_FILENAME
        )

        ranks_parents = _read_ranks_parents(
            directory / RANKS_PARENTS_FILENAME
        )

        attributes = (
            _read_attributes(directory / ATTRIBUTES_FILENAME)
            if load_attributes
            else None
        )

        return cls(
            taxonomy_lookup=taxonomy_lookup,
            ranks_parents=ranks_parents,
            attributes=attributes,
            resource_dir=directory,
        )

    def lookup_many(
        self,
        taxids: Iterable[Any] | pd.Series,
        *,
        include_unresolved: bool = True,
    ) -> pd.DataFrame:
        """Look up scientific names and ranks for multiple NCBI IDs.

        The lookup is performed as one vectorized DataFrame merge.

        Parameters
        ----------
        taxids
            NCBI taxonomy IDs. Duplicate IDs are removed while preserving
            first-occurrence order.
        include_unresolved
            When ``True``, retain requested IDs that are not found in the
            frozen lookup, with missing ``Taxon_name`` and ``Rank`` values.
            When ``False``, return only resolved IDs.

        Returns
        -------
        pandas.DataFrame
            Columns:

            - ``NCBI_ID``
            - ``Taxon_name``
            - ``Rank``
            - ``taxonomy_status``

            ``taxonomy_status`` is either ``"resolved"`` or ``"unresolved"``.
        """
        requested = _normalize_taxids(taxids).to_frame()

        result = requested.merge(
            self.taxonomy_lookup,
            on="NCBI_ID",
            how="left",
            sort=False,
            validate="one_to_one",
        )

        result["taxonomy_status"] = pd.Series(
            result["Taxon_name"].notna().map(
                {
                    True: "resolved",
                    False: "unresolved",
                }
            ),
            index=result.index,
            dtype="string",
        )

        if not include_unresolved:
            result = result.loc[
                result["taxonomy_status"].eq("resolved")
            ]

        return result.reset_index(drop=True)

    def parents_many(
        self,
        taxids: Iterable[Any] | pd.Series,
        *,
        include_unresolved: bool = True,
    ) -> pd.DataFrame:
        """Look up Bugphyzz parent assignments for multiple NCBI IDs.

        The parent is the selected ancestor used by the original R workflow:

        - strain → species
        - species → genus
        - genus → family

        Parameters
        ----------
        taxids
            NCBI taxonomy IDs. Duplicate IDs are removed while preserving
            first-occurrence order.
        include_unresolved
            When ``True``, retain requested IDs not found in
            ``ranks_parents.tsv``. When ``False``, return only IDs with a
            resolved parent assignment.

        Returns
        -------
        pandas.DataFrame
            Columns:

            - ``NCBI_ID``
            - ``Rank``
            - ``Parent_NCBI_ID``
            - ``Parent_name``
            - ``Parent_rank``
            - ``parent_status``
        """
        requested = _normalize_taxids(taxids).to_frame()

        result = requested.merge(
            self.ranks_parents,
            on="NCBI_ID",
            how="left",
            sort=False,
            validate="one_to_one",
        )

        result["parent_status"] = pd.Series(
            result["Parent_NCBI_ID"].notna().map(
                {
                    True: "resolved",
                    False: "unresolved",
                }
            ),
            index=result.index,
            dtype="string",
        )

        if not include_unresolved:
            result = result.loc[
                result["parent_status"].eq("resolved")
            ]

        return result.reset_index(drop=True)

    def unresolved_taxids(
        self,
        taxids: Iterable[Any] | pd.Series,
    ) -> pd.Series:
        """Return requested IDs absent from the frozen taxonomy lookup."""
        result = self.lookup_many(
            taxids,
            include_unresolved=True,
        )

        return (
            result.loc[
                result["taxonomy_status"].eq("unresolved"),
                "NCBI_ID",
            ]
            .reset_index(drop=True)
        )

    def validate_cross_resource_consistency(self) -> None:
        """Check rank agreement between the two frozen taxonomy tables.

        Raises
        ------
        TaxonomyResourceError
            If an NCBI ID has different ranks in ``taxonomy_lookup.tsv`` and
            ``ranks_parents.tsv``.
        """
        comparison = self.taxonomy_lookup[
            ["NCBI_ID", "Rank"]
        ].merge(
            self.ranks_parents[["NCBI_ID", "Rank"]],
            on="NCBI_ID",
            how="inner",
            suffixes=("_lookup", "_parents"),
            validate="one_to_one",
        )

        inconsistent = comparison.loc[
            comparison["Rank_lookup"].ne(comparison["Rank_parents"])
        ]

        if not inconsistent.empty:
            raise TaxonomyResourceError(
                "Rank disagreement between taxonomy_lookup.tsv and "
                "ranks_parents.tsv. Examples:\n"
                + inconsistent.head(10).to_string(index=False)
            )

    def summary(self) -> dict[str, int | str]:
        """Return a compact summary of the loaded frozen resources."""
        return {
            "snapshot": self.snapshot,
            "taxonomy_records": len(self.taxonomy_lookup),
            "parent_records": len(self.ranks_parents),
            "attribute_records": (
                len(self.attributes)
                if self.attributes is not None
                else 0
            ),
        }

