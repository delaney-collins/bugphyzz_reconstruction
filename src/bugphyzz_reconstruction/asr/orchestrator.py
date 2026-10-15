"""Public trait propagation and Bugphyzz resource-build orchestration.

This module is the Python counterpart of the orchestration in
``export_bugphyzz.R``.  The numerical algorithms live in the branch-specific
modules; this file dispatches traits, applies the one-percent coverage rule,
and assembles the three tables published as Bugphyzz resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping

import pandas as pd

from bugphyzz_reconstruction.asr.asr_shared import (
    combine_observed_and_predictions,
    format_final_asr_output,
    make_tree_taxonomy_table,
    prepare_mapped_predictions,
)
from bugphyzz_reconstruction.asr.continuous import (
    add_continuous_prediction_evidence,
    attach_continuous_nsti,
    calculate_continuous_nsti,
    prepare_continuous_asr_input,
    reconstruct_continuous_states,
    select_mappable_unannotated_states,
)
from bugphyzz_reconstruction.asr.discrete_pipeline import (
    prepare_discrete_asr_input,
    run_discrete_asr_pipeline,
)
from bugphyzz_reconstruction.asr.ltp import LTPResources
from bugphyzz_reconstruction.asr.release_signatures import make_signatures


SUPPORTED_TYPES = {"range", "binary", "multistate-intersection"}
ZENODO_FILENAMES = {
    "multistate": "bugphyzz_multistate.csv",
    "binary": "bugphyzz_binary.csv",
    "numeric": "bugphyzz_numeric.csv",
}
GMT_RANKS = ("genus", "strain", "species", "mixed")
GMT_TAX_ID_TYPES = ("Taxon_name", "NCBI_ID")
ZENODO_LEADING_COLUMNS = [
    "NCBI_ID",
    "Taxon_name",
    "Rank",
    "Attribute",
    "Attribute_value",
    "Evidence",
    "Frequency",
    "Score",
    "Attribute_source",
    "Confidence_in_curation",
    "Attribute_type",
]


@dataclass(frozen=True)
class TraitPropagationResult:
    """One propagated trait plus its coverage decision."""

    data: pd.DataFrame
    coverage: pd.DataFrame
    asr_run: bool


@dataclass(frozen=True)
class BugphyzzBuildResult:
    """Outputs of a complete resource build."""

    traits: dict[str, TraitPropagationResult]
    multistate: pd.DataFrame
    binary: pd.DataFrame
    numeric: pd.DataFrame
    paths: dict[str, Path]


def _single_metadata_value(
    data: pd.DataFrame,
    column: str,
    supplied: str | None,
) -> str:
    if supplied is not None:
        return supplied
    if column not in data:
        raise ValueError(f"prepared_data is missing required column: {column}")
    values = data[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(
            f"{column} must contain exactly one nonmissing value; "
            f"found {values.tolist()}."
        )
    return values[0]


def _observed_rows(prepared_data: pd.DataFrame) -> pd.DataFrame:
    """Rows that survive the R export's final missing-Evidence filter."""
    if "Evidence" not in prepared_data:
        raise ValueError("prepared_data is missing required column: Evidence")
    return prepared_data.loc[prepared_data["Evidence"].notna()].copy()


def _format_observed_only(
    prepared_data: pd.DataFrame,
    *,
    attribute_group: str,
    attribute_type: str,
) -> pd.DataFrame:
    observed = _observed_rows(prepared_data)
    combined = combine_observed_and_predictions(
        observed,
        observed.iloc[0:0].copy(),
        attribute_group=attribute_group,
        attribute_type=attribute_type,
    )
    return format_final_asr_output(
        combined,
        attribute_group=attribute_group,
        attribute_type=attribute_type,
    )


def propagate_continuous_trait(
    prepared_data: pd.DataFrame,
    resources: LTPResources,
    *,
    attribute_group: str | None = None,
    weighted: bool = True,
) -> TraitPropagationResult:
    """Propagate one range trait, or retain observations below 1% coverage."""
    group = _single_metadata_value(
        prepared_data, "Attribute_group", attribute_group
    )
    prepared = prepare_continuous_asr_input(
        resources.tree,
        resources.tip_data,
        prepared_data,
        trait_name=group,
    )
    asr_run = bool(prepared.coverage["asr_run"].iloc[0])
    if not asr_run:
        return TraitPropagationResult(
            _format_observed_only(
                prepared_data,
                attribute_group=group,
                attribute_type="range",
            ),
            prepared.coverage,
            False,
        )

    taxonomy = make_tree_taxonomy_table(
        resources.tip_data, resources.node_data
    )
    reconstructed = reconstruct_continuous_states(
        resources.tree, prepared, weighted=weighted
    )
    mappable = select_mappable_unannotated_states(
        reconstructed.states,
        mappable_labels=taxonomy["label"],
        annotated_tip_labels=prepared.annotated_tip_labels,
    )
    nsti = calculate_continuous_nsti(
        resources.tree,
        annotated_tip_labels=prepared.annotated_tip_labels,
        node_labels=mappable["label"].astype(str).tolist(),
    )
    predictions = prepare_mapped_predictions(
        add_continuous_prediction_evidence(
            attach_continuous_nsti(mappable, nsti)
        ),
        taxonomy,
        prepared_data,
        deduplicate_by="Attribute_value",
    )
    combined = combine_observed_and_predictions(
        _observed_rows(prepared_data),
        predictions,
        attribute_group=group,
        attribute_type="range",
    )
    final = format_final_asr_output(
        combined,
        attribute_group=group,
        attribute_type="range",
    )
    return TraitPropagationResult(final, prepared.coverage, True)


def propagate_discrete_trait(
    prepared_data: pd.DataFrame,
    resources: LTPResources,
    *,
    attribute_group: str | None = None,
    attribute_type: str | None = None,
) -> TraitPropagationResult:
    """Propagate one binary/multistate trait with R-compatible coverage logic."""
    group = _single_metadata_value(
        prepared_data, "Attribute_group", attribute_group
    )
    kind = _single_metadata_value(
        prepared_data, "Attribute_type", attribute_type
    )
    if kind not in {"binary", "multistate-intersection"}:
        raise ValueError(f"Unsupported discrete Attribute_type: {kind!r}")

    prepared = prepare_discrete_asr_input(
        resources.tree,
        resources.tip_data,
        prepared_data,
        trait_name=group,
    )
    asr_run = bool(prepared.coverage["asr_run"].iloc[0])
    if not asr_run:
        return TraitPropagationResult(
            _format_observed_only(
                prepared_data,
                attribute_group=group,
                attribute_type=kind,
            ),
            prepared.coverage,
            False,
        )

    taxonomy = make_tree_taxonomy_table(
        resources.tip_data, resources.node_data
    )
    stages = run_discrete_asr_pipeline(
        resources.tree,
        resources.tip_data,
        taxonomy,
        prepared_data,
        trait_name=group,
    )
    combined = combine_observed_and_predictions(
        _observed_rows(prepared_data),
        stages.predictions_long,
        attribute_group=group,
        attribute_type=kind,
        predictions_first=True,
    )
    final = format_final_asr_output(
        combined,
        attribute_group=group,
        attribute_type=kind,
    )
    return TraitPropagationResult(final, prepared.coverage, True)


def propagate_trait(
    prepared_data: pd.DataFrame,
    resources: LTPResources,
    *,
    attribute_group: str | None = None,
    attribute_type: str | None = None,
) -> TraitPropagationResult:
    """Dispatch one prepared trait to the continuous or discrete branch."""
    kind = _single_metadata_value(
        prepared_data, "Attribute_type", attribute_type
    )
    if kind == "range":
        return propagate_continuous_trait(
            prepared_data,
            resources,
            attribute_group=attribute_group,
        )
    if kind in {"binary", "multistate-intersection"}:
        return propagate_discrete_trait(
            prepared_data,
            resources,
            attribute_group=attribute_group,
            attribute_type=kind,
        )
    raise ValueError(
        f"Attribute_type {kind!r} is not directly reconstructable. "
        "Split multistate-union data into its prepared binary traits first."
    )


def prepare_zenodo_table(data: pd.DataFrame, *, numeric: bool = False) -> pd.DataFrame:
    """Apply the final global cleanup performed by ``export_bugphyzz.R``."""
    result = data.copy()
    if "NCBI_ID" in result:
        result["NCBI_ID"] = (
            result["NCBI_ID"].astype("string").str.replace(
                r"^\w__", "", regex=True
            )
        )
    if "Evidence" in result:
        result = result.loc[result["Evidence"].notna()].copy()
    result = result.drop(
        columns=["taxid", "Attribute_group"], errors="ignore"
    )
    if numeric:
        result["Attribute_type"] = "numeric"

    missing = [column for column in ZENODO_LEADING_COLUMNS if column not in result]
    if missing:
        raise ValueError(f"Final resource table is missing columns: {missing}")
    trailing = [
        column for column in result.columns
        if column not in ZENODO_LEADING_COLUMNS
    ]
    return result.loc[:, ZENODO_LEADING_COLUMNS + trailing].reset_index(drop=True)


def write_zenodo_exports(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    build_date: date | None = None,
) -> dict[str, Path]:
    """Write the three comment-prefixed CSV resources consumed by Bugphyzz."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = (build_date or date.today()).isoformat()
    header = (
        f"# bugphyzz {stamp}, License: Creative Commons Attribution 4.0 "
        "International, URL: https://waldronlab.io/bugphyzz\n"
    )
    paths: dict[str, Path] = {}
    for category, filename in ZENODO_FILENAMES.items():
        if category not in tables:
            raise ValueError(f"Missing export table: {category}")
        path = destination / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(header)
            tables[category].to_csv(handle, index=False)
        paths[category] = path
    return paths


def write_gmt_exports(
    tables: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    build_date: date | None = None,
) -> dict[str, Path]:
    """Write the eight GMT resources produced by ``export_bugphyzz.R``.

    Signatures are built independently for each attribute so numeric and
    discrete attributes are dispatched using their own ``Attribute_type``.
    Each GMT file contains one signature per line, followed by the ``NA``
    description emitted by ``bugsigdbr::writeGMT`` and its tab-separated
    taxon identifiers.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    stamp = (build_date or date.today()).isoformat()
    header = (
        f"# bugphyzz {stamp}, License: Creative Commons Attribution 4.0 "
        "International, URL: https://waldronlab.io/bugphyzz"
    )

    attribute_tables: list[pd.DataFrame] = []
    for category in ZENODO_FILENAMES:
        if category not in tables:
            raise ValueError(f"Missing export table: {category}")
        table = tables[category]
        if table.empty:
            continue
        if "Attribute" not in table:
            raise ValueError(
                f"Export table {category!r} is missing column: Attribute"
            )
        attribute_tables.extend(
            group.copy()
            for _, group in table.groupby("Attribute", sort=True, dropna=True)
        )

    paths: dict[str, Path] = {}
    for rank in GMT_RANKS:
        for tax_id_type in GMT_TAX_ID_TYPES:
            filename = f"bugphyzz-{rank}-{tax_id_type}.gmt"
            path = destination / filename
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(f"{header}\n")
                for attribute_table in attribute_tables:
                    signatures = make_signatures(
                        attribute_table,
                        tax_id_type=tax_id_type,
                        tax_level=rank,
                    )
                    for name, taxa in (signatures or {}).items():
                        handle.write(
                            "\t".join([name, "NA", *(str(taxon) for taxon in taxa)])
                            + "\n"
                        )
            paths[f"{rank}--{tax_id_type}"] = path
    return paths


def build_bugphyzz_resources(
    prepared_traits: Mapping[str, pd.DataFrame],
    resources: LTPResources,
    *,
    output_dir: str | Path | None = None,
    passthrough_multistate: pd.DataFrame | None = None,
) -> BugphyzzBuildResult:
    """Run all reconstructable traits and assemble Zenodo resource files.

    ``passthrough_multistate`` is for already-finalized habitat, AMR, disease,
    or other tables that the R build appends without running this ASR loop.
    """
    propagated: dict[str, TraitPropagationResult] = {}
    for name, prepared in prepared_traits.items():
        propagated[name] = propagate_trait(prepared, resources)

    frames = [item.data for item in propagated.values()]
    if passthrough_multistate is not None:
        frames.append(passthrough_multistate.copy())
    all_data = (
        pd.concat(frames, ignore_index=True, sort=False)
        if frames
        else pd.DataFrame(
            columns=ZENODO_LEADING_COLUMNS + ["taxid", "Attribute_group"]
        )
    )

    kinds = all_data.get(
        "Attribute_type",
        pd.Series(index=all_data.index, dtype="string"),
    ).astype("string")
    multistate_raw = all_data.loc[
        kinds.isin(["multistate-intersection", "multistate-union"])
    ]
    binary_raw = all_data.loc[kinds.eq("binary")]
    numeric_raw = all_data.loc[kinds.eq("range")]

    tables = {
        "multistate": prepare_zenodo_table(multistate_raw),
        "binary": prepare_zenodo_table(binary_raw),
        "numeric": prepare_zenodo_table(numeric_raw, numeric=True),
    }
    paths: dict[str, Path] = {}
    if output_dir is not None:
        paths.update(write_zenodo_exports(tables, output_dir))
        paths.update(write_gmt_exports(tables, output_dir))
    return BugphyzzBuildResult(
        traits=propagated,
        multistate=tables["multistate"],
        binary=tables["binary"],
        numeric=tables["numeric"],
        paths=paths,
    )
