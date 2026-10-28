"""End-to-end driver connecting Bugphyzz preparation and ASR assembly.

The lower-level modules operate on one stage at a time.  This module connects
raw physiology tables to filtering, ``getDataReady``-equivalent preparation,
trait propagation, multistate-union reassembly, and the final Zenodo tables.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import pandas as pd

from bugphyzz_reconstruction.asr.ltp import LTPResources
from bugphyzz_reconstruction.asr.orchestrator import (
    BugphyzzBuildResult,
    build_bugphyzz_resources,
    propagate_trait,
)
from bugphyzz_reconstruction.io.observations import (
    read_trait_observations,
    validate_trait_observations,
)
from bugphyzz_reconstruction.preprocessing.filtering import filter_data
from bugphyzz_reconstruction.preprocessing.pipeline import prepare_trait_data
from bugphyzz_reconstruction.taxonomy.frozen import FrozenTaxonomyProvider


PreparedTrait = pd.DataFrame | dict[str, pd.DataFrame]


@dataclass(frozen=True)
class BugphyzzPipelineResult:
    """Complete build result plus inspectable preprocessing checkpoints."""

    build: BugphyzzBuildResult
    filtered_traits: dict[str, pd.DataFrame]
    prepared_traits: dict[str, PreparedTrait]
    skipped_traits: dict[str, str]


@dataclass(frozen=True)
class TraitReconstructionResult:
    """Inspectable result of reconstructing one raw trait table.

    ``prepared`` is a mapping only for multistate-union traits. Those traits
    are reassembled directly from their prepared observations and therefore
    always have ``asr_run=False`` and an empty coverage table.
    """

    raw: pd.DataFrame
    filtered: pd.DataFrame
    prepared: PreparedTrait
    propagated: pd.DataFrame
    coverage: pd.DataFrame
    asr_run: bool
    attribute_group: str
    attribute_type: str


def _single_value(table: pd.DataFrame, column: str, *, context: str) -> str:
    if column not in table.columns:
        raise ValueError(f"{context} is missing required column: {column}")
    values = table[column].dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(
            f"{context} must contain exactly one nonmissing {column}; "
            f"found {values.tolist()}."
        )
    return values[0]


def reassemble_multistate_union(
    component_tables: Mapping[str, pd.DataFrame],
    *,
    attribute_group: str,
) -> pd.DataFrame:
    """Convert reconstructed binary components back to one union trait.

    A multistate-union category is prepared as a binary ``category--TRUE``
    versus ``category--FALSE`` trait. The R export does not run ASR for these
    traits. The published union representation therefore retains observed TRUE
    rows and relabels them as ``Attribute_value = category``.
    """
    frames: list[pd.DataFrame] = []

    for component_name, component_table in component_tables.items():
        table = component_table.copy()
        required = {"Attribute"}
        missing = required - set(table.columns)
        if missing:
            raise ValueError(
                f"Union component {component_name!r} is missing columns: "
                f"{sorted(missing)}."
            )

        # Prepared union components encode their Boolean state in Attribute.
        # Attribute_value can legitimately be absent after all-missing columns
        # are dropped (notably for antimicrobial-resistance inputs).
        is_true = table["Attribute"].astype("string").str.endswith("--TRUE")
        true_rows = table.loc[is_true].copy()
        if true_rows.empty:
            continue

        category = (
            true_rows["Attribute"]
            .astype("string")
            .str.replace(r"--TRUE$", "", regex=True)
        )
        true_rows["Attribute_value"] = category.fillna(str(component_name))
        true_rows["Attribute"] = attribute_group
        true_rows["Attribute_group"] = attribute_group
        true_rows["Attribute_type"] = "multistate-union"
        frames.append(true_rows)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def reconstruct_trait(
    observations: pd.DataFrame | str | Path,
    taxonomy: FrozenTaxonomyProvider,
    resources: LTPResources,
    *,
    allow_custom_vocabulary: bool = False,
    min_annotated_tips: int | None = None,
) -> TraitReconstructionResult:
    """Filter, prepare, and reconstruct one raw trait table.

    Parameters
    ----------
    observations
        A raw observation DataFrame or a CSV, TSV, or tab-delimited TXT path.
        File inputs preserve taxonomy identifiers as strings at read time.
    taxonomy
        Frozen taxonomy provider, including the attribute vocabulary used by
        filtering.
    resources
        LTP tree, tip data, and node data used by the ASR branches.
    allow_custom_vocabulary
        Accept the discrete ``Attribute`` terms declared in this observation
        table even when they are absent from the packaged Bugphyzz vocabulary.
        The terms are added only to an in-memory copy of the taxonomy provider.
    min_annotated_tips
        If supplied, run ASR when at least this many distinct LTP tips are
        annotated, instead of applying the historical one-percent rule.

    Returns
    -------
    TraitReconstructionResult
        The validated raw, filtered, prepared, and propagated stages together
        with the coverage decision and whether ASR ran.

    Notes
    -----
    Multistate-union traits deliberately bypass ASR. Their prepared binary
    components are reassembled into the published union representation.
    """
    if isinstance(observations, pd.DataFrame):
        raw = validate_trait_observations(observations.copy())
    elif isinstance(observations, (str, Path)):
        raw = read_trait_observations(observations)
    else:
        raise TypeError(
            "observations must be a pandas DataFrame or a CSV, TSV, or TXT path."
        )

    if min_annotated_tips is not None and min_annotated_tips < 1:
        raise ValueError("min_annotated_tips must be at least 1.")

    filtering_taxonomy = taxonomy
    if allow_custom_vocabulary:
        if taxonomy.attributes is None:
            raise ValueError(
                "allow_custom_vocabulary requires a taxonomy provider loaded "
                "with attributes.tsv."
            )
        attribute_type = _single_value(
            raw,
            "Attribute_type",
            context="raw trait",
        )
        if attribute_type != "range":
            declared = (
                raw.loc[:, ["Attribute_group", "Attribute"]]
                .dropna()
                .rename(
                    columns={
                        "Attribute_group": "attribute_group",
                        "Attribute": "attribute",
                    }
                )
                .drop_duplicates()
            )
            attributes = pd.concat(
                [taxonomy.attributes, declared],
                ignore_index=True,
                sort=False,
            ).drop_duplicates(subset=["attribute_group", "attribute"])
            filtering_taxonomy = replace(taxonomy, attributes=attributes)

    filtered = filter_data(raw, taxonomy=filtering_taxonomy)
    if filtered is None or filtered.empty:
        raise ValueError("Trait filtering produced no usable rows.")

    attribute_group = _single_value(
        filtered,
        "Attribute_group",
        context="filtered trait",
    )
    attribute_type = _single_value(
        filtered,
        "Attribute_type",
        context="filtered trait",
    )

    prepared = prepare_trait_data(filtered, taxonomy)
    if prepared is None or (
        isinstance(prepared, (pd.DataFrame, dict)) and len(prepared) == 0
    ):
        raise ValueError("Trait preparation produced no usable rows.")

    if isinstance(prepared, dict):
        propagated = reassemble_multistate_union(
            prepared,
            attribute_group=attribute_group,
        )
        if propagated.empty:
            raise ValueError(
                "Multistate-union reassembly produced no usable rows."
            )
        coverage = pd.DataFrame()
        asr_run = False
    else:
        propagation = propagate_trait(
            prepared,
            resources,
            attribute_group=attribute_group,
            attribute_type=attribute_type,
            min_annotated_tips=min_annotated_tips,
        )
        propagated = propagation.data
        coverage = propagation.coverage
        asr_run = propagation.asr_run

    return TraitReconstructionResult(
        raw=raw,
        filtered=filtered,
        prepared=prepared,
        propagated=propagated,
        coverage=coverage,
        asr_run=asr_run,
        attribute_group=attribute_group,
        attribute_type=attribute_type,
    )


def prepare_and_build_bugphyzz(
    raw_traits: Mapping[str, pd.DataFrame],
    taxonomy: FrozenTaxonomyProvider,
    resources: LTPResources,
    *,
    output_dir: str | Path | None = None,
    passthrough_multistate: pd.DataFrame | None = None,
) -> BugphyzzPipelineResult:
    """Run raw trait tables through preparation, ASR, and final assembly.

    Parameters
    ----------
    raw_traits
        Mapping returned by :func:`bugphyzz.physiologies.physiologies`, or an
        equivalent frozen mapping of raw trait tables.
    taxonomy
        Frozen taxonomy provider, loaded with ``attributes.tsv`` for discrete
        filtering.
    resources
        LTP tree, tip data, and node data.
    output_dir
        If supplied, write the three Zenodo-compatible CSV resources there.
    passthrough_multistate
        Optional already-finalized habitat, AMR, disease, or other multistate
        rows to append to the reconstructed union output.

    Returns
    -------
    BugphyzzPipelineResult
        Final resource tables plus filtered/prepared checkpoints and explicit
        reasons for traits that produced no usable data.
    """
    filtered_traits: dict[str, pd.DataFrame] = {}
    prepared_traits: dict[str, PreparedTrait] = {}
    ordinary_prepared: dict[str, pd.DataFrame] = {}
    union_tables: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}

    for trait_name, raw_table in raw_traits.items():
        if raw_table is None or raw_table.empty:
            skipped[trait_name] = "raw trait table is empty"
            continue

        filtered = filter_data(raw_table, taxonomy=taxonomy)
        if filtered is None or filtered.empty:
            skipped[trait_name] = "filtering produced no usable rows"
            continue
        filtered_traits[trait_name] = filtered

        prepared = prepare_trait_data(filtered, taxonomy)
        if prepared is None or (
            isinstance(prepared, (pd.DataFrame, dict)) and len(prepared) == 0
        ):
            skipped[trait_name] = "preparation produced no usable rows"
            continue
        prepared_traits[trait_name] = prepared

        if isinstance(prepared, pd.DataFrame):
            ordinary_prepared[trait_name] = prepared
            continue

        union_group = _single_value(
            filtered,
            "Attribute_group",
            context=f"multistate-union trait {trait_name!r}",
        )
        union_table = reassemble_multistate_union(
            prepared,
            attribute_group=union_group,
        )
        if not union_table.empty:
            union_tables.append(union_table)

    passthrough_frames = list(union_tables)
    if passthrough_multistate is not None and not passthrough_multistate.empty:
        passthrough_frames.append(passthrough_multistate.copy())
    combined_passthrough = (
        pd.concat(passthrough_frames, ignore_index=True, sort=False)
        if passthrough_frames
        else None
    )

    build = build_bugphyzz_resources(
        ordinary_prepared,
        resources,
        output_dir=output_dir,
        passthrough_multistate=combined_passthrough,
    )
    return BugphyzzPipelineResult(
        build=build,
        filtered_traits=filtered_traits,
        prepared_traits=prepared_traits,
        skipped_traits=skipped,
    )
