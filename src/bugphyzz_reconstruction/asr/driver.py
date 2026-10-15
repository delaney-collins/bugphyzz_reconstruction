"""End-to-end driver connecting Bugphyzz preparation and ASR assembly.

The lower-level modules operate on one stage at a time.  This module connects
raw physiology tables to filtering, ``getDataReady``-equivalent preparation,
trait propagation, multistate-union reassembly, and the final Zenodo tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from bugphyzz_reconstruction.asr.ltp import LTPResources
from bugphyzz_reconstruction.asr.orchestrator import (
    BugphyzzBuildResult,
    build_bugphyzz_resources,
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
