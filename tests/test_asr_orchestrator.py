"""End-to-end smoke test for public Bugphyzz build orchestration.

Numerical ASR parity is covered by the branch-specific parity suites. This
module instead uses tiny synthetic raw tables to verify that filtering,
preparation, low-coverage dispatch, union bypass, final table assembly, and all
CSV/GMT writers remain connected through the public driver.
"""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

import pandas as pd
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bugphyzz_reconstruction.asr.driver import (  # noqa: E402
    prepare_and_build_bugphyzz,
)
from bugphyzz_reconstruction.asr.ltp import (  # noqa: E402
    LTPResources,
    ltp,
)
from bugphyzz_reconstruction.taxonomy.frozen import (  # noqa: E402
    FrozenTaxonomyProvider,
)


RESOURCE_DIR = (
    REPOSITORY_ROOT
    / "src"
    / "bugphyzz_reconstruction"
    / "resources"
    / "taxonomy_2023"
)


def _common_raw_columns(
    *,
    attribute_group: str,
    attribute_type: str,
) -> dict[str, list[object]]:
    """Return metadata shared by one-row synthetic observation tables."""
    return {
        # This species is represented by an LTP tip, so the low-coverage path
        # contains one annotated tip rather than an all-hidden input.
        "NCBI_ID": ["1001240"],
        "Taxon_name": ["Cryobacterium roopkundense"],
        "Parent_NCBI_ID": [pd.NA],
        "Attribute_source": ["synthetic smoke fixture"],
        "Confidence_in_curation": ["high"],
        "Frequency": ["always"],
        "Evidence": ["exp"],
        "Attribute_type": [attribute_type],
        "Attribute_group": [attribute_group],
    }


@pytest.fixture(scope="module")
def taxonomy() -> FrozenTaxonomyProvider:
    """Load the packaged taxonomy and vocabulary once."""
    return FrozenTaxonomyProvider.from_tsv(
        resource_dir=RESOURCE_DIR,
        load_attributes=True,
    )


@pytest.fixture(scope="module")
def resources() -> LTPResources:
    """Load the packaged LTP resources once."""
    return ltp(
        resource_dir=RESOURCE_DIR,
        remove_gn_nodes=True,
        node_names=True,
    )


@pytest.fixture
def raw_smoke_traits() -> dict[str, pd.DataFrame]:
    """Create one valid observation for every supported orchestration path."""
    binary = _common_raw_columns(
        attribute_group="motility",
        attribute_type="binary",
    )
    binary.update(
        {"Attribute": ["motility"], "Attribute_value": [True]}
    )

    intersection = _common_raw_columns(
        attribute_group="shape",
        attribute_type="multistate-intersection",
    )
    intersection.update(
        {"Attribute": ["bent filament"], "Attribute_value": [True]}
    )

    numeric = _common_raw_columns(
        attribute_group="growth temperature",
        attribute_type="range",
    )
    numeric.update(
        {
            "Attribute_value_min": [30.0],
            "Attribute_value_max": [30.0],
        }
    )

    union = _common_raw_columns(
        attribute_group="habitat",
        attribute_type="multistate-union",
    )
    union.update(
        {"Attribute": ["aquatic biome"], "Attribute_value": [True]}
    )

    return {
        "motility": pd.DataFrame(binary),
        "shape": pd.DataFrame(intersection),
        "growth temperature": pd.DataFrame(numeric),
        "habitat": pd.DataFrame(union),
    }


def test_prepare_and_build_bugphyzz_end_to_end_smoke(
    taxonomy: FrozenTaxonomyProvider,
    resources: LTPResources,
    raw_smoke_traits: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """Build every resource category and file from synthetic raw inputs."""
    # One-row fixtures are intentionally too small to form every GMT
    # signature. The smoke assertion is that all eight GMT files are written.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Not enough data for creating signatures.*",
            category=UserWarning,
        )
        result = prepare_and_build_bugphyzz(
            raw_smoke_traits,
            taxonomy,
            resources,
            output_dir=tmp_path,
        )

    expected_traits = set(raw_smoke_traits)
    assert result.skipped_traits == {}
    assert set(result.filtered_traits) == expected_traits
    assert set(result.prepared_traits) == expected_traits

    # Union traits bypass ASR and are reassembled as passthrough multistate
    # rows. The remaining three traits exercise both propagation dispatchers.
    assert set(result.build.traits) == {
        "motility",
        "shape",
        "growth temperature",
    }
    for name, propagation in result.build.traits.items():
        assert not propagation.asr_run, f"{name} unexpectedly ran ASR"
        assert propagation.coverage["coverage_percent"].iloc[0] == 0
        assert propagation.data["Evidence"].notna().all()

    assert not result.build.binary.empty
    assert not result.build.multistate.empty
    assert not result.build.numeric.empty
    assert result.build.binary["Attribute_type"].eq("binary").all()
    assert result.build.multistate["Attribute_type"].isin(
        ["multistate-intersection", "multistate-union"]
    ).all()
    assert result.build.numeric["Attribute_type"].eq("numeric").all()

    habitat_rows = result.build.multistate.loc[
        result.build.multistate["Attribute"].eq("habitat")
    ]
    assert not habitat_rows.empty
    assert habitat_rows["Attribute_value"].eq("aquatic biome").all()

    # Three Zenodo CSVs plus four ranks × two identifier types in GMT format.
    assert len(result.build.paths) == 11
    assert all(path.is_file() for path in result.build.paths.values())
    assert {"binary", "multistate", "numeric"}.issubset(result.build.paths)

    for category in ("binary", "multistate", "numeric"):
        path = result.build.paths[category]
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("# bugphyzz ")

        exported = pd.read_csv(path, comment="#", dtype={"NCBI_ID": "string"})
        assert not exported.empty
        assert exported["Evidence"].notna().all()
        assert not exported["NCBI_ID"].str.contains(
            r"^[a-z]__",
            regex=True,
            na=False,
        ).any()

    gmt_paths = {
        key: path
        for key, path in result.build.paths.items()
        if "--" in key
    }
    assert len(gmt_paths) == 8
    for path in gmt_paths.values():
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.startswith("# bugphyzz ")
