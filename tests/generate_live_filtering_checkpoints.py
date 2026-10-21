"""Generate raw and filtered checkpoints from the live Bugphyzz sources.

This diagnostic utility calls the translated ``physiologies()`` source
adapter, then writes ``00_raw.tsv`` and ``01_filtered.tsv`` for each returned
trait.  It is intended for reviewing changes in Google Sheets, BacDive, and
the source-transformation code without running ancestral reconstruction.

Run all available traits from the repository root::

    python tests/generate_live_filtering_checkpoints.py

Run selected traits by repeating ``--trait``::

    python tests/generate_live_filtering_checkpoints.py \
        --trait motility --trait "growth temperature"

The destination is preserved unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# Permit direct execution from a src-layout checkout without requiring an
# editable installation in the selected interpreter.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bugphyzz_reconstruction.preprocessing.filtering import (  # noqa: E402
    filter_data,
)
from bugphyzz_reconstruction.sources.physiologies import (  # noqa: E402
    physiologies,
)
from bugphyzz_reconstruction.taxonomy.frozen import (  # noqa: E402
    FrozenTaxonomyProvider,
)


DESKTOP_ROOT = REPOSITORY_ROOT.parent
DEFAULT_OUTPUT_ROOT = (
    DESKTOP_ROOT / "asr_python" / "python_live_filtering_checkpoints"
)
DEFAULT_RESOURCE_DIR = (
    REPOSITORY_ROOT / "src" / "bugphyzz_reconstruction" / "resources"
)
INVALID_ATTRIBUTES_PATTERN = re.compile(
    r"^Invalid attributes in (.*?): (.*)$",
    flags=re.DOTALL,
)


def prepare_output_root(path: Path, *, overwrite: bool) -> None:
    """Create a clean output root, replacing only an explicitly named path."""
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {path}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True)


def validate_trait_name(name: str) -> str:
    """Return a trait name that is safe to use as one directory component."""
    normalized = name.strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"Invalid empty or relative trait name: {name!r}")
    if Path(normalized).name != normalized or "/" in normalized or "\\" in normalized:
        raise ValueError(f"Trait name contains a path separator: {name!r}")
    return normalized


def write_tsv(table: pd.DataFrame, path: Path) -> None:
    """Write a checkpoint with explicit, R-compatible missing values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, na_rep="NA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read live Google Sheets/BacDive resources and generate "
            "00_raw.tsv and 01_filtered.tsv checkpoints."
        )
    )
    parser.add_argument(
        "--trait",
        action="append",
        default=[],
        help=(
            "Trait to load. Repeat for multiple traits; omit to load all "
            "traits exposed by physiologies()."
        ),
    )
    parser.add_argument(
        "--resource-dir",
        type=Path,
        default=DEFAULT_RESOURCE_DIR,
        help="Package resource directory used by physiologies().",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination for trait checkpoint directories.",
    )
    parser.add_argument(
        "--full-source",
        action="store_true",
        help="Retain the full source citation when supported by physiologies().",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing output root before writing checkpoints.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resource_dir = args.resource_dir.resolve()
    output_root = args.output_root.resolve()
    taxonomy_resource_dir = resource_dir / "taxonomy_2023"

    if not resource_dir.is_dir():
        raise FileNotFoundError(
            f"Package resource directory does not exist: {resource_dir}"
        )
    if not taxonomy_resource_dir.is_dir():
        raise FileNotFoundError(
            "Taxonomy resource directory does not exist: "
            f"{taxonomy_resource_dir}"
        )

    selected_traits = list(dict.fromkeys(args.trait))
    keyword: str | list[str] = selected_traits if selected_traits else "all"

    # Fetch before creating the destination so a network/source failure does
    # not destroy a previous successful checkpoint set under --overwrite.
    raw_traits = physiologies(
        keyword,
        full_source=args.full_source,
        resource_dir=resource_dir,
    )
    if not raw_traits:
        raise RuntimeError("physiologies() returned no trait tables.")

    taxonomy = FrozenTaxonomyProvider.from_tsv(
        resource_dir=taxonomy_resource_dir,
        load_attributes=True,
    )
    prepare_output_root(output_root, overwrite=args.overwrite)

    skipped_terms: dict[str, set[str]] = defaultdict(set)
    manifest_rows: list[dict[str, object]] = []

    for supplied_name, raw in raw_traits.items():
        trait_name = validate_trait_name(str(supplied_name))
        trait_dir = output_root / trait_name
        trait_dir.mkdir(parents=True)
        write_tsv(raw, trait_dir / "00_raw.tsv")

        caught_warnings: list[warnings.WarningMessage]
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            filtered = filter_data(raw, taxonomy=taxonomy)

        for warning in caught_warnings:
            warning_text = str(warning.message)
            match = INVALID_ATTRIBUTES_PATTERN.match(warning_text)
            if match:
                warning_trait, terms = match.groups()
                skipped_terms[warning_trait].update(
                    term.strip()
                    for term in terms.split("---")
                    if term.strip()
                )
            warnings.showwarning(
                warning.message,
                warning.category,
                warning.filename,
                warning.lineno,
            )

        if filtered is None:
            # Keep the checkpoint pair structurally complete while recording
            # that filtering found no usable rows in the manifest.
            filtered = pd.DataFrame()
            status = "filtering produced no usable rows"
        elif filtered.empty:
            status = "filtered table is empty"
        else:
            status = "ok"
        write_tsv(filtered, trait_dir / "01_filtered.tsv")

        manifest_rows.append(
            {
                "trait": trait_name,
                "raw_rows": len(raw),
                "filtered_rows": len(filtered),
                "rows_dropped": len(raw) - len(filtered),
                "status": status,
                "raw_file": str((trait_dir / "00_raw.tsv").resolve()),
                "filtered_file": str(
                    (trait_dir / "01_filtered.tsv").resolve()
                ),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(
        output_root / "manifest.tsv",
        sep="\t",
        index=False,
        na_rep="NA",
    )

    skipped_lines = ["Skipped vocabulary terms", "========================", ""]
    if skipped_terms:
        for trait in sorted(skipped_terms, key=str.casefold):
            skipped_lines.append(f"[{trait}]")
            skipped_lines.extend(
                f"- {term}"
                for term in sorted(skipped_terms[trait], key=str.casefold)
            )
            skipped_lines.append("")
    else:
        skipped_lines.extend(["None", ""])
    (output_root / "skipped_vocabulary_terms.txt").write_text(
        "\n".join(skipped_lines) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "implementation": "python",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "live_sources": ["Google Sheets", "BacDive"],
        "selected_traits": selected_traits or "all",
        "resource_dir": str(resource_dir),
        "taxonomy_resource_dir": str(taxonomy_resource_dir),
        "full_source": bool(args.full_source),
        "trait_count": len(raw_traits),
        "python_version": sys.version,
        "pandas_version": pd.__version__,
    }
    (output_root / "generator_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Generated live filtering checkpoints in: {output_root}")
    for row in manifest_rows:
        print(
            f"  {row['trait']}: {row['raw_rows']:,} raw, "
            f"{row['filtered_rows']:,} filtered ({row['status']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
