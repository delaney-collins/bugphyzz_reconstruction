"""Reconstruct one user-curated binary trait from CSV, TSV, or TXT.

Install the package with ``python -m pip install -e .`` before running.
This workflow uses bundled taxonomy and tree resources; it does not fetch
observations from Google Sheets or BacDive.
"""

import argparse
from pathlib import Path


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, help="Input CSV, TSV, or tab-delimited TXT file.")
    parser.add_argument(
        "--output", type=Path, default=Path("asr_output/binary_trait_reconstruction.csv"),
        help="Output CSV (default: asr_output/binary_trait_reconstruction.csv).",
    )
    parser.add_argument(
        "--min-annotated-tips", type=positive_integer, default=None,
        help="Override the historical one-percent coverage rule with a positive tip count.",
    )
    parser.add_argument(
        "--strict-vocabulary", action="store_true",
        help="Require packaged vocabulary terms instead of accepting custom trait names.",
    )
    args = parser.parse_args()
    observations_path = args.observations.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not observations_path.is_file():
        parser.error(f"Observation file does not exist: {observations_path}")
    if observations_path == output_path:
        parser.error("Output must differ from the input observation file.")

    from bugphyzz_reconstruction.asr import reconstruct_trait
    from bugphyzz_reconstruction.asr.ltp import ltp
    from bugphyzz_reconstruction.io.observations import read_trait_observations
    from bugphyzz_reconstruction.taxonomy.frozen import FrozenTaxonomyProvider

    observations = read_trait_observations(observations_path)
    if not observations["Attribute_type"].eq("binary").all():
        parser.error("This example requires Attribute_type=binary.")
    taxonomy = FrozenTaxonomyProvider.from_tsv(load_attributes=True)
    resources = ltp()
    result = reconstruct_trait(
        observations,
        taxonomy,
        resources,
        allow_custom_vocabulary=not args.strict_vocabulary,
        min_annotated_tips=args.min_annotated_tips,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.propagated.to_csv(output_path, index=False)

    # Other inspectable stages: result.raw, result.filtered, result.prepared.
    print(result.coverage.to_string(index=False))
    print(f"ASR run: {result.asr_run}")
    print(f"Reconstructed rows: {len(result.propagated):,}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
