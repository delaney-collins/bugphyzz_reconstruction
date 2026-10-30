"""Run a full release build, or select traits for a smaller example.

Install the package first with ``python -m pip install -e .``.
Source trait retrieval requires an internet connection.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import re
import warnings

DEFAULT_TRAITS = [
    "growth temperature",
    "coding genes",
    "genome size",
    "length",
    "width",
    "optimal ph",
    "mutation rate per site per generation",
    "mutation rate per site per year",
    "animal pathogen",
    "antimicrobial sensitivity",
    "biofilm formation",
    "extreme environment",
    "health associated",
    "hydrogen gas producing",
    "lactate producing",
    "motility",
    "plant pathogenicity",
    "spore formation",
    "host-associated",
    "sphingolipid producing",
    "butyrate producing",
    "aerophilicity",
    "gram stain",
    "biosafety level",
    "COGEM pathogenicity rating",
    "shape",
    "spore shape",
    "arrangement",
    "hemolysis",
    "habitat",
    "disease association",
    "antimicrobial resistance",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("asr_output/full_build"),
        help="Output directory, relative to the current working directory by default.",
    )
    parser.add_argument(
        "--traits", nargs="+", default=DEFAULT_TRAITS,
        help='Traits to process (quote names containing spaces); defaults to all traits.',
    )
    args = parser.parse_args()

    from importlib.resources import files
    from bugphyzz_reconstruction.sources.physiologies import physiologies
    from bugphyzz_reconstruction.taxonomy.frozen import FrozenTaxonomyProvider
    from bugphyzz_reconstruction.asr.ltp import ltp
    from bugphyzz_reconstruction.asr.driver import prepare_and_build_bugphyzz

    resource_dir = Path(str(files("bugphyzz_reconstruction").joinpath("resources")))
    output_dir = args.output_dir.resolve()
    taxonomy = FrozenTaxonomyProvider.from_tsv(
        resource_dir / "taxonomy_2023", load_attributes=True,
    )
    resources = ltp(resource_dir=resource_dir / "taxonomy_2023")
    raw_traits = physiologies(
        args.traits, full_source=False, resource_dir=resource_dir,
    )

    skipped_terms: dict[str, set[str]] = defaultdict(set)
    invalid_attributes_pattern = re.compile(
        r"^Invalid attributes in (.*?): (.*)$",
        flags=re.DOTALL,
    )


    def collect_skipped_terms(
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file=None,
        line: str | None = None,
    ) -> None:
        """Record invalid vocabulary terms while preserving console warnings."""
        warning_text = str(message)
        match = invalid_attributes_pattern.match(warning_text)
        if match:
            trait, terms = match.groups()
            skipped_terms[trait].update(
                term.strip()
                for term in terms.split("---")
                if term.strip()
            )
        original_showwarning(
            message,
            category,
            filename,
            lineno,
            file=file,
            line=line,
        )


    with warnings.catch_warnings():
        original_showwarning = warnings.showwarning
        warnings.showwarning = collect_skipped_terms
        result = prepare_and_build_bugphyzz(
            raw_traits,
            taxonomy,
            resources,
            output_dir=output_dir,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    skipped_terms_path = output_dir / "skipped_vocabulary_terms.txt"
    report_lines = ["Skipped vocabulary terms", "========================", ""]

    if skipped_terms:
        for trait in sorted(skipped_terms, key=str.casefold):
            report_lines.append(f"[{trait}]")
            report_lines.extend(
                f"- {term}"
                for term in sorted(skipped_terms[trait], key=str.casefold)
            )
            report_lines.append("")
    else:
        report_lines.extend(["None", ""])

    report_lines.extend(["Fully skipped traits", "====================", ""])
    if result.skipped_traits:
        report_lines.extend(
            f"- {name}: {reason}"
            for name, reason in sorted(result.skipped_traits.items())
        )
    else:
        report_lines.append("None")

    skipped_terms_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("Written files:")
    for name, path in result.build.paths.items():
        print(f"  {name}: {path}")

    print("\nSkipped traits:")
    for name, reason in result.skipped_traits.items():
        print(f"  {name}: {reason}")

    print(f"\nSkipped vocabulary report: {skipped_terms_path}")


if __name__ == "__main__":
    main()
