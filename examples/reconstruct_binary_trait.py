"""Reconstruct one self-curated binary trait."""

from pathlib import Path
import sys


# Make the src-layout package importable when this example is run directly
# from a source checkout. This is unnecessary after installing the package.
repository_root = Path(__file__).resolve().parents[1]
source_root = repository_root / "src"
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

from bugphyzz_reconstruction.asr import reconstruct_trait
from bugphyzz_reconstruction.asr.ltp import ltp
from bugphyzz_reconstruction.taxonomy.frozen import FrozenTaxonomyProvider

# The CSV contains the complete raw-observation schema, including Attribute,
# Attribute_value, Attribute_type, and Attribute_group.
observations_path = (
    Path(__file__).resolve().parents[2] / "binary_curation_example.csv"
)

taxonomy = FrozenTaxonomyProvider.from_tsv(load_attributes=True)
resources = ltp()

result = reconstruct_trait(
    observations_path,
    taxonomy,
    resources,
    allow_custom_vocabulary=True,
    min_annotated_tips=3,
)

output_path = observations_path.with_name("binary_trait_reconstruction.csv")
result.propagated.to_csv(output_path, index=False)


# The reconstructed table is result.propagated. Other inspectable stages are
# available as result.raw, result.filtered, result.prepared, and result.coverage.
print(result.coverage.to_string(index=False))
print(f"ASR run: {result.asr_run}")
print(f"Reconstructed rows: {len(result.propagated):,}")
print(f"Saved to: {output_path}")
