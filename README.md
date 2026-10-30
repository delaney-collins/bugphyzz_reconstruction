# Bugphyzz reconstruction

This package prepares microbial trait data, reconstructs missing trait annotations, and writes the CSV and GMT files for a Bugphyzz data release. It runs independently of the user-facing `bugphyzz` package.

## Install

Use Python 3.10 or newer. Open a terminal in this project folder and run:

```powershell
python -m pip install -e .
```

## Run

From the repository root, run a smaller example with two traits:

```powershell
python examples/all.py --traits motility shape --output-dir asr_output/example
```

Run the full pipeline with all default traits:

```powershell
python examples/all.py --output-dir asr_output/full_build
```

Use `python examples/all.py --help` for options. Quote trait names containing spaces, for example `--traits "growth temperature"`. Output paths are relative to your current working directory; bundled resources are located automatically in the installed package.

The pipeline reads the bundled tree, taxonomy, and signature thresholds from `src/bugphyzz_reconstruction/resources/`. An internet connection is needed to retrieve source trait data.

## Outputs

By default, results are written to `asr_output/full_build/` (override with `--output-dir`):

- `bugphyzz_binary.csv`
- `bugphyzz_multistate.csv`
- `bugphyzz_numeric.csv`
- Eight GMT files containing trait signatures for genus, species, strain, and mixed ranks, using either NCBI IDs or taxon names.
- `skipped_vocabulary_terms.txt`, which reports skipped vocabulary terms and traits.

Running the pipeline again writes to the same output location.
