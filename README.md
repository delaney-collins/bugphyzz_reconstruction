# Bugphyzz reconstruction

This package prepares microbial trait data, reconstructs missing trait annotations, and writes the CSV and GMT files for a Bugphyzz data release. It runs independently of the user-facing `bugphyzz` package.

## Install

Use Python 3.10 or newer. Open a terminal in this project folder and run:

```powershell
python -m pip install -e .
```

## Run

Open `src/bugphyzz_reconstruction/asr/all.py` and change `package_dir` to the full path of this project folder on your computer. You can also edit the `traits` list to choose which traits to process.

Then run:

```powershell
python -m bugphyzz_reconstruction.asr.all
```

The pipeline reads the bundled tree, taxonomy, and signature thresholds from `src/bugphyzz_reconstruction/resources/`. An internet connection is needed to retrieve source trait data.

## Outputs

Results are written to `asr_output/full_build/`:

- `bugphyzz_binary.csv`
- `bugphyzz_multistate.csv`
- `bugphyzz_numeric.csv`
- Eight GMT files containing trait signatures for genus, species, strain, and mixed ranks, using either NCBI IDs or taxon names.
- `skipped_vocabulary_terms.txt`, which reports skipped vocabulary terms and traits.

Running the pipeline again writes to the same output location.
