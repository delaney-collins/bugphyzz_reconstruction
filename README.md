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

## User-curated binary traits

You can reconstruct your own observations without retrieving source data from Google Sheets or BacDive. After installing the package, run:

```powershell
python examples/reconstruct_binary_trait.py path/to/observations.csv --output asr_output/my_binary_trait.csv
```

Replace `path/to/observations.csv` with your file. CSV files use commas; TSV and TXT files use tabs. Input and output paths are relative to your current working directory. The script automatically uses the bundled frozen 2023 taxonomy and LTP tree, creates the output directory, and writes the propagated table to the specified CSV. Existing output files are overwritten.

Supply one binary trait group per file, with these columns:

| Columns | Contents |
| --- | --- |
| `NCBI_ID`, `Parent_NCBI_ID` | NCBI taxonomy identifiers. Include at least one of these columns and at least one usable identifier; either value may be blank when unknown. |
| `Taxon_name` | Taxon name associated with the observation. |
| `Attribute` | The binary trait name. |
| `Attribute_value` | `TRUE` or `FALSE` (also accepts `1` or `0`). |
| `Attribute_type` | `binary` for every row. |
| `Attribute_group` | One consistent trait group, typically the same name as `Attribute`. |
| `Attribute_source` | The source or citation for your observation. |
| `Confidence_in_curation` | Curation confidence, for example `high`. |
| `Frequency` | Observation frequency, for example `always`. |
| `Evidence` | Evidence category, for example `exp` for experimental observations. |

Custom trait names are accepted by default, using an in-memory vocabulary extension. Add `--strict-vocabulary` to require the packaged vocabulary. Taxonomy mapping still uses the bundled references, so not every supplied taxon will necessarily contribute an annotated tree tip.

By default, the script retains the pipeline's historical one-percent coverage rule for deciding whether to run ASR. For a small demonstration, you can explicitly override this with a minimum number of distinct annotated LTP tips:

```powershell
python examples/reconstruct_binary_trait.py path/to/observations.csv --min-annotated-tips 3 --output asr_output/my_binary_trait.csv
```

This override changes the coverage criterion; three input rows do not necessarily map to three distinct tree tips. Check the printed coverage table and `ASR run` value to see whether reconstruction ran. The output is `result.propagated`; the Python API also exposes `result.raw`, `result.filtered`, `result.prepared`, and `result.coverage` for inspection. Use `python examples/reconstruct_binary_trait.py --help` for all options.

## Outputs

By default, results are written to `asr_output/full_build/` (override with `--output-dir`):

- `bugphyzz_binary.csv`
- `bugphyzz_multistate.csv`
- `bugphyzz_numeric.csv`
- Eight GMT files containing trait signatures for genus, species, strain, and mixed ranks, using either NCBI IDs or taxon names.
- `skipped_vocabulary_terms.txt`, which reports skipped vocabulary terms and traits.

Running the pipeline again writes to the same output location.
