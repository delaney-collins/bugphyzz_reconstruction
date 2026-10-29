# ASR parity checkpoints and failure interpretation

This guide explains the diagnostic checkpoint files produced for the discrete and continuous ancestral-state reconstruction (ASR) pipelines. The checkpoints are intended to answer two questions:

1. At what stage do the Python and R pipelines first diverge?
2. What class of problem is suggested by that first divergence?

The most useful debugging rule is to compare checkpoints in numeric order and focus on the **first meaningful mismatch**. Later files usually inherit earlier differences, so a large discrepancy in `99_final_propagated.tsv` does not necessarily indicate a problem in final assembly.

## General comparison rules

- Compare row content by stable keys rather than file order. R and Python may emit equivalent rows in different orders.
- Treat small floating-point differences as expected unless they change a label, threshold decision, or retained row.
- Compare missing-value masks separately from numeric values. A matching value comparison can hide a disagreement about which cells are `NA`.
- For probability matrices, compare column labels and tree-tip or node identifiers before comparing values.
- Check run metadata before interpreting a mismatch. A different tree, taxonomy table, source snapshot, trait configuration, or coverage threshold makes downstream equality unlikely.
- R-only `.rds`, printed model, and structure files are inspection artifacts. They do not need a byte-for-byte Python counterpart.

## Discrete ASR checkpoints

Discrete traits include binary and multistate attributes. The pipeline builds a tip-by-state probability matrix, fits a continuous-time Markov model, reconstructs state probabilities, maps reconstructed tips back to taxonomy identifiers, and combines predictions with observed rows.

### Input construction and coverage

#### `00_prepared_input.tsv`

The prepared observation table supplied to the discrete ASR pipeline. It is the handoff from preprocessing into ASR.

Use it to verify:

- taxonomy and identifier normalization;
- accepted vocabulary and state labels;
- scores or frequencies derived from curated observations;
- attribute type and group selection;
- duplicate handling performed before ASR.

If this file differs, the problem is upstream of ASR: source loading, filtering, vocabulary normalization, taxonomy mapping, preparation, or numeric boundary behavior. Do not diagnose model fitting until this checkpoint is acceptably aligned.

#### `00_run_metadata.tsv`

Records the run context, such as trait name, input counts, tree-tip counts, coverage settings, and other diagnostic parameters.

A mismatch here often means the two runs are not directly comparable. Confirm that both runs used the same tree, taxonomy resources, trait, and threshold configuration.

#### `01_joined_tip_data.tsv`

Prepared observations after joining them to the phylogenetic tree-tip identifiers.

This checkpoint tests whether observation identifiers map to the tree in the same way. Missing rows usually indicate identifier type problems, taxonomy-version differences, parent-ID differences, or failed joins. Extra rows can indicate duplicated mappings.

#### `02_annotated_tips_long.tsv`

The mapped, annotated tips in long form: generally one record per tip and state contribution.

Differences introduced here point to aggregation, duplicate resolution, score conversion, or state-label handling after the taxonomy join.

#### `03_annotated_tip_state_matrix.tsv`

The annotated tip-by-state probability matrix used as the observed portion of the model input.

Check:

- identical tip identifiers;
- identical state columns and column order after alignment;
- row sums and probability bounds;
- matching `NA` masks;
- values near label or probability cutoffs.

Small numeric differences may come from floating-point arithmetic. A changed maximum-probability state, a different tie, or a changed `NA` mask is substantive.

#### `04_coverage.tsv`

Documents how many tree tips have observations and whether the trait passes the reconstruction threshold.

If Python reconstructs a trait that R skips, or vice versa, inspect this file first. Historical percentage calculations and integer rounding can differ at an exact boundary. When `min_annotated_tips` is explicitly supplied, confirm that both comparisons use the same count threshold.

#### `05_unannotated_uniform_state_matrix.tsv`

The state-probability rows assigned to unannotated tree tips before model fitting. With no observation, each state should normally receive a uniform probability.

Differences suggest disagreement about which tips are unannotated, the number of allowed states, or state-column construction.

#### `06_input_matrix_before_tree_order.tsv`

The combined annotated and unannotated state matrix before rows are reordered to match the phylogenetic tree.

If `03` and `05` agree but this file does not, inspect concatenation, duplicate tip IDs, missing tips, and matrix-column alignment.

#### `07_input_validation.tsv`

Diagnostic checks on the completed model-input matrix, such as dimensions, missing values, duplicated tips, probability bounds, row sums, and correspondence with the tree.

A failure here means the model should not be trusted even if it runs. Common causes are absent tree tips, duplicate rows, non-finite values, state-column mismatches, or probabilities that no longer sum appropriately after rounding or clipping.

#### `08_input_matrix_tree_order.tsv`

The final discrete model-input matrix in exact tree-tip order.

If `06` agrees by key but `08` does not, the likely issue is tree order, tip-name normalization, or row indexing rather than biological data. This is the last checkpoint before model fitting.

### Model fitting

#### `10_model_rates.tsv`

The fitted transition-rate parameters of the discrete Markov model.

Differences can be expected when the R and Python optimizers, constraints, parameterization, starting values, or convergence tolerances differ. Large or structurally different rates warrant checking `08` first, followed by the model design in `12` and root prior in `13`.

#### `11_model_scalars.tsv`

Scalar model diagnostics, such as likelihood-related values, convergence information, or other fitted summaries.

Use this to distinguish harmless parameter-format differences from a materially different optimum. A failed convergence flag, non-finite objective, or substantially worse likelihood indicates a fitting problem.

#### `12_model_index_matrix.tsv`

Describes which transition rates are shared or independently estimated between states.

If this differs, the pipelines are fitting different model structures. Rate values cannot be meaningfully compared until the index matrix agrees.

#### `13_model_root_prior.tsv`

The root-state prior used for ancestral reconstruction.

Different priors can alter every reconstructed probability even when the input matrix and fitted rates otherwise match. Confirm state names and their prior probabilities.

#### `14_model_q_matrix.tsv` (Python diagnostic)

The fitted instantaneous transition-rate matrix assembled from the fitted parameters.

This is the clearest cross-language object for verifying that the fitted model has been translated into the same state-to-state process. Compare it by state labels, not only by position.

#### `14_model_print.txt` and `15_model_structure.txt` (R diagnostics)

Human-readable and structural snapshots of the R fitted model. They help identify the model class, rate ordering, optimizer result, and stored components but are not expected to match a Python text representation byte for byte.

#### `fitMk_model.rds` (R artifact)

The serialized fitted R model. It is useful for interactive inspection in R or for regenerating downstream R checkpoints without refitting. It is not a parity target by itself.

### Reconstruction and taxonomy mapping

#### `20_ancr_all_tip_node_probabilities.tsv`

Reconstructed state probabilities for all returned tips and internal nodes.

This is the first direct test of the ancestral reconstruction calculation. If `08` and the effective fitted model agree but `20` diverges materially, investigate likelihood implementation, branch-length use, root treatment, conditional-likelihood normalization, node numbering, and probability clipping.

Compare tip and node identity explicitly; R and Python may number internal nodes differently.

#### `21_ancr_summary.tsv` (Python diagnostic)

A compact summary of the Python reconstruction result, including dimensions and probability diagnostics. It helps detect missing nodes, invalid probabilities, or structural surprises without comparing the full matrix.

#### `21_ancr_structure.txt` and `ancr_result.rds` (R diagnostics)

The R result's structural description and serialized object. Use these to understand how R labels tips and nodes and where probabilities are stored. They are inspection artifacts rather than tabular parity targets.

#### `22_reconstructed_mappable_unannotated.tsv`

The reconstructed rows restricted to unannotated tips that can be mapped back to taxonomy.

If `20` agrees but `22` does not, the discrepancy is selection or identifier mapping, not ASR mathematics. Check the annotated/unannotated mask, tip labels, and mappability rules.

#### `23_reconstructed_joined_to_taxonomy.tsv`

Reconstructed unannotated tips joined to taxonomy identifiers and names.

Differences point to taxonomy-table version, join-key type, duplicated taxonomy mappings, or parent/child identifier handling.

#### `24_reconstructed_deduplicated_by_ncbi.tsv`

Taxonomy-mapped reconstructions after resolving multiple tree-tip records that map to the same NCBI identifier.

If `23` agrees but `24` differs, inspect deterministic sorting and the rule used to select or aggregate duplicates.

#### `25_asr_predictions_long.tsv`

The reconstructed state probabilities converted into the long-form BugPhyzz observation schema.

Differences first appearing here usually concern wide-to-long reshaping, state labels, score formatting, evidence labels, attribute metadata, or filtering of negligible probabilities—not the ASR calculation itself.

### Final assembly

#### `26_observed_rows_retained.tsv`

The original prepared observations retained for the final output.

This verifies that observed data are preserved and not replaced by predictions. Missing rows suggest an assembly filter, schema mismatch, or incorrect observed/reconstructed mask.

#### `27_combined_before_formatting.tsv`

Observed rows and ASR predictions combined before final column selection, typing, sorting, and formatting.

If `25` and `26` agree but `27` differs, inspect concatenation, column alignment, duplicate removal, and precedence rules.

#### `99_final_propagated.tsv`

The final propagated trait table returned by the discrete pipeline.

Differences first introduced here are normally formatting or output-contract issues: column order, data types, taxonomy fields, evidence labels, sorting, or final deduplication. If an earlier checkpoint differs, treat `99` as a symptom rather than the source.

## Continuous ASR checkpoints

Continuous traits use one numeric value per annotated tip, reconstruct values across the tree, calculate nearest-sequenced-taxon distance or NSTI-style diagnostics, map predictions back to taxonomy, and combine predictions with retained observations.

### Input construction and coverage

#### `00_prepared_input.tsv`

The prepared numeric observations supplied to continuous ASR.

Verify numeric parsing, units, identifier normalization, duplicate handling, scores or frequencies, and any aggregation of multiple observations. If length or width produces no usable rows, this is the first file to inspect; an empty or absent checkpoint indicates filtering or preparation failed before ASR.

#### `00_run_metadata.tsv`

The continuous run context: trait, input size, tree size, threshold settings, and other diagnostics. Confirm comparable resources and configuration before interpreting parity.

#### `01_joined_tip_data.tsv`

Prepared numeric observations joined to tree tips.

Missing tips indicate identifier or taxonomy mapping problems. Multiple observations per tip can expose differences in aggregation policy.

#### `02_annotated_tip_values.tsv`

The single numeric value selected or aggregated for each annotated tree tip.

If `01` agrees but `02` differs, inspect duplicate aggregation, weighting, coercion, and missing-value handling.

#### `03_coverage.tsv`

The number and fraction of annotated tree tips and the decision to run or skip continuous ASR.

At exact percentage boundaries, historical rounding behavior can affect whether a trait passes. An explicit `min_annotated_tips` should be compared as a count, not as a rounded percentage.

#### `04_unannotated_tip_values.tsv`

Placeholder values for unannotated tree tips before reconstruction.

Differences mean the pipelines disagree about which tips lack observations or about the placeholder/missing-value representation.

#### `05_input_values_before_tree_order.tsv`

Annotated values and unannotated placeholders combined before tree ordering.

If `02` and `04` agree but `05` differs, inspect concatenation, duplicate tips, and identifier type.

#### `06_input_validation.tsv`

Validation of the continuous input vector, including dimensions, duplicate IDs, missingness, finite annotated values, and correspondence with tree tips.

A validation failure indicates invalid model input. Typical causes include nonnumeric values, all-missing inputs, duplicated tips, absent tree tips, or an incorrect annotated/unannotated mask.

#### `07_input_vector_tree_order.tsv`

The final continuous input vector in exact phylogenetic tree-tip order.

If `05` agrees by key but `07` differs, investigate tree order or tip-name normalization.

#### `08_continuous_asr_structure.txt` (R diagnostic)

A structural snapshot of the R continuous-ASR result or setup. It helps identify returned components, labels, and dimensions. It is not expected to have a byte-identical Python equivalent.

#### `continuous_asr_result.rds` (R artifact)

The serialized R reconstruction result, useful for interactive inspection and downstream checkpoint regeneration. It is not itself a cross-language parity target.

### Reconstruction, selection, and mapping

#### `09_all_reconstructed_states.tsv`

Reconstructed continuous values for all returned tips and internal nodes.

This is the main numerical ASR checkpoint. If `07` agrees but `09` differs substantially, inspect the continuous reconstruction method, branch lengths, root assumptions, node indexing, and treatment of missing tips. Compare tip identities separately from internal-node identities.

#### `10_mappable_unannotated_states.tsv`

Reconstructed values restricted to unannotated tips that can be mapped back into the output taxonomy.

If `09` agrees but `10` differs, the problem is selection or identifier mapping rather than continuous ASR.

#### `11_nsti.tsv`

Distance-to-observation information for reconstructed tips, typically the phylogenetic distance to the nearest annotated or sequenced taxon used as an NSTI-style confidence diagnostic.

Differences can arise from tree traversal, branch-length interpretation, the annotated-tip mask, or tie handling. A matching reconstructed value with a different NSTI points to distance calculation rather than ASR.

#### `12_reconstructed_states_with_nsti.tsv`

Reconstructed continuous values combined with their NSTI values.

If `10` and `11` agree independently but `12` differs, inspect the join keys and row alignment.

#### `13_reconstructed_joined_to_taxonomy.tsv`

Reconstructed values and NSTI joined to taxonomy identifiers and names.

Differences indicate taxonomy-resource or join behavior rather than reconstruction mathematics.

#### `14_final_asr_predictions.tsv`

The mapped reconstructions formatted as predicted BugPhyzz observations.

Differences first appearing here concern schema conversion, evidence labels, confidence or frequency conversion, attribute metadata, typing, or formatting.

### Final assembly

#### `15_observed_rows_retained.tsv`

Prepared observed numeric rows preserved for the final output.

Use this to confirm that curated values remain intact and are not overwritten by reconstructions. Missing observations indicate an assembly or filtering error.

#### `99_final_propagated.tsv`

The final continuous propagated table containing retained observations plus reconstructed predictions.

If `14` and `15` agree but `99` does not, inspect concatenation, final schema enforcement, sorting, type conversion, and deduplication. Earlier mismatches should be diagnosed at their first checkpoint.

## Interpreting the first mismatch

| First divergent stage | Most likely problem area |
|---|---|
| `00_prepared_input` | Source loading, filtering, vocabulary, taxonomy normalization, preparation, floating-point label boundaries |
| `01`–`03` discrete or `01`–`02` continuous | Tree-tip mapping, duplicate aggregation, score/value conversion, state construction |
| Coverage checkpoint | Threshold calculation, rounding, different tree size, explicit minimum-tip setting |
| Completed input before ordering | Annotated/unannotated mask, concatenation, state columns, duplicate tips |
| Tree-ordered input | Tree-tip names, ordering, indexing, tree/resource version |
| Model rates or model structure | Different model parameterization, optimizer, constraints, root prior, convergence |
| First reconstructed-state checkpoint | ASR algorithm, branch lengths, root treatment, likelihood normalization, node identity |
| Mappable reconstructed subset | Annotated mask, tip selection, identifier parsing |
| Taxonomy-joined reconstruction | Taxonomy snapshot, join keys, duplicate taxonomy mappings |
| Long-form predictions | Reshaping, metadata, evidence labels, score/confidence conversion |
| Retained observations | Incorrect filtering or observed/predicted precedence |
| Final propagated output only | Formatting, column types/order, sorting, concatenation, final deduplication |

## Common failure patterns

### Same rows, different order

Sort by stable identifiers and state or attribute columns before comparison. Ordering alone is not a parity failure unless a downstream operation incorrectly depends on position.

### Same shape, different `NA` mask

The `NA` mask marks which matrix or vector positions are missing. In ASR, missing means “unknown and to be reconstructed,” not zero and not a negative observation. A mask mismatch changes which tips the model treats as observed and can alter all downstream estimates.

### Tiny probability differences near a cutoff

Floating-point and optimizer differences are normal. They become important when they cross a classification boundary, change a tie, alter the selected state, or produce an invalid value just outside `[0, 1]`. Clip only small numerical excursions at the point probabilities enter score/frequency conversion; do not use clipping to conceal a materially incorrect model result.

### Matching prepared input but different final predictions

This localizes the discrepancy to ASR or downstream mapping. Compare the final tree-ordered input, fitted model representation, and first reconstructed-state checkpoint in that order. If those agree, move to mappability and taxonomy joins.

### Matching ASR probabilities but different final exports

The reconstruction is not the source of the discrepancy. Inspect taxonomy mapping, duplicate resolution, long-form conversion, retained observations, and final schema formatting.

### Trait is entirely skipped

Check, in order:

1. whether `00_prepared_input.tsv` exists and contains usable rows;
2. whether rows map to tree tips in `01_joined_tip_data.tsv`;
3. whether the coverage checkpoint passes the configured threshold;
4. whether validation reports finite values and a valid observed/missing mask.

An absent checkpoint usually means the stage was never reached; an empty checkpoint means it ran but produced no qualifying records.

### Old final exports disagree with current Python output

Final exports can reflect an older source snapshot, taxonomy, phylogenetic tree, or package behavior. A checkpoint comparison against a contemporaneous R reference is stronger evidence than a comparison against an old published export. Start with metadata and prepared input before concluding that prediction logic is wrong.

## Recommended parity workflow

1. Freeze the source data, taxonomy resources, tree, trait configuration, and software versions for both runs.
2. Generate all checkpoints from R and Python.
3. Compare metadata and `00_prepared_input.tsv`.
4. Compare by stable identifiers, ignoring row order unless order is the subject of the checkpoint.
5. Find the first meaningful mismatch.
6. Diagnose only the transformation between the last matching checkpoint and the first divergent one.
7. Use tolerances for floating-point values, but compare labels, keys, dimensions, and missingness exactly.
8. Re-run only the affected arm and trait until the first mismatch is resolved.
9. Confirm that the correction propagates through `99_final_propagated.tsv`.

This staged approach prevents source-data, model-fitting, taxonomy-mapping, and final-formatting differences from being conflated into a single end-output discrepancy.
