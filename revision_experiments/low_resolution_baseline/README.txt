Controlled low-resolution U-FNO experiment

design.txt and plan.txt specify the experiment before controlled testing.
Original model sources, old weights and prior test outputs remain unchanged.
The main comparison is new matched-budget retraining, not a reconstruction of
the old full-window experiment. It includes four architectures and seeds42/43/44.

Data
split.json — fixed400 train /100 validation source IDs and independent coarse test IDs.
training_normalization.json — moments fitted only on the400 training reservoirs.
source_normalized.pt — cache of source tensors using those fixed moments.
The new independent coarse100 data and existing fine100 data are outside this
folder and their complete file hashes are included in training_provenance.json.

Training
controlled_training.py — model adaptation, data processing and matched fitting.
*_history.json — every epoch, pair counts, learning rate and validation loss.
*_completed.json, training_summary.json — stopping and validation-selected epochs.
checkpoints/*_best.pt — selected checkpoints; *_last.pt — RNG-state continuation.
controlled_training_source_v1.py preserves the fresh running producer source.
Provenance-only entry-point checks were added while that uninterrupted run
continued. Its numerical function/class ASTs remained identical; no numerical
training function was changed. An explicit final audit binds that run's outputs.
Training kernels are not guaranteed bitwise deterministic.

Evaluation
evaluate_controlled.py refuses to evaluate before all12 fits finish and verifies
all checkpoint, preprocessing and dataset identities, including cached results.
*_evaluated.npz stores per-reservoir single-step and recursive errors.
*_rollout_predictions.npz stores every recursive predicted state, including failures.
coarse_per_seed_realisation.csv and fine_per_seed_realisation.csv each contain1200 rows.
rollout_per_seed_realisation.csv contains120000 rows (4 models ×3 seeds ×100 reservoirs ×100 steps).
evaluation_summary.json separates seed SD from conditional reservoir bootstrap intervals.
timing.json uses prespecified seed42, not the best test-performing run.
Training and test environments use the same declared float32 forward architecture.

Reproduction in the project's Python environment
python -m unittest test_controlled -v
python controlled_training.py
python evaluate_controlled.py
python plot_controlled.py
python validate_controlled.py

Source/data/configuration changes are rejected in an existing experiment folder.
Use a new registered experiment directory for a changed protocol. RNG-state
resume is supported after a clean interrupt; hard-kill recovery requires an
explicit audit because the last artifact seal cannot be updated after a kill.
Do not rerun update_manuscript.py on an already-integrated manuscript; inspect
manuscript_changes.json and the timestamped manuscript backup for those edits.

Statistics
The same100 reservoirs are evaluated for each seed; this does not create300
independent reservoir units. Seed sample SD uses3 run-level test means. Bootstrap
intervals average these fixed runs within each reservoir and resample reservoir
IDs (10000 repetitions). They condition on the trained runs, not all possible
training initialisations. Nonfinite mean/CI cells are not estimated from survivors.
The adaptation bundles depth reduction and decoder replacement; no separate
causal attribution to one component is made.
