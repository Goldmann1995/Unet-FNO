Supplementary analysis for reviewer comments 6, 9 and 10

No pretrained weights or original simulator sources were modified.
The experiment protocol is in protocol.txt. Runtime/checkpoint/source hashes
are in environment.json. Fixed source-file normalisation is in normalization.json.
New coarse-grid seeds/IDs and the complete evaluated fine-grid IDs are in
test_manifest.csv. All sample-level observations are retained, including failures.

Main outputs
independent_20x20_100.h5 — 100 new reference simulations, 500 stored states each.
rollout_per_realisation.csv — 30000 model/realisation/step error pairs.
rollout_mean_ci.csv — all 300 model/step means and 95% pointwise intervals.
*_rollout.npz — complete evaluated prediction/reference states and errors.
fine_grid_per_realisation.csv — 100 realisation summaries per model, 499 transitions each.
rollout_summary.json, fine_grid_summary.json — numerical aggregates used in the paper.
timing.json — every GPU repetition, matched trajectory wall time and paired speedup.
experiment_run.log — execution output.
validation.json — numerical and document consistency verification.

Reproduction
Use the project's existing Python environment and run from this folder.
python -m unittest test_analysis -v
python run_experiments.py --stage all
python plot_results.py
python validate_outputs.py

The data-generation stage resumes from saved per-realisation simulation files.
The evaluation stage requires CUDA and strictly loads the recorded checkpoints.
Do not rerun integrate_results.py against an already-integrated manuscript;
it consumes the pre-experiment manuscript preserved in the timestamped backup.
Use the recorded JSON changes to inspect the original replacements.

Interpretation
Bootstrap units are reservoir realisations, not time steps. Intervals reflect
test-field variability conditional on fixed checkpoints, not training repeats.
The new coarse set excludes all original 500 permeability fields by hash.
The fine grid is a separately computed, unpaired distribution-transfer dataset;
hash nonidentity across grid sizes does not establish independence of the
underlying continuous fields. Fixed cell-based smoothing changes physical scale.
The normalisation reproduces the entire original source-file preprocessing;
it excludes the new tests but does not retroactively repair historical leakage.
CPU/GPU timings compare specified implementations on specified processors and
are not hardware-independent or accuracy-equivalent acceleration factors.
