# Supplementary experiments

Run from this directory. Core models are resolved relative to the repository;
set `UNET_FNO_ROOT` only to override the model/data root.

## Inputs

Required root-relative files:

- `reservoir_fno_dataset_20x20/reservoir_fno_20260319_193856_samples_500.h5`
- `reservoir_fno_dataset_40x40/reservoir_fno_20260322_175636_samples_100.h5`
- Frozen evaluation: `checkpoints/best_fno.pth`, `checkpoints/best_u_fno.pth`,
  `checkpoints/best_unet_fno.pth`.

The independent generation stage creates `independent_20x20_100.h5` here.
Inputs are not bundled. See `protocol.txt` and `low_resolution_baseline/design.txt`.
Supplementary training and evaluation require CUDA.

## Commands

```bash
python run_experiments.py --help
python run_experiments.py --stage all
python low_resolution_baseline/controlled_training.py --help
python low_resolution_baseline/controlled_training.py
python low_resolution_baseline/evaluate_controlled.py
python train_v5_paired.py
python train_width_matched_fno.py
python train_two_step_fno.py
python channelized_test.py
```

Inspect the protocols before expensive runs. Diagnostics require generated arrays,
not only the archived JSON/CSV summaries.

## Results and plotting

Historical JSON/CSV outputs are under `../results/revision_experiments/` with their
original relative hierarchy. Fresh runs write here, avoiding automatic reuse of
old normalization, completion markers or provenance seals.

For archived plots, copy `plot_results.py` to `../results/revision_experiments/`
and `low_resolution_baseline/plot_controlled.py` to the matching archive
subdirectory, then run those copies. Figures go under `../results/fig/`.
Fresh-run plotting uses the scripts here. The optional `audit_panel_alignment`
authoring utility is not bundled: a warning reports when that audit is skipped.

Historical README.txt files describe the original authoring workspace, including
some document-editing/validation scripts excluded from this code release.
This guide describes the distributed layout.
