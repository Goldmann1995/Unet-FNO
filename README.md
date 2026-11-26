# UNet-FNO

Reservoir pressure and saturation prediction with FNO, U-FNO and UNet-FNO.
Includes the simulator, training pipeline, ablations and supplementary experiments.

Repository: https://github.com/Goldmann1995/Unet-FNO

## Installation

Use Python 3.10+ in a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU experiments require a PyTorch build suitable for your CUDA installation.
Dependencies are not pinned; historical versions remain in archived metadata.

## Repository structure

| Path | Purpose |
| --- | --- |
| `data_generation.py` | Reservoir simulator and HDF5 generation |
| `data_training.py` | UNet-FNO architecture and training |
| `fno.py`, `ufno.py` | FNO and U-FNO architectures |
| `run_all.py` | Main simulation, training and evaluation pipeline |
| `compare_models.py`, `ablation_study.py` | Comparison and ablations |
| `rollout.py` | Autoregressive evaluation |
| `data_visualization.py`, `plot_sample_permeability.py`, `Kr.py` | Visualization |
| `revision_experiments/` | Independent evaluation and controlled training |
| `results/revision_experiments/` | Archived JSON/CSV results and provenance |
| `archive/claude_fno.py` | Legacy prototype requiring unavailable `utilities3` |
| `source_manifest.json` | Original file hashes before release path adjustments |

## Main workflow

Run from the repository root. These commands perform substantial work:

```bash
python run_all.py --nx 20 --ny 20 --n_samples 500 --nt 500
python run_all.py --h5 path/to/dataset.h5
python ablation_study.py --h5 path/to/dataset.h5
python rollout.py --h5 path/to/dataset.h5 --ckpt_dir checkpoints --steps 100
```

Use `--help` to inspect settings. Evaluation-only runs require trained checkpoints.
Each pipeline has its own output/checkpoint locations; inspect options before reuse.
See [the supplementary guide](revision_experiments/GUIDE.md) for additional experiments.

## Tests

CPU tests require neither datasets nor checkpoints:

```bash
python -m unittest discover -s revision_experiments -p test_analysis.py -v
python -m unittest discover -s revision_experiments/low_resolution_baseline -p test_controlled.py -v
```

## Data and reproducibility

Datasets, weights, prediction arrays and temporary outputs are excluded from Git.
No download endpoint for data or weights is provided in this release. Exact
reproduction requires original inputs identified by the archived metadata;
newly generated datasets constitute new experiments.

Historical paths and hashes in result metadata are retained as provenance, not
runtime configuration. Do not copy archived training seals into a fresh run:
checks intentionally reject mismatched code, paths, data or artifacts.
Model algorithms are unchanged; trailing whitespace is normalized. Supplementary
path setup and optional plot auditing are adapted for distribution. The manifest
records both original and distributed file hashes.

Manuscripts, reviewer correspondence and document-editing automation remain in
the original authoring workspace. The existing [LICENSE](LICENSE) is retained.
