# Flow Matching & Rectified Flow (Reflow) on 2D Data

A compact PyTorch implementation of **Flow Matching** and **Rectified Flow / Reflow** (Liu et al., 2022),
showing how reflow straightens the ODE trajectories so samples can be generated with fewer integration steps.

## Method

1. **Phase 1 — Flow Matching**: train a velocity network `v(x, t)` on the straight-line path
   `x_t = t·x_1 + (1−t)·x_0` with target velocity `x_1 − x_0` (`core/fm_loss.py`).
2. **Phase 2 — Reflow**: integrate the trained model to build coupled pairs `(x_0, x_1)`,
   then retrain on these pairs (`phase2_reflow/`).
3. **Evaluation**: sample with an Euler ODE solver at 1, 5, 10 and 20 steps (`core/ode_solver.py`),
   plot samples and trajectories, and report mean **path straightness** (straight-line distance / arc length).

Datasets (`data/gaussian_8mode.py`): 8-mode Gaussian mixture, two moons, swiss roll, checkerboard.

## Usage

```bash
pip install -r requirements.txt
python test_core.py                         # sanity tests
python phase1_fm/train_fm.py                # checkpoints/fm_base.pt
python phase2_reflow/generate_pairs.py      # data/reflow_pairs.npy
python phase2_reflow/train_reflow.py        # checkpoints/reflow.pt
python infer.py                             # outputs/*.png + straightness comparison
```

Settings (dataset, model size, epochs, ODE steps) are in `config/hyperparameters.py`.
