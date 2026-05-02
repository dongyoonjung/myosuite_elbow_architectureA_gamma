# MyoSuite myoElbow · Architecture A+γ

γ-motor extension of Architecture A. Policy outputs **18 dims**
= `(α, γs_target, γd_target) × 6`. γ_actual follows γ_target through a
1st-order low-pass (τ_γ = 100 ms) and modulates the spinal reflex
(stretch threshold via γs, Ia gain & reciprocal inhibition via γd).
Observation space is identical to A (23 dims), so the same backbone weights
transfer cleanly.

Spec: `~/.claude/skills/myosuite_elbow_architectureA_gamma/SKILL.md`.
Base env source: sibling repo `myosuite_elbow_architectureA`.

## Layout

```
myosuite_elbow_architectureA_gamma/
├── myosuite_elbow_arch_a_gamma/
│   ├── __init__.py        # registers MyoElbowReflexAGamma-v0
│   ├── env.py             # MyoElbowReflexEnvAGamma(MyoElbowReflexEnvA)
│   ├── reflex.py          # γ-modulated reflex
│   └── _path.py           # locates sibling A repo
├── scripts/
│   ├── _paths_agamma.py   # sys.path helper for both packages
│   ├── train_AGamma.py    # PPO trainer (supports --init-from / freeze)
│   ├── eval_AGamma.py     # deterministic evaluation
│   └── transfer_from_A.py # A → A+γ weight transfer (Stage 0)
├── tests/
│   ├── __init__.py
│   └── test_smoke_AGamma.py
├── configs/AGamma_default.yaml
├── requirements.txt
└── .gitignore
```

## A package resolution

`myosuite_elbow_arch_a_gamma` imports from `myosuite_elbow_arch_a`. Resolution
order (both the package and the script helper use the same order):

1. `MYOSUITE_ELBOW_A_PATH` env var (path to A repo root).
2. Sibling directory `../myosuite_elbow_architectureA`.
3. Same-level directory `./myosuite_elbow_architectureA`.
4. `../../myosuite_elbow_architectureA` (deeper invocation).
5. `/root/myosuite_elbow_architectureA` (VESSL container default).

If none resolves the helper raises with the list of attempted paths instead
of failing later with a confusing `ModuleNotFoundError`.

On VESSL containers the recommended layout puts both repos under `/root/`:

```
/root/myosuite_elbow_architectureA/
/root/myosuite_elbow_architectureA_gamma/
```

so the sibling-directory probe resolves automatically.

## Local development (WSL, conda env `rl_myosuite`, Python 3.11)

```bash
cd ~/projects/myosuite_elbowrl/myosuite_elbow_architectureA_gamma

# Smoke test (action shape, NaN scan, γ low-pass convergence, k=0 baseline)
python tests/test_smoke_AGamma.py

# Short sanity training (CPU-only on local)
python scripts/train_AGamma.py \
    --timesteps 50000 \
    --n-envs 4 \
    --vec-env subproc \
    --device cpu \
    --tag AGamma_local_short

# Evaluate
python scripts/eval_AGamma.py \
    --model models/AGamma_local_short/ppo_AGamma_local_short_final.zip \
    --vecnorm models/AGamma_local_short/vec_normalize_AGamma_local_short.pkl \
    --episodes 20
```

## VESSL workflow (SNU 공대 `snu-eng-dgx-light`)

This project is run on VESSL via local-edit → push → pull-on-container.
See `~/.claude/skills/vessl-snu-workflow/SKILL.md` for full details. Quick
recap of the steps that involve this repo:

### One-time: create the GitHub repo

On WSL:
```bash
cd ~/projects/myosuite_elbowrl/myosuite_elbow_architectureA_gamma
git init
git config user.name "dongyoonjung"
git config user.email "aaronjdy11@gmail.com"
git add .
git commit -m "Initial commit: A+γ scaffold"
git branch -M main
# Create empty private repo "myosuite_elbow_architectureA_gamma" on github.com first.
git remote add origin https://github.com/dongyoonjung/myosuite_elbow_architectureA_gamma.git
git push -u origin main
```

### Routine: edit locally, push, pull on container

```bash
# Local (WSL):
git add scripts/train_AGamma.py
git commit -m "tweak γ-init"
git push origin main

# Container (after SSH in, port from VESSL UI metadata page):
cd /root/myosuite_elbow_architectureA_gamma
git pull origin main
```

If this is the first time on the container, clone both A and A+γ side by side:
```bash
cd /root
git clone https://github.com/dongyoonjung/myosuite_elbow_architectureA.git
git clone https://github.com/dongyoonjung/myosuite_elbow_architectureA_gamma.git
```

### One-time container env setup (per workspace lifetime, persists in `/root/`)

Skill reference: `vessl-snu-workflow/references/env-setup.md`.

```bash
# Create persistent venv at /root/venv
python -m venv /root/venv
mkdir -p /root/.cache/pip
echo 'export PIP_CACHE_DIR=/root/.cache/pip' >> /root/.bashrc
echo 'source /root/venv/bin/activate' >> /root/.bashrc
# Headless MuJoCo: skip GLFW/X11 fallbacks.
echo 'export MUJOCO_GL=egl' >> /root/.bashrc
source /root/.bashrc

# Install the stack
pip install -r /root/myosuite_elbow_architectureA_gamma/requirements.txt
```

### Run training on the cluster

```bash
cd /root/myosuite_elbow_architectureA_gamma

# Make sure A is reachable. The default sibling probe finds it because both repos
# live under /root/. To be explicit:
export MYOSUITE_ELBOW_A_PATH=/root/myosuite_elbow_architectureA

# A100 MIG, MuJoCo CPU collection bottleneck → n_envs is the bigger lever.
python scripts/train_AGamma.py \
    --timesteps 2000000 \
    --n-envs 16 \
    --vec-env subproc \
    --device auto \
    --tag AGamma_v1 \
    --k-th 0.10 --k-v 1.0 --tau-gamma 0.100 --gamma-init 0.5 \
    | tee logs/train_AGamma_v1.out
```

`--vec-env subproc --n-envs 16` is the proven sweet spot on the cluster's
A100 MIG instance (see `vessl-snu-workflow/references/known-gotchas.md` —
`DummyVecEnv` is single-process, so larger `n_envs` only helps with subproc).

`--device auto` lets SB3 pick CUDA if available. Effect of GPU on this small
MLP is modest because collection (MuJoCo) is CPU-bound; for `n_envs ≤ 8`
running with `--device cpu` is often faster end-to-end.

The `logs/` directory is created automatically by the trainer, so `tee` works
on a fresh clone.

### A → A+γ warm-start (Stages 0–2 from SKILL.md §9)

Stage 0 — produce a seeded A+γ model from a trained A model:

```bash
# Same command works locally or on the container.
python scripts/transfer_from_A.py \
    --source-model    /root/myosuite_elbow_architectureA/models/A_v8/ppo_A_v8_final.zip \
    --source-vecnorm  /root/myosuite_elbow_architectureA/models/A_v8/vec_normalize_A_v8.pkl \
    --out-model       models/AGamma_seed/ppo_AGamma_seed.zip \
    --out-vecnorm     models/AGamma_seed/vec_normalize_AGamma_seed.pkl
```

The transfer script:
- Copies every backbone tensor whose shape matches A→A+γ exactly.
- Pads `action_net` (rows 0:6 ← A, rows 6:18 ← N(0, 0.01) / bias 0 → γ_target ≈ 0.5).
- Sets `log_std[6:18] = -1.5` (smaller std on γ for safer initial exploration).
- Resets `VecNormalize.ret_rms` (A+γ effort uses post-reflex muscle input,
  so reward magnitudes differ; pass `--keep-ret-rms` to skip).
- **Raises** on any unexpected shape mismatch (e.g. if A was trained with a
  different `policy_kwargs.net_arch`) instead of silently doing a partial copy.

Stage 1 — γ-adaptation (backbone frozen, only γ heads + log_std trainable):

```bash
python scripts/train_AGamma.py \
    --init-from   models/AGamma_seed/ppo_AGamma_seed.zip \
    --init-vecnorm models/AGamma_seed/vec_normalize_AGamma_seed.pkl \
    --freeze-backbone-steps 100000 \
    --timesteps 100000 \
    --n-envs 16 --vec-env subproc --device auto \
    --tag AGamma_v1_stage1
```

`--freeze-backbone-steps N` keeps everything except `action_net.weight`,
`action_net.bias`, and `log_std` frozen for the first N steps, then unfreezes
the rest. Setting `N == --timesteps` runs Stage 1 only.

Stage 2 — full fine-tune from the Stage 1 checkpoint (lower LR per spec):

```bash
python scripts/train_AGamma.py \
    --init-from    models/AGamma_v1_stage1/ppo_AGamma_v1_stage1_final.zip \
    --init-vecnorm models/AGamma_v1_stage1/vec_normalize_AGamma_v1_stage1.pkl \
    --timesteps 1000000 \
    --lr 1e-4 \
    --n-envs 16 --vec-env subproc --device auto \
    --tag AGamma_v1_stage2
```

If you want to skip Stage 1 entirely and go straight to full fine-tune from
the seed model, drop `--freeze-backbone-steps`.

## Key hyperparameters worth sweeping

From the spec (§5.5 sanity table, §10.1 ablation grid):

| Variable | Suggested grid | Why |
|---|---|---|
| `--k-th` | 0.0 / 0.05 / 0.10 / 0.20 | γs leverage on threshold |
| `--k-v` | 0.0 / 0.5 / 1.0 / 2.0 | γd leverage on Ia gain |
| `--tau-gamma` | 0.05 / 0.10 / 0.20 | γ low-pass time constant |
| `--gamma-init` | 0.0 / 0.5 | initial γ at reset |
| env `gamma` (effort) | 0.03 / 0.05 | post-reflex muscle_input is bigger than α |

## Notes

- This package never modifies anything in the A repo.
- Use the `rl_myosuite` conda env locally (Python 3.11). On the container the
  venv is at `/root/venv` (Python 3.10) — both work; the requirements.txt
  pin only the four core RL stack libraries to keep this consistent.
- The Architecture A run history (TensorBoard, model zips) lives next to A's
  repo — not duplicated here.
- Re-running training with the same `--tag` appends to the existing
  `logs/reward_<tag>.tsv` rather than truncating it (the SB3 TensorBoard
  writer continues to auto-suffix run directories with `_1, _2, …`).
