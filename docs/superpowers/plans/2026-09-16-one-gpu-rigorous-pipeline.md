# One-GPU Rigorous Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the MERGE experiment protocol test-firewalled, resumable, numerically stable, statistically complete, and serializable onto exactly one GPU.

**Architecture:** Keep the existing cached-feature multimodal architecture and strengthen the execution boundaries around it. Training becomes validation-only; final test access moves to a frozen-manifest evaluator; optimizer state and research-controller state become resumable; all Slurm stages request one GPU and form an `afterok` dependency chain.

**Tech Stack:** Python 3.9, PyTorch 2.x, Transformers, pandas, NumPy, SciPy, scikit-learn, Optuna, Slurm, pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-one-gpu-rigorous-pipeline-design.md`

## Global Constraints

- Production execution requires CUDA and exactly one Slurm GPU allocation per running stage.
- Train/tune/sweep/ablation code must never evaluate the test split.
- Test evaluation requires a frozen validation-only manifest and checkpoint hashes.
- Practical CSPA with AdamW + secant beta is empirical, not certified.
- Python syntax must remain compatible with Python 3.9.
- Arrays must be throttled with `%1`.

---

### Task 1: Test-set firewall and frozen final evaluation

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/evaluate.py`
- Create: `scripts/select_finalists.py`
- Create: `scripts/final_evaluate.py`
- Test: `tests/test_research_protocol.py`

**Interfaces:**
- Produces validation-only `results/runs/<run_id>/metrics.json` and `validation_predictions.csv`.
- Produces `results/manifests/final_selection.yaml` with checkpoint hashes.
- Final evaluator consumes that manifest and writes `results/final_test/<run_id>/`.

- [ ] Write a failing test that rejects test construction in `scripts/train.py` and requires the final evaluator to accept `--selection-manifest`.
- [ ] Run `PYTHONPATH=src pytest tests/test_research_protocol.py -q` and confirm the test fails for the old implementation.
- [ ] Remove all test construction/evaluation from training and implement frozen selection + explicit final evaluation.
- [ ] Re-run the focused test and confirm it passes.
- [ ] Run the full test suite.

### Task 2: Full resumable checkpoints

**Files:**
- Modify: `src/merge_emotion/engine/checkpoint.py`
- Modify: `src/merge_emotion/engine/trainer.py`
- Modify: `scripts/train.py`
- Test: `tests/test_checkpoint.py`
- Test: `tests/test_research_protocol.py`

**Interfaces:**
- `config_fingerprint(config) -> str`.
- Checkpoints store optimizer/scheduler/scaler/RNG plus `trainer_state`.
- `train(..., resume_checkpoint=Path|None)` resumes from the following epoch.

- [ ] Add a failing checkpoint round-trip test for `trainer_state` and configuration fingerprint.
- [ ] Run the focused test and observe the expected failure.
- [ ] Add checkpoint fingerprint/state persistence and restore logic.
- [ ] Add `--run-id` and `--resume` handling to the training entry point.
- [ ] Re-run focused and full tests.

### Task 3: Stable affect-aware contrastive objective

**Files:**
- Modify: `src/merge_emotion/objectives/affect_aware.py`
- Test: `tests/test_research_protocol.py`

**Interfaces:**
- `affect_aware_contrastive_loss(...) -> scalar Tensor` remains API-compatible.

- [ ] Add a failing extreme-logit test using a very small temperature.
- [ ] Verify the direct-exponential implementation fails or produces non-finite behavior.
- [ ] Replace manual exponentiation with weighted logits plus cross-entropy/log-sum-exp semantics.
- [ ] Confirm the loss and gradients remain finite.

### Task 4: Comparator correctness and CSPA claim discipline

**Files:**
- Modify: `src/merge_emotion/optim/multitask.py`
- Modify: `src/merge_emotion/optim/cspa.py`
- Modify: `src/merge_emotion/engine/trainer.py`
- Modify: `configs/pipeline/common.yaml`
- Test: `tests/test_cspa.py`
- Test: `tests/test_research_protocol.py`

**Interfaces:**
- `cagrad_two_task(g_primary, g_aux, c=0.5) -> Tensor`.
- `CSPAController.choose(...) -> CSPAState` logs feasible caps and directional margin.

- [ ] Add a failing test comparing two-task CAGrad against an exact SciPy simplex reference.
- [ ] Add a failing quadratic test for the conditional CSPA descent inequality.
- [ ] Implement device-local high-precision scalar optimization for the two-task CAGrad dual.
- [ ] Apply PCGrad/CAGrad only to shared representation parameters while preserving primary task-specific gradients and `lambda_align` scaling.
- [ ] Persist CSPA/curvature estimator state across resume.
- [ ] Document the conditional raw-SGD theorem scope and empirical AdamW scope.
- [ ] Run optimizer/CSPA focused tests and the full suite.

### Task 5: GPU-safe feature extraction

**Files:**
- Modify: `scripts/extract_audio_features.py`
- Modify: `scripts/extract_lyrics_features.py`
- Modify: `src/merge_emotion/data/dataset.py`

**Interfaces:**
- Audio cache: `cache/audio/mert_v1_95m.pt`.
- Lyrics cache: `cache/lyrics/roberta_base.pt`.

- [ ] Add CUDA-required execution flags and finite-value checks.
- [ ] Use MERT feature-vector masks for padded audio pooling.
- [ ] Disable the unused RoBERTa pooler and use masked mean chunk pooling.
- [ ] Store float32 cached representations and validated boolean masks.
- [ ] Validate cache schema during dataset loading.

### Task 6: Statistical reporting

**Files:**
- Modify: `src/merge_emotion/analysis/statistics.py`
- Modify: `src/merge_emotion/analysis/plotting.py`
- Modify: `scripts/aggregate_results.py`
- Modify: `scripts/make_tables.py`
- Modify: `scripts/make_figures.py`
- Create: `scripts/analyze_final_statistics.py`
- Test: `tests/test_research_protocol.py`

**Interfaces:**
- Validation summary: `results/summaries/all_runs.csv`.
- Final summary: `results/summaries/final_test_runs.csv`.
- Tables live under `results/tables/`; figures under `figures/`.

- [ ] Add a failing seed-summary test requiring bootstrap CI columns.
- [ ] Implement mean/SD/bootstrap CI, paired seed differences, Spearman relationships, and paired prediction-level Macro-F1 bootstrap.
- [ ] Generate lambda sensitivity, alignment-vs-emotion, Pareto, CSPA diagnostic, training-curve, and confusion-matrix outputs.
- [ ] Run focused and full tests.

### Task 7: Single-GPU serial Slurm pipeline

**Files:**
- Create: `slurm/00_code_preflight.sbatch`
- Modify: `slurm/00_smoke.sbatch`
- Modify: `slurm/01_audio_features.sbatch`
- Modify: `slurm/02_lyrics_features.sbatch`
- Modify: `slurm/03_tune.sbatch`
- Modify: `slurm/04_baselines.sbatch`
- Modify: `slurm/05_lambda_sweep.sbatch`
- Create: `slurm/05b_select_lambda.sbatch`
- Modify: `slurm/06_optimizer_comparison.sbatch`
- Modify: `slurm/07_ablations.sbatch`
- Modify: `slurm/08_final_seeds.sbatch`
- Create: `slurm/09_select_finalists.sbatch`
- Create: `slurm/10_final_evaluate.sbatch`
- Create: `slurm/11_analysis.sbatch`
- Create: `scripts/select_alignment.py`
- Create: `scripts/submit_one_gpu_pipeline.sh`
- Test: `tests/test_research_protocol.py`

**Interfaces:**
- All jobs request one GPU.
- Every array uses `%1`.
- Submission script links stages with `afterok` dependencies.

- [ ] Add a failing test that rejects any array lacking `%1`.
- [ ] Update every stage to request `--gres=gpu:1` and require CUDA for expensive code paths.
- [ ] Serialize arrays with `%1` and create deterministic run IDs with `--resume auto`.
- [ ] Add the one-command dependency-chain submitter.
- [ ] Run `bash -n` across all Slurm files and the submitter.

### Task 8: Final verification

**Files:**
- Modify: `docs/experiment_protocol.md`
- Create: `docs/novelty_positioning.md`

**Interfaces:**
- Documentation records the test firewall, theorem scope, prior-art positioning, and execution order.

- [ ] Run `PYTHONPATH=src pytest -q`.
- [ ] Run `python -m compileall -q src scripts tests`.
- [ ] Compose every pipeline experiment config.
- [ ] Run `bash -n slurm/*.sbatch` and `bash -n scripts/submit_one_gpu_pipeline.sh`.
- [ ] On UHHPC, run the one-GPU preflight and confirm A100 BF16, MERT, RoBERTa, and actual MERGE decoding before the production chain.
