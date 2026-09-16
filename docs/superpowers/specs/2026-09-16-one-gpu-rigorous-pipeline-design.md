# One-GPU Rigorous Emotion-Alignment Pipeline Design

## Goal

Harden the MERGE audio-lyrics research pipeline so that it is methodologically defensible, resumable, numerically stable, and executable when only one GPU allocation is available. All expensive and administrative stages must run serially on the GPU partition; CPU work uses only the CPUs attached to that single GPU allocation.

## Evaluation protocol

Training, tuning, sweeps, optimizer comparisons, ablations, and seed confirmation may use only train and validation data. `scripts/train.py` must never construct a test dataset. The test split is opened only by `scripts/final_evaluate.py` after `scripts/select_finalists.py` freezes a validation-only selection manifest containing checkpoint SHA-256 hashes. Final evaluation must reject an unfrozen selection manifest or a checkpoint whose hash changed.

## Resume semantics

A stable `run_id` owns one run directory and checkpoint directory. `--resume auto` restores `last.pt`, including model, optimizer, scheduler, AMP scaler, RNG states, epoch, global step, best validation metric, early-stopping patience, CSPA state, and curvature-estimator state. A stable configuration fingerprint must reject incompatible resumes. Completed runs should become idempotent no-ops when resubmitted with `--resume auto`.

## Objectives and optimization

The required primary emotion objective remains multimodal cross-entropy plus weighted unimodal auxiliary cross-entropies. Symmetric audio-lyrics contrastive alignment remains the standard alignment objective. Affect-aware alignment may reweight negatives according to the four MERGE valence/arousal quadrants, but must use a log-sum-exp/cross-entropy formulation rather than direct exponentiation.

PCGrad and CAGrad are comparators, not claimed contributions. Gradient surgery is applied only to shared representation parameters; task-specific primary gradients are preserved. The alignment gradient supplied to these comparators is scaled by the configured `lambda_align`. The two-task CAGrad implementation must optimize the same simplex dual objective as the official exact formulation and be numerically checked against a SciPy reference in tests.

CSPA is the candidate methodological contribution: an adaptive alignment coefficient is capped by an auxiliary-gradient norm ratio and a beta-smooth primary-loss descent constraint. With a valid smoothness upper bound and a raw SGD update, the derivation is a conditional descent guarantee. With AdamW and the empirical secant curvature estimate used in practical experiments, CSPA is reported only as an empirical safeguard/diagnostic, never as a certified theorem.

## Feature extraction

MERT and RoBERTa are frozen feature backbones. MERT operates at 24 kHz and produces up to six 5-second segment representations per song. Pooling must respect the feature-vector attention mask for padded audio. RoBERTa uses up to eight 256-token chunks, masked mean pooling, no unused pooler layer, BF16 autocast on GPU, and float32 cached representations. Feature caches must contain finite values and deterministic ID mappings.

## Statistics

Validation and final-test outputs remain separate. Reporting includes mean, standard deviation, bootstrap 95% confidence intervals across seeds, paired seed-level comparisons against the fixed multimodal reference, paired prediction-level bootstrap Macro-F1 differences, Spearman alignment-performance relationships, lambda sensitivity, ablation comparisons, retrieval metrics, and alignment gap. Tables and figures are generated programmatically from run outputs.

## One-GPU execution

Every Slurm stage requests `--gres=gpu:1`. Array jobs use `%1`, ensuring at most one array element runs concurrently. The full pipeline is submitted as `afterok` dependencies: code/data preflight, MERT extraction, RoBERTa extraction, one-epoch smoke, Optuna tuning, required baselines, lambda sweep, lambda selection, optimizer comparison, ablations, five-seed confirmation, frozen validation selection, one-time test evaluation, then analysis. There is no dependence on a CPU partition.

## Novelty claim discipline

The repository must not claim that valence/arousal-aware contrastive learning, cross-attention, PCGrad, or CAGrad are novel. The candidate contribution is the primary-emotion-preserving adaptive cross-modal alignment controller and its analysis in the audio-lyrics setting. Any “first”, “novel”, or formal guarantee claim requires a dedicated literature audit and evidence matching the stated assumptions.

## Acceptance criteria

The implementation is accepted when: all Python sources compile under Python 3.9 grammar; all tests pass; all Slurm scripts pass `bash -n`; configs compose; the training entry point contains no test-set construction; final evaluation is frozen-manifest gated; resume state round-trips; affect-aware loss is finite under extreme logits; CAGrad matches the official exact two-task reference within numerical tolerance; all array jobs are `%1`; and a real A100 BF16 smoke verifies model forward/backward, MERT, RoBERTa, and actual MERGE decoding before the production chain.
