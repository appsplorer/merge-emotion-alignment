# Experiment protocol

## Test-set firewall

`scripts/train.py`, hyperparameter tuning, lambda sweeps, optimizer comparisons, and ablations are train/validation only. They never instantiate the test split. Development results are written as validation metrics and validation predictions.

After all hyperparameters, methods, and final seeds are fixed using validation only, `scripts/select_finalists.py` creates `results/manifests/final_selection.yaml`. It records the selected run IDs and SHA-256 hashes of their best checkpoints. Only `scripts/final_evaluate.py` is allowed to instantiate the test split, and it refuses to run unless the selection manifest is frozen and the checkpoint hashes match.

## Hyperparameter order

1. Cache frozen MERT and RoBERTa representations.
2. Tune shared learning rate, weight decay, dropout, and unimodal auxiliary weight on validation using the no-alignment multimodal model.
3. Run preliminary required baselines.
4. Sweep fixed alignment weight on validation only.
5. Freeze the selected fixed alignment weight.
6. Compare scalarization, PCGrad, CAGrad, CSPA, and affect-aware variants on validation.
7. Run five predefined seeds for all final methods.
8. Freeze the final run manifest.
9. Evaluate the frozen runs on the held-out test set exactly once.
10. Generate statistics, tables, and figures from the frozen outputs.

## One-GPU execution

All compute stages are submitted to the GPU partition with `--gres=gpu:1`. Slurm arrays are throttled with `%1`, so at most one experiment task runs at any time. CPU work such as data loading, pytest, CSV aggregation, and statistics uses CPUs attached to that same GPU allocation; the pipeline does not require a separate CPU partition.

## Resume

Production run IDs are deterministic. `--resume auto` restores model, optimizer, scheduler, AMP scaler, Python/NumPy/PyTorch/CUDA RNG states, epoch, global step, best validation metric, early-stopping patience, CSPA EMA state, and secant-curvature state from `last.pt`. A configuration fingerprint rejects incompatible resumes.

## Statistical reporting

Final reporting includes per-method mean, standard deviation, and bootstrap 95% confidence intervals across five seeds; paired seed-level differences; paired prediction-level bootstrap differences against the fixed-alignment multimodal reference; lambda sensitivity; Spearman alignment-vs-emotion association; confusion matrices; and CSPA optimization diagnostics.
