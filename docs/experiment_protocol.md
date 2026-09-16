# Experiment Protocol

1. Use only official MERGE train/validation/test splits.
2. Select models and hyperparameters on validation macro-F1.
3. Do not inspect test metrics during model selection.
4. Save a fully resolved configuration for every run.
5. Save `last.pt` and `best.pt` checkpoints for training runs.
6. Record random seed, Git commit SHA, software versions, GPU, Slurm job ID, and wall-clock time.
7. Run the required audio-only, lyrics-only, multimodal-no-alignment, and standard-alignment baselines before proposed methods.
8. Use repeated seeds for final comparisons and report uncertainty.
9. Generate final tables and figures from stored run outputs.
