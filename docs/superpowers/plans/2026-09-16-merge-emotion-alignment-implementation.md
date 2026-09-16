# MERGE Emotion Alignment HPC/GitHub Implementation Plan

**Goal:** Build a reproducible HPC/GitHub research repository for the MERGE audio-lyrics emotion-alignment assignment with configuration-driven experiments, checkpoint/resume, automatic evaluation, statistical analysis, and publication-quality figures.

**Architecture:** One project directory on HPC is the Git working tree. Code, configurations, tests, summaries, tables, figures, and report sources are tracked; datasets, feature caches, checkpoints, logs, Optuna storage, pretrained model files, and raw run outputs remain HPC-only.

## Tasks

1. Scaffold repository and establish Git synchronization.
2. Add verified MERGE acquisition with checksum validation.
3. Inspect archive and implement official split loader with leakage checks.
4. Implement MERT and lyrics feature-cache pipelines.
5. Implement required audio-only, lyrics-only, multimodal-no-alignment, and standard-alignment baselines.
6. Implement checkpoint/resume and provenance metadata.
7. Implement alignment/retrieval/classification metrics.
8. Run controlled lambda/temperature/unimodal-loss sensitivity studies.
9. Implement optimization baselines and proposed affect-aware/CSPA methods with ablations.
10. Add multi-seed statistics, automatic tables/figures, compute accounting, and clean-clone reproducibility checks.

## Global Constraints

- Project root: `/beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment`
- Hugging Face cache: `/beegfs/general/sa25abo/research_workplace/Models/huggingface_cache`
- Git remote: `git@github.com:appsplorer/merge-emotion-alignment.git`
- Main dataset: MERGE Bimodal Complete v1.1.0, Zenodo record 13939205.
- Main split: official 70/15/15 train/validation/test.
- Primary model-selection metric: validation macro-F1.
- Test metrics are not used for model selection.
- Every real training run saves resolved config, metadata, `last.pt`, and `best.pt`.
- Large artifacts are excluded from Git.
