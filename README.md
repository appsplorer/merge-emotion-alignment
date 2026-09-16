# MERGE Emotion Alignment

Research code for learning emotion-aligned representations from paired audio and lyrics on the MERGE Bimodal dataset.

## Project principles

- Official MERGE train/validation/test splits only.
- Validation data is used for model selection; test data is reserved for final evaluation.
- Large datasets, pretrained weights, cached features, checkpoints, and raw run directories stay on HPC and are excluded from Git.
- Every run stores its resolved configuration, seed, Git commit, environment metadata, metrics, and predictions.
- Final tables and figures are generated programmatically from run outputs.

## HPC paths

Project:
`/beegfs/general/sa25abo/research_workplace/Projects/merge-emotion-alignment`

Hugging Face cache:
`/beegfs/general/sa25abo/research_workplace/Models/huggingface_cache`
