# MOF Porosity

Machine-learning prediction of metal-organic framework (MOF) porosity, packaged
for the LMDS (Local Model Deployment Service) platform.

Given a linker (as a SMILES string) and a metal, an ensemble of three binary
classifiers predicts whether the pore limiting diameter exceeds 2.4 Å, 4.4 Å and
5.9 Å, and combines those into an overall porosity category. The underlying
model is described in
[*Machine-Learning Prediction of Metal–Organic Framework Guest Accessibility from Linker and Metal Chemistry*](https://onlinelibrary.wiley.com/doi/10.1002/anie.202114573)
and was reported for the LMDS in
[*The Liverpool Materials Discovery Server*](https://doi.org/10.1039/D3DD00093A).
Predictions were ~80.5% accurate on the held-out test set.

This is a conversion of the original Flask app to the LMDS model-template
contract (`schema/model.json` + a single `process` handler). The web/REST layer
is gone; the tool now runs Mode-driven under the shared runner.

## Quick Start

```bash
# Instance mode — pass linker(s) and metal(s) directly (no Docker, no S3):
scripts/run_instance.sh "O=C(O)c1ccc(cc1)C(=O)O" "Zn,Cu"

# Dataset mode — batch-predict from a CSV of linker,metal pairs:
scripts/run_dataset.sh data/input.csv
# -> writes data/output/predictions.csv
```

## Modes

| Mode | Response | Input | Output |
|------|----------|-------|--------|
| `instance` | immediate | `linkers` (array of SMILES), `metals` (array of symbols) | `predictions` (array of objects), `count` |
| `dataset` | deferred | `file` (CSV with `linker`,`metal` columns) | `predictions.csv` asset |

**Broadcasting** (instance mode) mirrors the original tool: pass many linkers
with one metal, one linker with many metals, or two equal-length lists paired in
order.

Each prediction object contains:

| Field | Meaning |
|-------|---------|
| `linker`, `metal` | the pair the prediction was made for |
| `pld_gt_2_4A` | model M1: pore limiting diameter > 2.4 Å |
| `pld_gt_4_4A` | model M2: pore limiting diameter > 4.4 Å |
| `pld_gt_5_9A` | model M3: pore limiting diameter > 5.9 Å |
| `category` | overall porosity category from the ensemble |

Only some elements are supported as the metal — see
[`src/model/artifacts/elemental_descriptors.csv`](src/model/artifacts/elemental_descriptors.csv).

## Project Structure

```
├── schema/
│   ├── model.json              # This tool's contract (instance + dataset modes)
│   └── model.schema.json       # Shared meta-schema (copied from template)
├── src/
│   ├── cmds/                   # Entrypoints — copied from template, unmodified
│   ├── infrastructure/         # Runner/handler/S3/logging — copied, unmodified
│   └── model/
│       ├── __init__.py         # handler = MOFPorosityModel()
│       ├── model.py            # Prediction logic (descriptors, scaling, ensemble)
│       ├── types.py            # Auto-generated from model.json
│       └── artifacts/          # Model files — provided at runtime, NOT in git
│                               #   (only .gitkeep + README.md tracked)
├── data/                       # Local sample inputs + run outputs (output/ ignored)
├── scripts/                    # run_instance.sh, run_dataset.sh
├── tests/test_model.py
├── Dockerfile
└── pyproject.toml
```

## Python version and pinned dependencies

> **This tool runs on Python 3.11, not 3.12.**

The three model files were serialised with **scikit-learn 1.2.0**. scikit-learn
changed its decision-tree pickle format in 1.3 (adding the `missing_go_to_left`
node field), so versions ≥ 1.3 cannot load these artifacts, and 1.2.x ships no
Python 3.12 wheels. The scientific stack is therefore pinned to the model's
training-time versions (`scikit-learn==1.2.0`, `numpy<2`, `pandas<2.2`) to
guarantee predictions identical to the published model, and `requires-python` is
`>=3.11,<3.12`. The template's Dockerfile already supports this via
`PYTHON_VERSION` (defaulted to `3.11` here).

To move to a newer stack, re-save the ensemble under the target scikit-learn
version and drop the pins.

`rdkit` and `mordredcommunity` (a maintained fork of the original, unmaintained
`mordred`) supply the linker descriptors.

## Model artifacts (runtime assets)

The ~90 MB of model files (the three ensemble classifiers, the two scalers and
the elemental descriptor table) are **not committed to git** and **not baked
into the image**. They are declared as `assets` in `schema/model.json` and
provided by the LMDS platform at run time:

1. An admin uploads a file for each declared asset path under the version's
   **Manage → Versions → Assets** page (only schema-declared paths are accepted).
2. On each run, the runner downloads the assets from S3 to their declared paths
   under the container WORKDIR (`./artifacts/…`) before the handler executes.

The model reads them from `./artifacts/` (the package's `artifacts/` dir is the
fallback for a local source checkout; `MODEL_ARTIFACTS_DIR` overrides both), and
fails fast with a clear message if any required file is missing. See
[`src/model/artifacts/README.md`](src/model/artifacts/README.md) for the manifest.

**Local development:** put the files in `src/model/artifacts/`; everything runs
as normal.

## Local development

```bash
uv sync --extra dev                                   # create the 3.11 venv
uv run validate-model schema/model.json               # validate the contract
uv run model-types schema/model.json -o src/model/types.py
uv run pytest tests/ -v                               # run tests
```

Direct invocation (what the scripts wrap):

```bash
uv run model-run --local --parameters '{"mode":"instance","inputs":{"linkers":{"value":["O=C(O)c1ccc(cc1)C(=O)O"]},"metals":{"value":["Zn"]}},"parameters":{}}'
```

## Docker

```bash
# 3.11 is the default build arg; the model requires it.
docker build -t mof-porosity:1.0.0 .

# On the platform the runner downloads the declared assets to ./artifacts before
# running. To run the image by hand, put the model files where the runner would
# — mount them at /app/artifacts — and the model reads them from there:
docker run --rm -v /path/to/artifacts:/app/artifacts:ro mof-porosity:1.0.0 --local \
  --parameters '{"mode":"instance","inputs":{"linkers":{"value":["CCO"]},"metals":{"value":["Fe"]}},"parameters":{}}'
```

## Environment Variables

Copy `.env.example` to `.env` and adjust for remote (S3) runs:

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `DEBUG` | Logging verbosity |
| `S3_ENDPOINT` | `http://minio:9000` | S3-compatible storage endpoint |
| `S3_ACCESS_KEY` | `minioadmin` | S3 access key |
| `S3_SECRET_KEY` | `minioadmin` | S3 secret key |
| `S3_BUCKET` | `lmds` | S3 bucket name |
