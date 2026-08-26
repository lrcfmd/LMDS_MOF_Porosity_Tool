# Model artifacts (runtime assets)

These files are **not committed to git**. They are declared as `assets` in
[`schema/model.json`](../../../schema/model.json) and provided to the tool at
run time by the LMDS platform:

1. An admin uploads a file for each declared asset path under the version's
   **Manage → Versions → Assets** page (the upload UI only accepts paths that
   are declared in the schema).
2. When a run starts, the runner downloads each asset from S3 to its declared
   path under the container WORKDIR (`./artifacts/…`), before the handler runs.

The model reads them from `./artifacts/` (or the package's `artifacts/` dir for
a local source checkout; `MODEL_ARTIFACTS_DIR` overrides both). It fails fast if
any required file is missing.

| Asset path (declared in model.json) | ~Size | What it is |
|-------------------------------------|------|------------|
| `artifacts/model_M1.joblib` | 52 MB | Ensemble classifier M1 (PLD > 2.4 Å) |
| `artifacts/model_M2.joblib` | 27 MB | Ensemble classifier M2 (PLD > 4.4 Å) |
| `artifacts/model_M3.joblib` | 14 MB | Ensemble classifier M3 (PLD > 5.9 Å) |
| `artifacts/scaler.joblib` | 39 KB | Scaler for the mordred linker descriptors |
| `artifacts/metal_scaler.joblib` | <1 KB | Scaler for the metal descriptors |
| `artifacts/elemental_descriptors.csv` | 4 KB | Elemental descriptor table, indexed by `Symbol` |

All artifacts were produced with the pinned training-time stack
(scikit-learn 1.2.0 — see `pyproject.toml`) and must match those versions.

**Local development:** place these files in this directory
(`src/model/artifacts/`); running from source resolves them here by default.
