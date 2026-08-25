# Model artifacts (provided at runtime)

These files are **not committed to git** — they are provided to the application
at runtime (mounted into this directory, or pointed to via the
`MODEL_ARTIFACTS_DIR` environment variable). The model refuses to start if any
are missing.

Place the following files in this directory (or in whatever directory
`MODEL_ARTIFACTS_DIR` points at):

| File | ~Size | What it is |
|------|------|------------|
| `model_M1.joblib` | 52 MB | Ensemble classifier M1 (pore limiting diameter > 2.4 Å) |
| `model_M2.joblib` | 27 MB | Ensemble classifier M2 (pore limiting diameter > 4.4 Å) |
| `model_M3.joblib` | 14 MB | Ensemble classifier M3 (pore limiting diameter > 5.9 Å) |
| `scaler.joblib` | 39 KB | Scaler for the mordred linker descriptors |
| `metal_scaler.joblib` | <1 KB | Scaler for the metal descriptors |
| `elemental_descriptors.csv` | 4 KB | Elemental descriptor table, indexed by element `Symbol` |

All artifacts were produced with the pinned training-time stack
(scikit-learn 1.2.0 — see `pyproject.toml`). They must match those versions.

## How they reach the running application

- **Local development:** place the files directly in this directory. Running
  from source resolves artifacts here by default.
- **Container / LMDS deployment:** the image does **not** bundle these files.
  Provide them at runtime by either
  - mounting a directory of artifacts and setting `MODEL_ARTIFACTS_DIR` to it
    (e.g. `-e MODEL_ARTIFACTS_DIR=/models -v /path/to/models:/models`), or
  - mounting them onto the default location the container expects (`/models`).
