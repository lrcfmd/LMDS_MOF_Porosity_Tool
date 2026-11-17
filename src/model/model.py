import csv
import os
from pathlib import Path

import joblib
import pandas as pd
from mordred import Calculator, descriptors as mordred_descriptors
from rdkit import Chem

from infrastructure.handler import InputError
from infrastructure.logging import newLogger


# Model artifacts are NOT committed to git; they are declared as `assets` in
# schema/model.json and provided at runtime. On the platform the runner
# downloads them to their declared paths under the container WORKDIR
# (./artifacts/…), so that is the primary location. For a local source checkout
# the files live in the package's own artifacts/ dir. MODEL_ARTIFACTS_DIR
# overrides both. See artifacts/README.md.
def _artifacts_dir() -> Path:
    env = os.environ.get("MODEL_ARTIFACTS_DIR")
    if env:
        return Path(env)
    workdir_assets = Path("artifacts")  # where the runner downloads assets
    if workdir_assets.is_dir():
        return workdir_assets
    return Path(__file__).resolve().parent / "artifacts"


# Files that must be present in the artifacts directory at runtime.
REQUIRED_ARTIFACTS = (
    "elemental_descriptors.csv",
    "scaler.joblib",
    "metal_scaler.joblib",
    "model_M1.joblib",
    "model_M2.joblib",
    "model_M3.joblib",
)

# The 50 mordred descriptors the linker models were trained on, in order.
LINKER_DESCRIPTORS = [
    "ABC", "SpAbs_A", "SpAD_A", "nAromAtom", "nAromBond", "nC", "ATS1dv",
    "VR2_DzZ", "VR2_Dzm", "VR2_Dzv", "VR2_Dzse", "VR2_Dzpe", "VR2_Dzare",
    "VR2_Dzp", "VR2_Dzi", "BertzCT", "nBondsO", "nBondsA", "nBondsM",
    "nBondsKD", "C3SP2", "Xp-1d", "Xp-3d", "Xp-4d", "Xp-5d", "Xp-6d",
    "Xp-7d", "VR2_Dt", "VR2_D", "NaaCH", "NaasC", "ETA_beta", "ETA_beta_ns",
    "ETA_eta_RL", "ETA_eta_FL", "PEOE_VSA7", "SMR_VSA7", "SlogP_VSA6",
    "MID", "MID_C", "MPC2", "MPC3", "TpiPC10", "nRing", "n6Ring", "naRing",
    "n6aRing", "MWC01", "Zagreb1", "Zagreb2",
]

# Metal descriptor columns pulled from elemental_descriptors.csv, in order.
METAL_DESCRIPTORS = [
    "Atomic_Number", "Atomic_Weight", "Atomic Radius", "Mulliken EN",
    "polarizability(A^3)", "electron affinity(kJ/mol)",
]

# Most items listed in one error message.
_MAX_LISTED = 5


def _listing(items: list[str]) -> str:
    shown = "; ".join(items[:_MAX_LISTED])
    if len(items) > _MAX_LISTED:
        shown += f"; and {len(items) - _MAX_LISTED} more"
    return shown


def smiles_problem(smiles: str) -> str | None:
    """Why ``smiles`` can't be used as a linker, or None if it parses."""
    if Chem.MolFromSmiles(smiles) is not None:
        return None
    if Chem.MolFromSmiles(smiles, sanitize=False) is not None:
        return "chemically invalid, e.g. an atom with too many bonds or an aromatic ring that cannot be kekulised"
    return "not valid SMILES syntax"


def read_pairs_csv(path: Path) -> list[tuple[int, str, str]]:
    """Read a ``linker,metal`` CSV into ``(row number, linker, metal)`` tuples.

    Row numbers count the header as row 1 (as a spreadsheet shows them). Fully
    blank rows are skipped; anything else unusable raises InputError.
    """
    try:
        # utf-8-sig: tolerate the byte-order mark Excel writes.
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = [(n or "").strip() for n in (reader.fieldnames or [])]
            if not any(fieldnames):
                raise InputError(
                    "The CSV file is empty. It needs a header row 'linker,metal' "
                    "followed by one row per MOF."
                )
            missing = [c for c in ("linker", "metal") if c not in fieldnames]
            if missing:
                raise InputError(
                    "The CSV must have 'linker' and 'metal' columns (comma-"
                    f"separated, named in the first row); missing: "
                    f"{', '.join(repr(c) for c in missing)}. Found columns: "
                    f"{', '.join(repr(c) for c in fieldnames)}."
                )
            reader.fieldnames = fieldnames
            rows, half_filled = [], []
            for row in reader:
                line = reader.line_num
                linker = (row.get("linker") or "").strip()
                metal = (row.get("metal") or "").strip()
                if not linker and not metal:
                    continue  # blank row
                if not linker or not metal:
                    empty = "linker" if not linker else "metal"
                    half_filled.append(f"row {line} has no {empty}")
                    continue
                rows.append((line, linker, metal))
    except UnicodeDecodeError:
        raise InputError(
            "The CSV file could not be read as text. Please save it as a "
            "UTF-8 CSV file."
        ) from None
    except csv.Error as e:
        raise InputError(f"The CSV file could not be parsed ({e}).") from None

    if half_filled:
        raise InputError(
            f"Every row needs both a linker and a metal: {_listing(half_filled)}."
        )
    if not rows:
        raise InputError(
            "No linker/metal rows found in input file: the CSV has a header "
            "but no data rows."
        )
    return rows


class MOFPorosityModel:
    """Predict metal-organic framework (MOF) porosity from linker + metal chemistry.

    An ensemble of three binary classifiers (M1/M2/M3) each predicts whether the
    pore limiting diameter exceeds a threshold (2.4 Å, 4.4 Å, 5.9 Å). Their
    combination gives an overall porosity category. Ported from the LMDS Flask
    app; the published model is described in doi:10.1002/anie.202114573.

    Two Modes share one prediction path:
      ``instance`` — linker/metal values in, predictions returned inline.
      ``dataset``  — a CSV of linker,metal pairs in, a predictions CSV out.
    """

    def __init__(self):
        self.logger = newLogger("model")
        self._loaded = False
        self.elements = None

    def _elements(self) -> pd.DataFrame:
        """The elemental descriptor table (small; loaded on its own so metals
        can be checked before the heavy models are loaded)."""
        if getattr(self, "elements", None) is None:
            path = _artifacts_dir() / "elemental_descriptors.csv"
            if not path.is_file():
                self._load()  # raises the missing-artifacts error
            self.elements = (
                pd.read_csv(path, index_col="Symbol").drop("Name", axis=1)
            )
        return self.elements

    def supported_metals(self) -> list[str]:
        """Element symbols with a complete set of metal descriptors."""
        table = self._elements()
        complete = table[METAL_DESCRIPTORS].notna().all(axis=1)
        return [str(s) for s in table.index[complete]]

    def check_inputs(
        self, linkers: list[str], metals: list[str],
        linker_labels: list[str] | None = None,
        metal_labels: list[str] | None = None,
    ) -> None:
        """Raise InputError if any linker SMILES or metal symbol is unusable.

        ``*_labels`` say where each value came from in the message (e.g.
        "row 3"); they default to "linker 1", "metal 2", ...
        """
        linker_labels = linker_labels or [f"linker {i + 1}" for i in range(len(linkers))]
        metal_labels = metal_labels or [f"metal {i + 1}" for i in range(len(metals))]

        bad_smiles = []
        for label, smi in zip(linker_labels, linkers):
            problem = smiles_problem(smi)
            if problem:
                bad_smiles.append(f"{label} {smi!r} is {problem}")
        if bad_smiles:
            raise InputError(f"Invalid SMILES for the linker: {_listing(bad_smiles)}.")

        table = self._elements()
        supported = set(self.supported_metals())
        lookup = {s.lower(): s for s in supported}
        bad_metals = []
        for label, metal in zip(metal_labels, metals):
            if metal in supported:
                continue
            if metal in table.index:
                why = "has no descriptor data in the model, so it cannot be used"
            elif metal.lower() in lookup:
                why = f"is not a valid element symbol (did you mean {lookup[metal.lower()]!r}?)"
            else:
                why = "is not a recognised element symbol (give the symbol only, e.g. Zn, with no charge or name)"
            bad_metals.append(f"{label} {metal!r} {why}")
        if bad_metals:
            raise InputError(f"Unknown metal: {_listing(bad_metals)}.")

    def _load(self) -> None:
        """Lazily load featurisers, scalers and models (heavy; done once)."""
        if self._loaded:
            return
        artifacts = _artifacts_dir()
        self.logger.info("loading artifacts", dir=str(artifacts))

        missing = [f for f in REQUIRED_ARTIFACTS if not (artifacts / f).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Model artifacts missing from {artifacts}: {', '.join(missing)}. "
                "These files are provided to the application at runtime and are "
                "not committed to git — see src/model/artifacts/README.md. Set "
                "MODEL_ARTIFACTS_DIR to the directory that holds them, or mount "
                "them into the default location."
            )

        self.calc = Calculator(mordred_descriptors, ignore_3D=True)
        self._elements()
        self.scaler = joblib.load(artifacts / "scaler.joblib")
        self.metal_scaler = joblib.load(artifacts / "metal_scaler.joblib")
        self.m1 = joblib.load(artifacts / "model_M1.joblib")
        self.m2 = joblib.load(artifacts / "model_M2.joblib")
        self.m3 = joblib.load(artifacts / "model_M3.joblib")
        self._loaded = True

    @staticmethod
    def _category(m1: bool, m2: bool, m3: bool) -> str:
        """Map the three ensemble predictions to an overall porosity category."""
        if not m1:
            return "porosity < 2.4 Å"
        elif not m2:
            return "2.4 Å < porosity < 4.4 Å"
        elif not m3:
            return "4.4 Å < porosity < 5.9 Å"
        else:
            return "porosity > 5.9 Å"

    def predict(
        self, linkers: list[str], metals: list[str], checked: bool = False
    ) -> list[dict]:
        """Core prediction path shared by both Modes.

        Broadcasting mirrors the original tool: a single metal is applied to all
        linkers, a single linker is applied to all metals, otherwise the two
        lists are paired in order (and must be equal length).

        Inputs are checked (InputError) before the models are loaded; pass
        ``checked=True`` if :meth:`check_inputs` has already been run.
        """
        if not linkers or not metals:
            raise InputError("Both 'linkers' and 'metals' must be provided.")
        if len(metals) != 1 and len(linkers) != 1 and len(linkers) != len(metals):
            raise InputError(
                "Linkers and metals must be the same length (they are paired in "
                "order), or one of them a single value used with all of the "
                f"others; you gave {len(linkers)} linkers and {len(metals)} metals."
            )
        if not checked:
            self.check_inputs(linkers, metals)
        self._load()

        # Load and scale metal descriptors.
        metal_descriptors = (
            self.elements.loc[metals][METAL_DESCRIPTORS].reset_index(drop=True)
        )
        metal_descriptors = pd.DataFrame(
            self.metal_scaler.transform(metal_descriptors),
            columns=metal_descriptors.columns,
        )

        # Compute and scale linker descriptors.
        results = []
        for linker in linkers:
            mol = Chem.MolFromSmiles(linker)
            if mol is None:  # unreachable after check_inputs
                raise InputError(f"Invalid SMILES for the linker: {linker!r}.")
            results.append(self.calc(mol).asdict())
        results = pd.DataFrame(results)
        scaled_linker_descriptors = pd.DataFrame(
            self.scaler.transform(results), columns=results.columns
        )[LINKER_DESCRIPTORS]
        scaled_linker_descriptors = scaled_linker_descriptors.fillna(0.0)

        # Broadcast metals against linkers (or vice versa).
        if len(metal_descriptors) == 1:
            metals = metals * len(scaled_linker_descriptors)
            metal_descriptors = pd.concat(
                [metal_descriptors] * len(scaled_linker_descriptors), axis=0
            )
            metal_descriptors.index = scaled_linker_descriptors.index
        elif len(scaled_linker_descriptors) == 1:
            linkers = linkers * len(metals)
            scaled_linker_descriptors = pd.concat(
                [scaled_linker_descriptors] * len(metal_descriptors), axis=0
            )
            scaled_linker_descriptors.index = metal_descriptors.index
        elif len(scaled_linker_descriptors) != len(metal_descriptors):
            # unreachable: lengths are checked above
            raise ValueError(
                "linkers and metals must be the same length, or one must be "
                f"a single value (got {len(linkers)} linkers, {len(metals)} metals)"
            )

        # Merge linker + metal descriptors and run the ensemble.
        all_descs = scaled_linker_descriptors.merge(
            metal_descriptors, left_index=True, right_index=True
        )
        m1_preds = self.m1.predict(all_descs)
        m2_preds = self.m2.predict(all_descs)
        m3_preds = self.m3.predict(all_descs)

        predictions = []
        for i in range(len(metals)):
            m1, m2, m3 = bool(m1_preds[i]), bool(m2_preds[i]), bool(m3_preds[i])
            predictions.append({
                "linker": linkers[i],
                "metal": metals[i],
                "pld_gt_2_4A": m1,
                "pld_gt_4_4A": m2,
                "pld_gt_5_9A": m3,
                "category": self._category(m1, m2, m3),
            })
        return predictions

    def process(
        self,
        mode: str,
        values: dict,
        files: dict[str, tuple[Path, str]],
        output_dir: Path,
        parameters: dict,
        logger,
    ) -> dict:
        """Run the selected Mode."""
        if mode == "instance":
            return self._process_instance(values, logger)
        elif mode == "dataset":
            return self._process_dataset(files, output_dir, logger)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def _process_instance(self, values: dict, logger) -> dict:
        """Instance Mode: linker/metal values -> predictions returned inline."""
        linkers = [str(s).strip() for s in values.get("linkers") or [] if str(s).strip()]
        metals = [str(m).strip() for m in values.get("metals") or [] if str(m).strip()]

        if not linkers or not metals:
            logger.warning("missing linkers or metals")
            raise InputError(
                "Both 'linkers' and 'metals' must be provided: give at least one "
                "linker SMILES and one metal symbol."
            )

        predictions = self.predict(linkers, metals)
        logger.info("computed predictions", count=len(predictions))
        return {"predictions": predictions, "count": len(predictions)}

    def _process_dataset(
        self, files: dict[str, tuple[Path, str]], output_dir: Path, logger
    ) -> dict:
        """Dataset Mode: CSV of linker,metal pairs -> predictions.csv asset."""
        if "file" not in files:
            logger.error("no input file provided")
            raise InputError("No input file provided. Please upload a CSV file.")

        file_path, _ = files["file"]
        rows = read_pairs_csv(file_path)
        labels = [f"row {line}" for line, _, _ in rows]
        linkers = [linker for _, linker, _ in rows]
        metals = [metal for _, _, metal in rows]
        self.check_inputs(linkers, metals, labels, labels)

        predictions = self.predict(linkers, metals, checked=True)
        logger.info("computed predictions from file", rows=len(predictions))

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "predictions.csv"
        fieldnames = [
            "linker", "metal", "pld_gt_2_4A", "pld_gt_4_4A", "pld_gt_5_9A",
            "category",
        ]
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)
        logger.info("saved asset", path=str(output_path))

        return {}  # no value outputs; the asset is the output
