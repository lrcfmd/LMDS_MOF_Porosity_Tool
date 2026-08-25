import csv
import os
from pathlib import Path

import joblib
import pandas as pd
from mordred import Calculator, descriptors as mordred_descriptors
from rdkit import Chem

from infrastructure.logging import newLogger


# Model artifacts are NOT committed to git; they are provided to the application
# at runtime. Resolve their directory from MODEL_ARTIFACTS_DIR (set by the
# deployment, e.g. a mounted volume), falling back to the package's own
# artifacts/ dir for running from a source checkout. See artifacts/README.md.
def _artifacts_dir() -> Path:
    env = os.environ.get("MODEL_ARTIFACTS_DIR")
    if env:
        return Path(env)
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
        self.elements = (
            pd.read_csv(artifacts / "elemental_descriptors.csv", index_col="Symbol")
            .drop("Name", axis=1)
        )
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

    def predict(self, linkers: list[str], metals: list[str]) -> list[dict]:
        """Core prediction path shared by both Modes.

        Broadcasting mirrors the original tool: a single metal is applied to all
        linkers, a single linker is applied to all metals, otherwise the two
        lists are paired in order (and must be equal length).
        """
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
            if mol is None:
                raise ValueError(f"Invalid SMILES: {linker!r}")
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
        linkers = [s.strip() for s in values.get("linkers", []) if str(s).strip()]
        metals = [m.strip() for m in values.get("metals", []) if str(m).strip()]

        if not linkers or not metals:
            logger.warning("missing linkers or metals")
            raise ValueError("Both 'linkers' and 'metals' must be provided")

        predictions = self.predict(linkers, metals)
        logger.info("computed predictions", count=len(predictions))
        return {"predictions": predictions, "count": len(predictions)}

    def _process_dataset(
        self, files: dict[str, tuple[Path, str]], output_dir: Path, logger
    ) -> dict:
        """Dataset Mode: CSV of linker,metal pairs -> predictions.csv asset."""
        if "file" not in files:
            logger.error("no input file provided")
            raise ValueError("No input file provided")

        file_path, _ = files["file"]
        linkers, metals = [], []
        with file_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not {"linker", "metal"} <= set(
                name.strip() for name in reader.fieldnames
            ):
                raise ValueError("CSV must have 'linker' and 'metal' columns")
            # Normalise header whitespace, then read rows.
            for row in reader:
                clean = {k.strip(): (v or "").strip() for k, v in row.items()}
                if clean.get("linker") and clean.get("metal"):
                    linkers.append(clean["linker"])
                    metals.append(clean["metal"])

        if not linkers:
            logger.error("no rows found in input file")
            raise ValueError("No linker/metal rows found in input file")

        predictions = self.predict(linkers, metals)
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
