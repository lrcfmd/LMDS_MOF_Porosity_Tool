import csv
from pathlib import Path

import pytest

from infrastructure.logging import newLogger
from model.model import MOFPorosityModel

LOG = newLogger("test")
CATEGORIES = {
    "porosity < 2.4 Å",
    "2.4 Å < porosity < 4.4 Å",
    "4.4 Å < porosity < 5.9 Å",
    "porosity > 5.9 Å",
}


@pytest.fixture(scope="module")
def model() -> MOFPorosityModel:
    return MOFPorosityModel()


def _assert_prediction(p: dict, linker: str, metal: str) -> None:
    assert p["linker"] == linker
    assert p["metal"] == metal
    assert isinstance(p["pld_gt_2_4A"], bool)
    assert isinstance(p["pld_gt_4_4A"], bool)
    assert isinstance(p["pld_gt_5_9A"], bool)
    assert p["category"] in CATEGORIES


def test_instance_single(model: MOFPorosityModel) -> None:
    result = model.process(
        "instance",
        {"linkers": ["O=C(O)c1ccc(cc1)C(=O)O"], "metals": ["Zn"]},
        {}, Path("/tmp/mof_out"), {}, LOG,
    )
    assert result["count"] == 1
    assert len(result["predictions"]) == 1
    _assert_prediction(result["predictions"][0], "O=C(O)c1ccc(cc1)C(=O)O", "Zn")


def test_instance_broadcast_one_metal(model: MOFPorosityModel) -> None:
    linkers = ["O=C(O)c1ccc(cc1)C(=O)O", "c1cc(cc(c1)C(=O)O)C(=O)O"]
    result = model.process(
        "instance", {"linkers": linkers, "metals": ["Cu"]},
        {}, Path("/tmp/mof_out"), {}, LOG,
    )
    assert result["count"] == 2
    assert [p["linker"] for p in result["predictions"]] == linkers
    assert all(p["metal"] == "Cu" for p in result["predictions"])


def test_instance_broadcast_one_linker(model: MOFPorosityModel) -> None:
    result = model.process(
        "instance",
        {"linkers": ["O=C(O)c1ccc(cc1)C(=O)O"], "metals": ["Zn", "Cu", "Fe"]},
        {}, Path("/tmp/mof_out"), {}, LOG,
    )
    assert result["count"] == 3
    assert [p["metal"] for p in result["predictions"]] == ["Zn", "Cu", "Fe"]


def test_instance_paired(model: MOFPorosityModel) -> None:
    result = model.predict(
        ["O=C(O)c1ccc(cc1)C(=O)O", "c1cc(cc(c1)C(=O)O)C(=O)O"], ["Zn", "Cu"]
    )
    assert len(result) == 2
    assert result[0]["metal"] == "Zn"
    assert result[1]["metal"] == "Cu"


def test_instance_missing_inputs(model: MOFPorosityModel) -> None:
    with pytest.raises(ValueError, match="must be provided"):
        model.process("instance", {"linkers": [], "metals": ["Zn"]},
                      {}, Path("/tmp/mof_out"), {}, LOG)


def test_instance_invalid_smiles(model: MOFPorosityModel) -> None:
    with pytest.raises(ValueError, match="Invalid SMILES"):
        model.predict(["not_a_smiles!!!"], ["Zn"])


def test_mismatched_lengths(model: MOFPorosityModel) -> None:
    with pytest.raises(ValueError, match="same length"):
        model.predict(
            ["O=C(O)c1ccc(cc1)C(=O)O", "c1cc(cc(c1)C(=O)O)C(=O)O"],
            ["Zn", "Cu", "Fe"],
        )


def test_dataset(model: MOFPorosityModel, tmp_path: Path) -> None:
    csv_file = tmp_path / "in.csv"
    csv_file.write_text(
        "linker,metal\n"
        "O=C(O)c1ccc(cc1)C(=O)O,Zn\n"
        "c1cc(cc(c1)C(=O)O)C(=O)O,Cu\n"
    )
    out = tmp_path / "output"
    result = model.process(
        "dataset", {}, {"file": (csv_file, "text/csv")}, out, {}, LOG
    )
    assert result == {}

    predictions_csv = out / "predictions.csv"
    assert predictions_csv.exists()
    with predictions_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["linker"] == "O=C(O)c1ccc(cc1)C(=O)O"
    assert rows[0]["category"] in CATEGORIES


def test_dataset_no_file(model: MOFPorosityModel) -> None:
    with pytest.raises(ValueError, match="No input file"):
        model.process("dataset", {}, {}, Path("/tmp/mof_out"), {}, LOG)


def test_dataset_bad_columns(model: MOFPorosityModel, tmp_path: Path) -> None:
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("smiles,element\nCCO,Zn\n")
    with pytest.raises(ValueError, match="'linker' and 'metal' columns"):
        model.process("dataset", {}, {"file": (csv_file, "text/csv")},
                      tmp_path / "out", {}, LOG)


def test_unknown_mode(model: MOFPorosityModel) -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        model.process("bogus", {}, {}, Path("/tmp/mof_out"), {}, LOG)
