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


# ---------------------------------------------------------------------------
# Input errors: bad input raises InputError (a plain message naming the row or
# value) before the heavy models are loaded.
# ---------------------------------------------------------------------------

import json
import re

from infrastructure.handler import InputError
from model.model import read_pairs_csv

SCHEMA = Path(__file__).resolve().parent.parent / "schema" / "model.json"
BDC = "O=C(O)c1ccc(cc1)C(=O)O"


def _fresh() -> MOFPorosityModel:
    return MOFPorosityModel()  # models not loaded yet


def _run_csv(m: MOFPorosityModel, tmp_path: Path, text: str, **kw) -> dict:
    f = tmp_path / "in.csv"
    if isinstance(text, bytes):
        f.write_bytes(text)
    else:
        f.write_text(text, **kw)
    return m.process("dataset", {}, {"file": (f, "text/csv")}, tmp_path / "out", {}, LOG)


@pytest.mark.parametrize("smiles, why", [
    ("c1ccccc", "not valid SMILES syntax"),
    ("C(C)(C)(C)(C)C", "chemically invalid"),
])
def test_instance_bad_smiles_is_input_error(smiles: str, why: str) -> None:
    m = _fresh()
    with pytest.raises(InputError, match=rf"linker 2 '{re.escape(smiles)}' is {why}"):
        m.process("instance", {"linkers": [BDC, smiles], "metals": ["Zn"]},
                  {}, Path("/tmp/mof_out"), {}, LOG)
    assert not m._loaded


@pytest.mark.parametrize("metal, why", [
    ("zn", "did you mean 'Zn'"),
    ("Zn2+", "not a recognised element symbol"),
    ("Zinc", "not a recognised element symbol"),
    ("Kr", "no descriptor data"),
])
def test_instance_unknown_metal(metal: str, why: str) -> None:
    m = _fresh()
    with pytest.raises(InputError, match=rf"Unknown metal: metal 2 '{re.escape(metal)}' .*{why}"):
        m.process("instance", {"linkers": [BDC], "metals": ["Cu", metal]},
                  {}, Path("/tmp/mof_out"), {}, LOG)
    assert not m._loaded


def test_instance_mismatched_lengths_is_input_error() -> None:
    m = _fresh()
    with pytest.raises(InputError, match="2 linkers and 3 metals"):
        m.process("instance", {"linkers": [BDC, BDC], "metals": ["Zn", "Cu", "Fe"]},
                  {}, Path("/tmp/mof_out"), {}, LOG)
    assert not m._loaded


def test_instance_blank_values_only() -> None:
    with pytest.raises(InputError, match="must be provided"):
        _fresh().process("instance", {"linkers": ["  "], "metals": ["Zn"]},
                         {}, Path("/tmp/mof_out"), {}, LOG)


@pytest.mark.parametrize("text, message", [
    ("", "CSV file is empty"),
    ("linker\nCCO\n", "missing: 'metal'. Found columns: 'linker'"),
    ("linker;metal\nCCO;Zn\n", "missing: 'linker', 'metal'"),
    ("linker,metal\n", "header but no data rows"),
    ("linker,metal\n,,\n\n", "no data rows"),
    (f"linker,metal\n{BDC},Zn\n{BDC},\n,Cu\n", "row 3 has no metal; row 4 has no linker"),
    (f"linker,metal\n{BDC},Zn\nc1ccccc,Cu\n", "row 3 'c1ccccc' is not valid SMILES"),
    (f"linker,metal\n{BDC},Zn\n{BDC},Xx\n", "Unknown metal: row 3 'Xx'"),
])
def test_dataset_bad_csv(tmp_path: Path, text: str, message: str) -> None:
    m = _fresh()
    with pytest.raises(InputError, match=message):
        _run_csv(m, tmp_path, text)
    assert not m._loaded


def test_dataset_not_utf8(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="UTF-8"):
        _run_csv(_fresh(), tmp_path, b"linker,metal\n\xff\xfe\x00,Zn\n")


def test_dataset_excel_bom_and_blank_rows(model: MOFPorosityModel, tmp_path: Path) -> None:
    # Excel writes a byte-order mark; blank rows are skipped.
    _run_csv(model, tmp_path, f" linker , metal \n{BDC},Zn\n,\n\n{BDC},Cu\n",
             encoding="utf-8-sig")
    with (tmp_path / "out" / "predictions.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["metal"] for r in rows] == ["Zn", "Cu"]


def test_read_pairs_csv_row_numbers(tmp_path: Path) -> None:
    f = tmp_path / "in.csv"
    f.write_text(f"linker,metal,notes\n{BDC},Zn,x\n\n{BDC},Cu,y\n")
    assert read_pairs_csv(f) == [(2, BDC, "Zn"), (4, BDC, "Cu")]


def test_schema_example_parses_and_predicts(model: MOFPorosityModel, tmp_path: Path) -> None:
    field = json.loads(SCHEMA.read_text())["execution"]["modes"]["dataset"]["inputs"]["file"]
    f = tmp_path / "example.csv"
    f.write_text(field["example"])
    rows = read_pairs_csv(f)
    assert 3 <= len(rows) <= 5
    assert [c["name"] for c in field["columns"]] == ["linker", "metal"]
    model.check_inputs([r[1] for r in rows], [r[2] for r in rows])
    _run_csv(model, tmp_path, field["example"])
    assert (tmp_path / "out" / "predictions.csv").exists()
