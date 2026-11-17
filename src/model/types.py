# Auto-generated from model.json - DO NOT EDIT
# Regenerate with: uv run model-types <path-to-model.json> -o src/model/types.py

from pathlib import Path
from typing import TypedDict, Literal

class InstanceInputs(TypedDict, total=False):
    # SMILES string(s) of the organic linker(s), e.g. O=C(O)c1ccc(cc1)C(=O)O
    linkers: list[str]
    # Metal element symbol(s), e.g. Zn, Cu, Fe (one for all linkers, or one per linker in order)
    metals: list[str]

class DatasetInputs(TypedDict, total=False):
    # CSV of MOFs, one per row: the linker SMILES and the metal's element symbol
    file: tuple[Path, str]