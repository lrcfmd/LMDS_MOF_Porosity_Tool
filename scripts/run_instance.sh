#!/usr/bin/env bash
set -euo pipefail

# Predict MOF porosity in instance mode locally (no S3 required).
# Usage: ./scripts/run_instance.sh "<linkers>" "<metals>"
#   <linkers>  comma-separated SMILES string(s) of the linker(s)
#   <metals>   comma-separated elemental symbol(s) of the metal(s)
#
# Broadcasting: one metal applies to all linkers, one linker applies to all
# metals, otherwise the lists are paired in order.
#
# Examples:
#   ./scripts/run_instance.sh "O=C(O)c1ccc(cc1)C(=O)O" "Zn"
#   ./scripts/run_instance.sh "O=C(O)c1ccc(cc1)C(=O)O,c1cc(ccc1C(=O)O)C(=O)O" "Cu"

LINKERS_RAW="${1:-O=C(O)c1ccc(cc1)C(=O)O}"
METALS_RAW="${2:-Zn}"

# Split comma/space separated inputs into JSON arrays.
to_json_array() {
    printf '%s' "$1" | tr ',' '\n' | sed 's/^ *//;s/ *$//' | grep -v '^$' | jq -R . | jq -s .
}

LINKERS_JSON=$(to_json_array "${LINKERS_RAW}")
METALS_JSON=$(to_json_array "${METALS_RAW}")

uv run model-run --local \
  --parameters "$(jq -n \
    --argjson linkers "${LINKERS_JSON}" \
    --argjson metals "${METALS_JSON}" \
    '{mode: "instance", inputs: {linkers: {value: $linkers}, metals: {value: $metals}}, parameters: {}}')"
