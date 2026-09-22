#!/usr/bin/env bash
# Builds the dtfx_engine pybind11 extension and drops it into research_scripts/ so any
# script there can `import dtfx_engine` exactly like it already does `import dtfx_robust_search`.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PYTHON_EXE="$(command -v python3)"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PYTHON_EXE"
cmake --build build -j

SO_FILE="$(find build -maxdepth 1 -name 'dtfx_engine*.so' | head -1)"
if [ -z "$SO_FILE" ]; then
  echo "Build succeeded but no dtfx_engine*.so found in build/" >&2
  exit 1
fi

cp "$SO_FILE" ../dtfxStrategy/research_scripts/
echo "Built and copied: $(basename "$SO_FILE") -> ../dtfxStrategy/research_scripts/"
echo "Any script in dtfxStrategy/research_scripts/ can now: import dtfx_engine"
