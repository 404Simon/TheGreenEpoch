#!/usr/bin/env bash
# Phase C figure pipeline wrapper (Nix-aware).
#
# The venv's numpy/matplotlib C-extensions need shared libraries (libstdc++,
# libz) that a plain venv launch cannot always resolve on Nix. This wrapper
# collects the needed library directories and prepends them to LD_LIBRARY_PATH:
#   1. keep an LD_LIBRARY_PATH already set by the caller (trusted as-is);
#   2. append the gcc-lib dir holding libstdc++.so.6 and the zlib dir holding
#      libz.so.1 found under /nix/store (each is a Nix store path);
#   3. if neither is found, warn and try anyway.
#
# Usage: bash publication/eenergy/figures/make_figures.sh

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$DIR/../.venv"
PY="$VENV/bin/python3"
SCRIPT="$DIR/make_figures.py"

if [[ ! -x "$PY" ]]; then
  echo "ERROR: venv python not found at $PY" >&2
  echo "Create it with: python3 -m venv publication/eenergy/.venv" >&2
  echo "then:           publication/eenergy/.venv/bin/pip install matplotlib" >&2
  exit 1
fi

extra=""
for name in libstdc++.so.6 libz.so.1; do
  lib="$(find /nix/store -name "$name" 2>/dev/null | head -n1 || true)"
  if [[ -n "$lib" ]]; then
    extra="$extra:$(dirname "$lib")"
  fi
done

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}${extra}"
if [[ -z "${LD_LIBRARY_PATH:-}" ]]; then
  echo "WARNING: no libstdc++.so.6 / libz.so.1 found under /nix/store; trying without them." >&2
else
  echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH#:}"
fi

export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"

exec "$PY" "$SCRIPT" "$@"
