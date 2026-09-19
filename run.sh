#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python || ! -f frontend/dist/index.html ]]; then
  printf "Run ./setup.sh first.\n" >&2
  exit 1
fi
exec .venv/bin/python -m allocator.app "$@"
