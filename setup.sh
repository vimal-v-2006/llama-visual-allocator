#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
command -v python3 >/dev/null || { printf "Python 3.11+ required\n" >&2; exit 1; }
command -v npm >/dev/null || { printf "Node.js 20+ and npm required\n" >&2; exit 1; }
if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install -r requirements.txt
(cd frontend && npm ci && npm run build)
printf "Ready. Run ./run.sh\n"
