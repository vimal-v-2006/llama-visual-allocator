# Llama.cpp Visual Allocator

A local node workspace for inspecting GGUF tensors, budgeting memory, assigning tensor groups to devices, and launching an installed `llama-server` — with a visual tensor-group graph, real (not invented) CLI flags, and honest memory estimates that never hide uncertainty. The application **does not patch llama.cpp** and never silently invents unsupported CLI flags.

## Table of Contents

1. [What it does](#what-it-does)
2. [Requirements](#requirements)
3. [Installation and quick start](#installation-and-quick-start)
4. [First launch: the one-use link](#first-launch-the-one-use-link)
5. [Getting-started guide](#getting-started-guide)
6. [Placement semantics](#placement-semantics)
7. [Modes and estimate limits](#modes-and-estimate-limits)
8. [Configuration reference](#configuration-reference)
9. [Security boundary](#security-boundary)
10. [Project layout](#project-layout)
11. [Development and tests](#development-and-tests)
12. [Verified local environment (host reference)](#verified-local-environment-host-reference)
13. [Troubleshooting](#troubleshooting)

---

## What it does

- **Metadata-only GGUF parsing.** Reads actual per-tensor names, shapes, types, and byte sizes from every shard without loading weights into memory. Architecture is discovered from the file, including unusual tensor groups.
- **Real weight budgeting.** Memory plans use actual per-tensor block sizes — never file-size ÷ layer count.
- **Visual tensor-group graph.** Nodes for tensor groups and devices; drag connections or use placement controls to pin groups to devices. Auto-allocation preserves your pins.
- **Two allocation modes.** *Manual tensor graph* (whole-tensor `--override-tensor` routing) and *standard layer split* (`--split-mode layer`, `--tensor-split`, `--main-gpu`, `-ngl`, optional KV offload).
- **Capability-gated flags.** The selected executable's live `--help`/`--version`/`--list-devices` decide what exists. Removed options may still appear in help; they are treated as unsupported.
- **Honest status.** Plans report `FIT`, `NEAR LIMIT`, `OOM`, or `UNKNOWN`. Unmodeled requirements (hybrid recurrent state, MTP extra contexts, vision compute) stay **UNKNOWN**, never relabeled as a false green FIT. An explicit acknowledgment can allow launching an UNKNOWN plan; it never bypasses errors or OOM.
- **Process ownership.** Starts and stops only the application-owned `llama-server` (default `127.0.0.1:8096`), streams its logs and health, and reports best-effort telemetry without fabricating unavailable counters.
- **Config export/import.** Save/Load JSON configuration, including graph layout. Configs reference local paths; they never carry weights or raw commands.

## Requirements

- Linux x86-64
- Python 3.11+
- Node.js 20+ / npm (frontend build)
- A working `llama-server` build of llama.cpp (your build; CUDA/Vulkan support comes from it, not from this application)
- One or more GGUF model files (use the first shard for sharded models)
- No sudo required. The app executes a selected local program with your user permissions — only select binaries you trust.

## Installation and quick start

From a fresh checkout:

```bash
git clone https://github.com/vimal-v-2006/llama-visual-allocator
cd llama-visual-allocator
./setup.sh   # creates .venv, installs deps, builds frontend/dist
./run.sh     # starts the control plane on http://127.0.0.1:8095
```

`./run.sh` prints a **one-use launch URL** and opens it in your browser. The control plane defaults to `http://127.0.0.1:8095`; the managed model server defaults to `http://127.0.0.1:8096`. These are intentionally different ports.

If a port is already used, pick a different model-server port in the Server node rather than stopping unrelated processes.

## First launch: the one-use link

The printed URL has the form `http://127.0.0.1:8095/#launch=<nonce>`. Opening it exchanges the one-use nonce for a session credential that the UI keeps in browser **sessionStorage** for that origin — no cookies, no host-scoped session.

- **Do not share the link.** The nonce works exactly once.
- The **bare URL alone cannot create authority.** If you lose the session (closed the tab, cleared storage), restart the application and open its new one-use link. Re-using the tab whose sessionStorage holds the session also works.
- Never copy credentials into configuration files.

## Getting-started guide

1. **Point at your binaries and model.** In the app, set the **llama-server executable** path and the **GGUF model path** (first shard for sharded models).
2. **Load Model.** This probes the executable (version, flags, devices, cache types) and reads the GGUF tensor headers. The model card shows architecture, groups, and real weight bytes.
3. **Set memory caps and backend reserves.** For each device set a cap (MiB) below the hardware total to leave room for the desktop and other apps, plus a backend reserve. Remember: **CUDA0 and Vulkan0 on the same NVIDIA GPU are not independent pools** — the app marks the alias and will not count it twice.
4. **Tune the context.** Set context size, batch, microbatch, parallel slots, K/V cache formats, and Flash Attention. Budget estimates update live as settings change.
5. **Allocate.** Use **Auto Maximum Fit** (pack as much as possible) or **Auto Balance** (spread across enabled devices). Then pin manual placements: connect tensor-group outputs to device inputs in the graph, or use the per-group placement controls. Automatic allocation preserves manual pins.
6. **Inspect before launching.** Review the generated command, warnings, and the per-device memory breakdown (weights / KV / compute / backend / draft / overhead / mapped). A memory estimate is not a promise from the driver; unknown architecture-dependent requirements stay visible instead of producing a false green FIT.
7. **Start Server.** Validates the graph/configuration and launches only a supported, launchable plan. Watch real logs, health, RSS, and (when the server emits them) prompt/generation throughput and per-GPU process residency. **Stop Server** stops only the application-owned runtime.
8. **Save / Load.** Export the JSON configuration (including graph layout) to disk or import one to restore a workspace.

**Modeling the big unknowns.** Hybrid recurrent state/checkpoints, MTP extra contexts, and vision-image compute are not fully modeled. They remain **UNKNOWN**; you may explicitly acknowledge an unknown estimate in the Server node to allow a cautious launch. That acknowledgment never overrides OOM or invalid settings.

## Placement semantics

- Weights are budgeted from actual per-tensor block sizes, not file-size divided by number of layers.
- Tensor overrides operate at **whole tensor boundaries**. A packed MoE experts tensor is indivisible through standard `--override-tensor`; individual expert sliders would be misleading, so per-expert slices inside a packed tensor are unsupported. Only real separately stored tensor groups can be routed.
- Shared experts can be routed separately **when they exist as separate tensors**.
- The installed executable's `--help`, `--version`, and `--list-devices` determine capabilities. Removed options may still appear in help; they must not be treated as supported.
- CUDA and Vulkan device names are discovered from the executable. The application never assumes a GPU's display name is an accepted buffer name.
- **mmap / SSD is file backing, not another processor or free RAM.** Mmapped weights remain CPU-executed and need resident working memory. The displayed mapped-file quantity is separate from estimated resident RAM, not additive extra capacity.
- Context/KV and compute/backend reserves are **approximations**. Driver allocations, graph scheduling, weight repacking, and backend fallback can differ at runtime. Logs remain authoritative; the diagram is not proof of measured placement. Some operator/backend combinations fall back to another buffer even when a tensor was assigned — compare actual server allocation logs with estimates.
- The MVP prioritizes conservative, reproducible commands over hidden llama.cpp auto-fit changes. Speculative decoding/MTP is capability-gated; draft compatibility and architecture-specific cache behavior cannot be guaranteed solely from a file path.

## Modes and estimate limits

- **Manual tensor graph:** automatic or pinned group placement compiles to real `--override-tensor` arguments. KV stays on CPU because this mode uses `gpu-layers=0` — routing weights does not relocate the layer cache. The UI explains this rather than promising GPU KV from graph weight placement.
- **Standard layer split:** select GPU order, ratios, main GPU, and offloaded layer count. KV offload (`--kv-offload`) can follow standard layer placement. Graph pins are retained but inactive in this mode.
- **Sampling/reasoning overrides** are optional; unset fields emit no flags and retain installed runtime defaults.
- **Built-in MTP** (`spec.mode: draft-mtp`) accepts an empty draft path only when the model declares next-token prediction layers (`nextn_predict_layers > 0`). External draft mode (`draft-simple`) requires a separate GGUF. The configured spec overhead is an allowance, not a measured additional-context peak.
- **Vision** (`mmproj`) parses the real projector GGUF's weights when enabled; projector placement is capability-gated. Image compute and model compatibility remain UNKNOWN.
- Use the node-focus buttons to read/edit a node at working scale; Overview zooms out to show the entire graph.

## Configuration reference

Config schema (all JSON bytes are bytes; memory settings are MiB):

```jsonc
{
  "executable": "/path/to/llama-server",
  "model_path": "/path/to/model.gguf",
  "devices": { "CUDA0": { "cap_mib": 15000, "backend_mib": 1024, "enabled": true } },
  "placements": { "<group_id>": "CUDA0" },
  "locked": ["<group_id>"],
  "context": { "size": 4096, "batch": 512, "ubatch": 128,
               "k_type": "f16", "v_type": "f16", "flash": "auto",
               "parallel": 1, "kv_offload": false },
  "spec": { "mode": "none", "model_path": "", "tokens": 4,
            "p_min": 0.75, "device": "RAM", "overhead_mib": 512 },
  "server": { "host": "127.0.0.1", "port": 8096, "allow_network": false },
  "overhead_mib": 512,
  "compute_mib": 512,
  "mmap": true,
  "positions": {},
  "edges": []
}
```

Typed optional sections (backward compatible, see `FLAG_CONTROLS.md` for the full contract):

| Section | Keys | Notes |
|---|---|---|
| `placement_mode` | `"tensor"` (default) \| `"layer"` | Layer mode uses only standard split flags; graph pins retained but inactive |
| `split` | `mode`, `ratios[]`, `main_gpu`, `gpu_layers`, `devices[]` | Empty lists resolve to enabled probed GPUs, equal ratios; assignments estimated at whole-layer boundaries from actual group weights |
| `sampling` | `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `repeat_penalty` | Empty emits no sampler flags (installed defaults); never silently imposes a personal recipe |
| `reasoning` | `jinja` (bool), `effort` (none…xhigh), `preserve` (bool) | Empty emits nothing |
| `vision` | `enabled`, `model_path`, `device` | Parses real projector weights; capability-gated placement |
| `environment` | `cuda_disable_graphs` (bool) | When enabled emits only `GGML_CUDA_DISABLE_GRAPHS=1`. **Disabled means the variable is ABSENT — `'0'` also disables graphs** (ggml-cuda checks presence, not value). The runtime applies only explicit child-environment overrides after clearing inherited GGML settings |
| `server.allow_network` | bool | Non-loopback binding of the *managed* server requires explicit `true`; the control plane is always loopback-only |
| `acknowledge_estimate` | bool | Permits launching an UNKNOWN plan only with this flag **and** no errors **and** no OOM rows |

All selected flags are capability-gated against the live installed help. The backend ignores any client-supplied argv/env and derives a fresh engine plan before launch.

## Security boundary

- The application **executes a selected local program with your user permissions**. Only select llama-server binaries you trust. It does not need sudo.
- The **control API is loopback-bound and authenticated**: one-use bootstrap authority is exchanged for a separate session credential (sessionStorage, not a host-scoped cookie). Exact localhost Host/Origin and `sec-fetch-site` are enforced; JSON bodies are bounded; static UI contains no secrets.
- The managed service is distinct from the control plane. The managed model server defaults to loopback; binding it to `0.0.0.0` requires explicit network-access acknowledgment in the UI. **That exposes an unauthenticated inference service** — protect it with your firewall or an authenticated reverse proxy. Do not expose it directly to the internet.
- Commands use argument arrays, never a shell. No arbitrary extra-flags textbox is provided.
- Imported configurations cannot supply raw commands or model metadata to the backend — they are validated and re-planned by the engine.

## Project layout

```
allocator/
  app.py           FastAPI control plane: bootstrap, auth, API, static serving
  engine.py        Metadata-only GGUF parser, capability probe, memory planner,
                   argv/env compiler (no gguf dependency at runtime)
  runtime.py       Owned llama-server lifecycle, logs, health, telemetry
  supervisor.py    Process supervision helpers
frontend/
  src/             React + @xyflow/react node workspace (React Flow canvas)
  dist/            Production build served by the control plane (built by setup.sh)
tests/             Python unittest suite (stdlib) + engine/API/runtime tests
scripts/           probe_mixed_backend.py, verify_workspace.py (browser acceptance)
verification/      Recorded real-run outputs (mixed-backend log, acceptance JSON)
run.sh             Launcher: requires .venv + frontend/dist, starts the app
setup.sh           Fresh-checkout bootstrap
```

## Development and tests

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v     # Python engine/API/runtime tests
cd frontend
npm test          # Vitest: workspace helpers, node routing, React components
npm run build     # Production bundle -> frontend/dist
```

`ARCHITECTURE.md` describes the engine/API/UI contracts; `FLAG_CONTROLS.md` the typed flag-control contract. The Python engine is separated from process ownership and HTTP authentication. Engine-semantics tests normalize probed device `free_bytes` to `total_bytes` so they are hermetic (live-telemetry correctness is covered separately by the probe tests); `plan()` budgets `min(config cap, device total, live free bytes)` by design.

To reproduce the browser acceptance in real Chrome: `.venv/bin/python scripts/verify_workspace.py` (uses the small model in `tests/models/`, source/license in `tests/models/SOURCES.md`).

## Verified local environment (host reference)

The following records the environment where the MVP was verified end-to-end; paths are host-specific.

Executable: `/home/vimal/AI/llama.cpp/build/bin/llama-server` — reports `0.4.1-dev`, build `10984`, commit `543158132`. Devices: CUDA0 (RTX 5070 Ti, 15833 MiB), Vulkan0 (alias of the same NVIDIA GPU), Vulkan1 (AMD RX 6700 XT, 12272 MiB), plus CPU/RAM and MMAP.

A real mixed-backend smoke test with the small SmolLM2-135M Q4_K_M model (`tests/models/SOURCES.md`) loaded successfully, answered a prompt, and shut down cleanly. Raw output: `verification/mixed-backend.log` (verbosity 4 explicitly shows CUDA0 and Vulkan1 model/compute buffers) and `verification/mixed-backend.json`. This is an integration sanity check, **not a performance benchmark**.

The user-supplied Qwen3.8-27B model (`/home/vimal/Documents/Models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf`) reports architecture `qwen35`, 65 blocks (including the next-token prediction block), 866 tensors, 17,548,181,504 weight bytes; its optional BF16 vision projector has 931,126,208 weight bytes. `verification/user-model-reference.json` contains independently extracted metadata. The large model was parsed/planned, **not** launched for long-context/MTP/vision benchmarking; end-to-end runtime validation used the small model.

A separate Qwen3.8-Flash-Next GSQ-RCO model (architecture `qwen4exp`, 48 layers, 1,224 tensors across two shards, 75,828,974,080 tensor bytes including a 28.8 GB per-layer embedding tensor) was also inspected; the build can expose lazy loading for some tensors, but a conservative full-residency plan must not claim it fits by counting the SSD as extra RAM. See `verification/official-gguf-reference.json` for independently extracted metadata. The PyPI GGUF reader at verification time did not yet understand this build's Q2_0 type 42; this project's metadata-only parser handles it.

Browser acceptance (`verification/workspace-acceptance.json`, reproduced by `scripts/verify_workspace.py`) covers: auth bootstrap, nonce replay rejection, hostile Host and unauthenticated rejection, no cookies, real GGUF load, Auto Maximum Fit, context edit increasing KV budget, manual group pin preserved by Auto Balance, save download and config import, actual UI Start button, healthy server, real completion, actual UI Stop button, desktop/narrow overflow checks, zero page JavaScript errors.

## Troubleshooting

- **Access denied / launch link already used:** re-use the browser tab whose sessionStorage holds the session, or restart the application and open its new one-use link. Do not copy credentials into configuration files.
- **Unsupported flag or device:** probe the exact executable you intend to run. GPU capability is a property of the installed build and can change after rebuilding.
- **OOM / unknown estimate:** lower context or batch, change supported KV formats, revise placement/caps, or use a smaller model. mmap does not remove the need for RAM.
- **Model load fails:** ensure every shard is present. Check the error and runtime logs; a valid GGUF container does not guarantee that the installed llama.cpp supports the model architecture.
- **Port already in use:** select a different model-server port rather than stopping unrelated processes.
- **Performance fields show unavailable:** throughput/acceptance require the server to emit corresponding measurements, usually after an inference request. A stopped process does not have measured GPU residency.
- **Graph allocates a tensor but runtime chooses another buffer:** some operator/backend combinations can fall back. Compare actual server allocation logs with estimates; never treat the diagram as proof of measured placement.
- **`setup.sh` says run it first (from `run.sh`):** the .venv or `frontend/dist` is missing — run `./setup.sh` in the project root.

## Design references

- `ARCHITECTURE.md` — engine/API/UI contracts, ownership boundaries, verified host facts.
- `FLAG_CONTROLS.md` — typed optional config sections and their exact flag/env semantics.
