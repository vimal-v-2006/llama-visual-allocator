# Handoff — working MVP verified

Project: `/home/vimal/Documents/Projects/llama-visual-allocator`

## Current status

The MVP is implemented, built, and verified end-to-end. All subagents finished. Parent resolved the integration blockers. **Do not repeat the old hidden-node or environment-null fixes.**

The control plane was started with `./run.sh` on `http://127.0.0.1:8095`; terminal process handle `proc_d75dfa6de88f` (PID83579 at startup). Readiness verified: `/`200, unauthenticated `/api/state`403. The launcher invokes the system browser with a one-use launch-fragment URL. Runtime process details are temporary: check before restarting/stopping. No large user model is running.

**Session update (re-verified same day, ~09:27):** control plane still the same PID 83579, `/`200 and unauthenticated `/api/state`403 confirmed again. A user-owned 27B llama-server (`/bin/bash /home/vimal/AI/scripts/run-qwen.sh`, PID 85217, started 09:18, bound `0.0.0.0:8080`) now occupies ~13.4 GiB CUDA0 and Vulkan1 — the "no large user model is running" line above no longer holds; it was intentionally left untouched. Python suite re-run exposed one environment-dependent failure: `test_hybrid_real_model_uses_only_attention_layers_for_kv` asserts status UNKNOWN / acknowledged-launchable, which requires free GPU headroom because plan caps are `min(cap, total, free)` by design; with the user's server running it correctly reports OOM. Fix (tests only, no engine contract change): `ControlsTests.setUpClass` now normalizes `free_bytes = total_bytes` so engine-semantics tests are hermetic. Result: **34/34 Python tests pass with the user's server running**. Frontend re-verified: **22/22 tests pass**; `frontend/dist/index.html` is newer than all sources, so the recorded `npm run build` remains current (not rebuilt; running app keeps serving current dist). Browser acceptance not re-run (recorded output `verification/workspace-acceptance.json` verified intact: Paris… completion, ~200 prompt tps / 59.75 generation tps).

If resuming later:
```bash
cd /home/vimal/Documents/Projects/llama-visual-allocator
./run.sh
```
Do not start a second copy if8095 is already occupied. Use the already-open browser tab. If the session is lost, stop/restart the owned control-plane process and open its new one-use link. Static bare URL alone cannot create authority.

## Verified by parent (real execution)

- Python: **34 tests passed** via `.venv/bin/python -m unittest discover -s tests -q`.
- Frontend: **22 tests passed** via `cd frontend && npm test`.
- Production frontend: `npm run build` successful.
- Real Chrome acceptance: `.venv/bin/python scripts/verify_workspace.py` passed.
- Browser test covers auth bootstrap, nonce replay rejection, hostile Host and unauthenticated rejection, no cookies, real GGUF load, Auto Maximum Fit, context edit increasing KV budget, manual group pin to Vulkan1 preserved by Auto Balance, Save download and config import, actual **UI Start button**, healthy server, real completion, actual **UI Stop button**, desktop/narrow overflow checks, zero page JavaScript errors.
- `verification/workspace-acceptance.json` records actual output. Latest completion: Paris…; approx200 prompt tokens/s and59.75 generation tokens/s. These are smoke-test timings, not a performance benchmark.
- `verification/mixed-backend.log` at verbosity4 explicitly shows CUDA0 model buffer24.01MiB, Vulkan1 model buffer23.22MiB and compute buffers on both, followed by successful generation and clean exit. This is not merely command acceptance.
- Screenshots `verification/workspace-desktop.png`, `workspace-narrow.png` captured. Overview is intentionally zoomed out; focus buttons make individual nodes readable/editable.

## Resolved parent fixes

1. ReactFlow controlled-node dimensions were discarded, leaving nodes CSS visibility:hidden. `Graph.jsx` now preserves dimensions in measurements state and supplies node.measured. Real browser confirms context controls visible/editable.
2. Layers now sort numerically rather than lexicographically.
3. Disabled CUDA graphs override now returns empty env dict, not null or '0'. Runtime sanitizes inherited environment. Enabled emits only `GGML_CUDA_DISABLE_GRAPHS=1`.
4. Acknowledged UNKNOWN plan is launchable only with explicit config flag and no errors/OOM. Status remains UNKNOWN. Regression tested.
5. Trusted installed binary is0775 vimal:vimal. Runtime allows current-user-owned executable group write only for verified private same-name group whose members are owner/root; world-write and shared-group-write still rejected. User binary permissions were NOT changed.
6. mmap CPU-backed node remains usable despite alias_of RAM; duplicate physical GPU aliases remain disabled. Regression tested.
7. Removed remote Google Fonts import so UI does not rely on external assets/CSP exceptions.
8. README and FLAG_CONTROLS.md updated to match current env, network, uncertainty semantics.

## Intentional MVP limits / future work

- Memory is an estimate, not an OOM guarantee. Tensor weight sizes are real; compute/backend/headroom allowances are editable approximations.
- Hybrid recurrent state/checkpoints, MTP extra contexts and vision-image compute are not fully modeled. They remain UNKNOWN; explicit acknowledgment can allow a cautious launch. Never relabel as proven FIT.
- mmap/SSD is CPU file backing, not extra physical RAM. Current planner conservatively budgets resident CPU weights. Advanced lazy embedding row residency is not modeled.
- Manual tensor graph mode uses ngl0 and cannot move layer KV via weight overrides. Standard layer split mode supports GPU KV. UI explains this.
- Per-expert slices inside a packed tensor are unsupported. Only real separately stored tensor groups can be routed.
- GPU telemetry is best effort; unsupported/unavailable process counters must remain unavailable, not fabricated. Global GPU use is separate.
- Large Qwen and mmproj were parsed/planned, NOT started for long-context/MTP/vision benchmarking. End-to-end runtime validation used a small actual GGUF.
- Node routing helpers and manual-drop behavior have frontend tests; the browser acceptance specifically verifies manual placement via selection/dropdown and preserved locks, not every possible handle-to-handle drag gesture.
- No installer package/public GitHub release requested or created. No git repo initialized.
- Direct network exposure of the managed inference server requires acknowledgment; control API remains authenticated loopback-only. Use firewall/authenticated reverse proxy for real network deployments.

## Real model/build paths

Executable `/home/vimal/AI/llama.cpp/build/bin/llama-server`: version0.4.1-dev build10984 commit543158132.
User model `/home/vimal/Documents/Models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf`; optional `mmproj-BF16.gguf` same folder. qwen35,65blocks including nextn1,866tensors,17548181504weight bytes. Projector931126208bytes. Reference `verification/user-model-reference.json`.

Small verification model `tests/models/SmolLM2-135M-Instruct-Q4_K_M.gguf` (source/license reference in tests/models/SOURCES.md).
A separate huge qwen4exp in ~/models was inspected earlier, not the user's target model. Parser handles its shards and Q2_0type42.

## Test hermeticity note

`plan()` budgets `cap = min(config cap_mib, device total, live free_bytes)`; `default_config` seeds `cap_mib` from live free bytes. Engine tests that assert FIT/UNKNOWN/launchability therefore normalize probe `free_bytes` to `total_bytes` in test setup. Live-telemetry correctness is covered separately by `tests/test_engine_probe.py`.

## Development

Project .venv includes fastapi/uvicorn/httpx/pytest/gguf/playwright. Chrome `/usr/bin/google-chrome`. Frontend React/Vite/@xyflow/react. Engine uses a metadata-only parser without requiring gguf dependency at runtime. Read `README.md`, `ARCHITECTURE.md`, `FLAG_CONTROLS.md` before modifying contracts.

Keep new work scoped to real requested improvements; do not rebuild this app from scratch.
