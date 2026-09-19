# Engine flag controls contract

Engine adds these typed optional sections, preserving old config compatibility:
- `placement_mode: 'tensor' | 'layer'` default tensor (manual graph overrides). Layer mode uses only standard split flags, never override-tensor. Graph placements/locks retained but inactive in layer mode.
- `split: {mode:'layer', ratios:[], main_gpu:0, gpu_layers:999, devices:[]}`. Empty lists resolve to enabled probed GPUs, equal ratios. Explicit ratios preserved (e.g. [11,6]); assignments estimated at whole-layer boundaries from actual group weights, not a byte split.
- `context.kv_offload: false` retains old default. True removes --no-kv-offload and estimates per-layer cache on assigned devices.
- `sampling: {}` optional keys temperature, top_p, top_k, min_p, presence_penalty, repeat_penalty. Empty defaults emit no sampler flags (installed defaults). UI may provide normal placeholders, never silently impose personal recipe.
- `reasoning: {}` optional jinja (bool), effort (none|minimal|low|medium|high|xhigh), preserve (bool). Empty emits nothing.
- `vision: {enabled:false, model_path:'', device:'RAM'}` enabled parses real projector GGUF weights; capability-gated projector placement.
- `environment: {cuda_disable_graphs:false}` plan returns `env: {'GGML_CUDA_DISABLE_GRAPHS':'1'}` when enabled, `{}` when disabled; runtime applies only explicit child environment overrides after clearing inherited GGML settings. **IMPORTANT: disabled means the variable is ABSENT; '0' also disables graphs!** Installed ggml-cuda/common.cuh tests getenv presence, not value.
- `server: {host:'127.0.0.1', port:8096, allow_network:false}` non-loopback requires explicit true allow_network. This only applies to managed llama-server, never control-plane binding.
- `acknowledge_estimate:false`: engine and backend permit UNKNOWN only with this explicit acknowledgment AND no errors AND no OOM rows. Never bypass errors/OOM.
- `spec.mode:'draft-mtp'` supports empty model_path for same-model built-in MTP. Extra memory unknown, warning/UNKNOWN, not missing-path error. External draft-simple still requires real model.

All selected flags capability-gated against live installed help. New plan retains existing fields; adds env and memory row `vision` bytes (included in used_bytes). Backend must ignore client argv/env and derive a fresh engine plan before launch. Do not start the large Qwen model for verification.

Verified implementation notes:
- Installed source counts the output layer in `gpu_layers` (`start=max(block_count+1-ngl,0)`). Group estimates follow cumulative ratio boundaries over actual layer indices. This is approximate: backend fallbacks or tied-weight duplication can differ.
- Manual tensor mode emits ngl=0: KV remains CPU even when kv_offload=true, since tensor overrides do not change layer device assignment. UI should explain this rather than promise GPU KV from graph weight placement.
- Hybrid KV uses only actual attn_k layers, not all recurrent blocks. Recurrent states/checkpoints remain unknown rather than an inflated fabricated estimate. User Qwen qwen35: 866 tensors, 17,548,181,504 weight bytes; q4 K+V at 131072 has 2,566,914,048 known attention-cache bytes. Never launched.
- Projector has 931,126,208 parsed weight bytes; image compute/model compatibility remain UNKNOWN.
- Same-model MTP requires declared positive nextn_predict_layers, no external path. Configured spec.overhead_mib is an allowance, not a measured additional-context peak.

