# Llama.cpp Visual Allocator MVP

Local-only Python FastAPI control plane and React + @xyflow/react canvas. Linux target. This app does not modify llama.cpp. UI theme: graphite surfaces, amber primary action, cyan CUDA, violet Vulkan, green RAM; compact command/inspect surface.

## Shared contract (all JSON bytes are bytes, memory settings are MiB)

Python package `allocator/`. Frontend `frontend/`. Tests `tests/` unittest (stdlib). App served on 127.0.0.1:8095, managed llama-server default 127.0.0.1:8096.

`allocator/engine.py` exports:
- `probe(executable: str) -> dict`: {executable,version,flags:[str],cache_types:[str],devices:[{id,name,total_bytes,free_bytes,kind}],warnings:[str],help:str}. GPU IDs from --list-devices; RAM + MMAP included. Avoid duplicate physical NVIDIA Vulkan0 with CUDA0 by marking unavailable/alias or excluding default Vulkan0.
- `read_model(path: str) -> dict`: {path,name,architecture,metadata:{},tensors:[{name,shape:[int],type:str,nbytes:int,layer:int|null,group:str}],groups:[{id,label,layer:int|null,kind,tensor_names:[str],nbytes:int}],weight_bytes,file_bytes,layer_count,warnings:[str]}. Real GGUF, all shards, no loading weights into memory. groups IDs stable.
- `default_config(probe_result=None) -> dict`: config schema below.
- `plan(model:dict,caps:dict,config:dict,mode:str|None=None) -> dict`: {config,allocation:{group_id:device_id},memory:[{id,name,cap_bytes,used_bytes,free_bytes,status,weights,kv,compute,backend,draft,overhead,mapped_bytes}],status:'FIT'|'NEAR LIMIT'|'OOM'|'UNKNOWN',warnings:[str],errors:[str],argv:[str],command:str,launchable:bool,assumptions:[str]}. `caps` is probe result. mode `maximum`/`balance` means allocate unlocked groups; manual locks stay. Without mode preserve assignments. Actual group tensor sums, conservative context/KV estimates; uncertainty explicit and no false FIT. MMAP is CPU-backed, not separate compute, count resident budget conservatively in RAM plus mapped bytes separately. Device allocations compile to actual supported override-tensor buffer IDs only. Fail closed when unsupported. Avoid removed/deprecated flags appearing in help. Do not infer buffer names solely from device IDs without installed source/probe proof.

Config schema:
{executable:str,model_path:str,devices:{id:{cap_mib:number,backend_mib:number,enabled:bool}},placements:{group_id:device_id},locked:[group_id],context:{size:4096,batch:512,ubatch:128,k_type:'f16',v_type:'f16',flash:'auto',parallel:1},spec:{mode:'none'|'draft-simple'|'draft-mtp',model_path:'',tokens:4,p_min:0.75,device:'RAM',overhead_mib:512},server:{host:'127.0.0.1',port:8096},overhead_mib:512,compute_mib:512,mmap:true,positions:{},edges:[]}

## API
All `/api/*` except bootstrap require Authorization: Bearer <session>. No cookies. Client consumes #launch=<nonce>, POST /api/bootstrap {nonce}, receives {token}, stores sessionStorage, removes fragment. Random one-use nonce not accepted as API bearer. Enforce exact localhost Host/Origin, sec-fetch-site; bounded JSON body. Static UI contains no secrets.
- POST /api/bootstrap {nonce} -> {token}
- GET /api/state -> {config,capabilities,model:null|model,plan:null|plan,runtime}
- POST /api/probe {executable} -> {capabilities,config}
- POST /api/model {path,executable?} -> {model,capabilities,config,plan}
- POST /api/plan {config,mode?:'maximum'|'balance'} -> plan directly
- POST /api/start {config} -> runtime (fresh capability/model/path checks; launch only valid plan)
- POST /api/stop {} -> runtime
- GET /api/runtime -> {running,pid,returncode,health,logs:[str],rss_bytes,prompt_tps:number|null,generation_tps:number|null,acceptance:number|null,gpu_process_bytes:dict,gpu_global:[],uptime:number}
- Save/load config frontend JSON download/file upload, validated by /api/plan, store graph positions/edges in config. Backend may remember config only in memory; no arbitrary write paths.
Errors {detail:str} appropriate status. Operations async polling 2sec. API never trusts client-supplied model metadata or commands.

## Ownership
Engine agent owns allocator/engine.py and engine tests. Frontend agent owns frontend/ exclusively. Backend agent owns allocator/app.py, allocator/runtime.py, allocator/supervisor.py and API/runtime tests. Parent owns integration, docs, scripts, dependency setup. Avoid editing each other's files without coordinating.

## Verified host facts
Executable /home/vimal/AI/llama.cpp/build/bin/llama-server version 0.4.1-dev build10984 commit543158132. Installed source /home/vimal/AI/llama.cpp (READ ONLY). Devices CUDA0 NVIDIA RTX5070Ti 15833MiB; Vulkan0 same NVIDIA alias; Vulkan1 AMD RX6700XT12272MiB. System RAM33558487040 bytes. Actual model /home/vimal/models/Qwen3.8-Flash-Next-GSQ-RCO/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf (must read second shard too). Help has --override-tensor, --device, --gpu-layers, --split-mode, --tensor-split, --load-mode (NOT old mmap flags), --fit, --cache-type-k/v, --spec-type, --spec-draft-model, --spec-draft-n-max, --spec-draft-p-min. Old --draft-max is explicitly removed, do NOT use. Probe executable live and inspect source for buffer names/routing. Do not count Vulkan0 and CUDA0 as separate capacity.
