"""GGUF inventory and conservative llama.cpp planning; never loads tensor payloads."""
from __future__ import annotations

import copy
import ipaddress
import bisect
import math
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess

MIB = 1024 ** 2
DEFAULT_EXECUTABLE = '/home/vimal/AI/llama.cpp/build/bin/llama-server'
# GGML block sizes from upstream gguf-py/gguf/constants.py. Unknown types fail closed.
QUANTS = {0:('F32',1,4),1:('F16',1,2),2:('Q4_0',32,18),3:('Q4_1',32,20),
6:('Q5_0',32,22),7:('Q5_1',32,24),8:('Q8_0',32,34),9:('Q8_1',32,40),
10:('Q2_K',256,84),11:('Q3_K',256,110),12:('Q4_K',256,144),13:('Q5_K',256,176),
14:('Q6_K',256,210),15:('Q8_K',256,292),16:('IQ2_XXS',256,66),17:('IQ2_XS',256,74),
18:('IQ3_XXS',256,98),19:('IQ1_S',256,50),20:('IQ4_NL',32,18),21:('IQ3_S',256,110),
22:('IQ2_S',256,82),23:('IQ4_XS',256,136),24:('I8',1,1),25:('I16',1,2),26:('I32',1,4),
27:('I64',1,8),28:('F64',1,8),29:('IQ1_M',256,56),30:('BF16',1,2),34:('TQ1_0',256,54),
35:('TQ2_0',256,66),39:('MXFP4',32,17),40:('NVFP4',64,36),41:('Q1_0',128,18),42:('Q2_0',64,18)}


def probe(executable: str) -> dict:
    executable = str(Path(executable).expanduser().resolve(strict=True))
    if not os.access(executable, os.X_OK):
        raise ValueError('Executable is not executable')
    warnings = []

    def run(*args, allow_error=False):
        try:
            p = subprocess.run([executable, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=30, env={k:v for k,v in os.environ.items()
                               if not k.startswith(('LLAMA_', 'GGML_'))})
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f'Capability probe failed: {exc}') from exc
        if p.returncode and not allow_error:
            raise ValueError(f'Capability probe failed ({p.returncode}): {p.stdout[:1000]}')
        return p.stdout

    help_text, version, listing = run('--help'), run('--version'), run('--list-devices')
    blocks = re.split(r'(?m)^(?=-\S)', help_text)
    flags = set()
    for block in blocks:
        if 'removed' in block.lower() or 'deprecated' in block.lower():
            continue
        if block.startswith('-'):
            # Only option declaration, not prose mentioning other flags.
            declaration = block.split('\n')[0].split('  ', 1)[0]
            # Short aliases use padding; instead extract up to description's 2+ spaces.
            declaration = re.split(r'\s{2,}(?=[a-z])', block.split('\n')[0])[0]
            flags.update(re.findall(r'(?<!\w)--[a-z][a-z0-9-]*', declaration))
    cache = re.search(r'--cache-type-k\b[\s\S]*?allowed values: ([^\n]+)', help_text)
    cache_types = [x.strip() for x in cache[1].split(',')] if cache else []
    buffers = []
    if '--override-tensor' in flags:
        text = run('--override-tensor', 'probe=__allocator_probe_invalid__', allow_error=True)
        if 'Available buffer types:' in text:
            buffers = re.findall(r'(?m)^\s{2,}([A-Za-z][A-Za-z0-9_:.-]*)\s*$', text.split('Available buffer types:',1)[1])
    if not buffers:
        warnings.append('No runtime buffer type enumeration; tensor routing cannot be verified.')
    devices = []
    for match in re.finditer(r'(?m)^\s*(\w+): (.+) \((\d+) MiB, (\d+) MiB free\)', listing):
        ident, name, total, free = match.groups()
        alias = next((d['id'] for d in devices if d['name'] == name), None)
        device = dict(id=ident, name=name, total_bytes=int(total)*MIB, free_bytes=int(free)*MIB,
                      kind='cuda' if ident.startswith('CUDA') else 'vulkan' if ident.startswith('Vulkan') else 'gpu',
                      available=not bool(alias), buffer_type=ident if ident in buffers else None)
        if alias:
            device['alias_of'] = alias
            warnings.append(f'{ident} aliases {alias}; not independent physical capacity.')
        devices.append(device)
    mem = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, val = line.split(':',1)
            mem[key] = int(val.strip().split()[0])*1024
    except (OSError, ValueError):
        warnings.append('System RAM capacity unavailable.')
    ram = dict(id='RAM', name='System RAM', total_bytes=mem.get('MemTotal',0), free_bytes=mem.get('MemAvailable',0),
               kind='ram', available=True, buffer_type='CPU' if 'CPU' in buffers else None)
    devices += [ram, dict(ram, id='MMAP', name='Memory mapped (RAM-backed)', kind='mmap', alias_of='RAM')]
    return dict(executable=executable, version=version.strip(), flags=sorted(flags), cache_types=cache_types,
                devices=devices, warnings=warnings, help=help_text, buffer_types=buffers)


def default_config(probe_result=None) -> dict:
    caps = probe_result or {}
    return dict(executable=caps.get('executable', DEFAULT_EXECUTABLE), model_path='',
                devices={d['id']: dict(cap_mib=d.get('free_bytes', 0)/MIB,
                    backend_mib=0 if d['id'] in ('RAM', 'MMAP') else 256,
                    enabled=d.get('available', True)) for d in caps.get('devices', [])},
                placements={}, locked=[], context=dict(size=4096, batch=512, ubatch=128,
                    k_type='f16', v_type='f16', flash='auto', parallel=1, kv_offload=False),
                spec=dict(mode='none', model_path='', tokens=4, p_min=0.75, device='RAM', overhead_mib=512),
                placement_mode='tensor', split=dict(mode='layer', ratios=[], main_gpu=0, gpu_layers=999, devices=[]),
                sampling={}, reasoning={}, vision=dict(enabled=False, model_path='', device='RAM'),
                environment=dict(cuda_disable_graphs=False), acknowledge_estimate=False,
                server=dict(host='127.0.0.1', port=8096, allow_network=False), overhead_mib=512, compute_mib=512,
                mmap=True, positions={}, edges=[])


def _kv_bytes(model, context):
    """CPU KV upper envelope for dense attention; unknown architectures stay unknown."""
    arch, meta = model['architecture'], model['metadata']
    supported = arch in {'llama', 'qwen2', 'qwen3', 'qwen2moe', 'qwen3moe', 'mistral', 'gemma', 'gemma2'}
    prefix = arch + '.'
    layers = model['layer_count']
    hybrid = any('.ssm.' in k for k in meta)
    if hybrid:
        # Only layers with real K projection tensors allocate full-attention cache.
        layers = len({t['layer'] for t in model['tensors'] if '.attn_k.weight' in t['name']})
        supported = False
    heads = meta.get(prefix + 'attention.head_count')
    kvheads = meta.get(prefix + 'attention.head_count_kv', heads)
    embedding = meta.get(prefix + 'embedding_length')
    if not all(isinstance(v, int) and v > 0 for v in (layers, heads, kvheads, embedding)):
        return 0, False
    key = meta.get(prefix + 'attention.key_length', embedding // heads)
    value = meta.get(prefix + 'attention.value_length', embedding // heads)
    types = {v[0].lower():v[1:] for v in QUANTS.values()}
    tokens = (context['size'] + 255) // 256 * 256
    total = 0
    for dim, typ in ((key, context['k_type']), (value, context['v_type'])):
        block, size = types[typ]
        total += math.ceil(dim * kvheads / block) * size * tokens * layers
    # Hybrid/recurrent/indexer caches have extra persistent states and checkpoints.
    # Reserve a conservative extra envelope but do NOT certify these architectures.
    if hybrid or any('.attention.indexer.' in k for k in meta):
        # Recurrent state/checkpoint memory depends on runtime layout. Do not invent
        # an all-layer dense cache or certify an unmeasured recurrent-state formula.
        supported = False
    return int(total), supported


def _validate(cfg, model, caps, mode):
    errors = []
    def number(value, low, high, label, integer=False):
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not low <= value <= high or (integer and int(value) != value):
            errors.append(f'Invalid {label}: expected {low}..{high}' + (' integer' if integer else ''))
            return False
        return True
    if mode not in (None,'maximum','balance'):
        errors.append('Unsupported allocation mode')
    if cfg['executable'] != caps['executable']:
        errors.append('Executable differs from probed executable; re-probe first')
    if cfg['model_path'] and str(Path(cfg['model_path']).expanduser().resolve()) != model['path']:
        errors.append('Model path differs from parsed model')
    for key in ('devices','placements','context','server','spec','positions'):
        if not isinstance(cfg[key],dict):
            errors.append(f'{key} must be an object')
    if errors:
        return errors
    if any(not isinstance(dc,dict) for dc in cfg['devices'].values()):
        return ['Device settings must be objects']
    for section, allowed in [('sampling',{'temperature','top_p','top_k','min_p','presence_penalty','repeat_penalty'}),
                              ('reasoning',{'jinja','effort','preserve'})]:
        if set(cfg[section])-allowed:
            errors.append('Unknown '+section+' setting')
    c = cfg['context']
    for key, limit in [('size',1048576),('batch',65536),('ubatch',65536),('parallel',256)]:
        number(c[key],1,limit,'context.'+key,True)
    if not errors and (c['ubatch'] > c['batch'] or c['parallel'] > c['size']):
        errors.append('ubatch must not exceed batch; parallel must not exceed context size')
    for key in ('k_type','v_type'):
        if c[key] not in caps['cache_types'] or c[key] not in {q[0].lower() for q in QUANTS.values()}:
            errors.append('Unsupported cache type: ' + str(c[key]))
    if c['flash'] not in ('auto','on','off'):
        errors.append('Invalid flash attention mode')
    if c['v_type'] not in ('f16','f32','bf16') and c['flash'] != 'on':
        errors.append('Quantized V cache requires explicit flash attention on; backend support must be verified')
    for key in ('overhead_mib','compute_mib'):
        number(cfg[key],0,1048576,key)
    known = {d['id']:d for d in caps['devices']}
    for obj, key in [(cfg,'acknowledge_estimate'), (c,'kv_offload'), (cfg['vision'],'enabled'),
                     (cfg['environment'],'cuda_disable_graphs'), (cfg['server'],'allow_network')]:
        if not isinstance(obj[key],bool):
            errors.append(key + ' must be boolean')
    for key, limits in {'temperature':(0,100), 'top_p':(0,1), 'top_k':(0,1000000),
                        'min_p':(0,1), 'presence_penalty':(-100,100), 'repeat_penalty':(0,100)}.items():
        if key in cfg['sampling']:
            number(cfg['sampling'][key], *limits, 'sampling.'+key, integer=key=='top_k')
    for key in ('jinja','preserve'):
        if key in cfg['reasoning'] and not isinstance(cfg['reasoning'][key],bool):
            errors.append('reasoning.'+key+' must be boolean')
    if cfg['reasoning'].get('effort','default') not in ('default','none','minimal','low','medium','high','xhigh','max'):
        errors.append('Invalid reasoning effort')
    if cfg['server']['host'] != 'localhost':
        try:
            if not isinstance(cfg['server']['host'],str):
                raise ValueError('host must be a string')
            ipaddress.ip_address(cfg['server']['host'])
        except (ValueError, TypeError):
            errors.append('Server host must be an IP address or localhost')
    if cfg['placement_mode'] not in ('tensor','layer'):
        errors.append('Invalid placement_mode')
    split = cfg['split']
    if split['mode'] != 'layer':
        errors.append('Only standard layer split is supported')
    number(split['gpu_layers'],0,1000000,'split.gpu_layers',True)
    number(split['main_gpu'],0,1024,'split.main_gpu',True)
    if not isinstance(split['devices'],list) or any(not isinstance(i,str) or i not in known or i in ('RAM','MMAP') or not known[i].get('available',True) or not cfg['devices'].get(i,{}).get('enabled',False) for i in split['devices']):
        errors.append('Invalid split devices')
    elif len(set(split['devices'])) != len(split['devices']):
        errors.append('Duplicate split devices')
    ratios = split['ratios']
    if not isinstance(ratios,list):
        errors.append('split.ratios must be a list')
    elif ratios:
        valid = [number(r,0,1000000,'split.ratio') for r in ratios]
        if all(valid) and sum(ratios) <= 0:
            errors.append('split ratios must have positive total')
    for ident, dc in cfg['devices'].items():
        if ident not in known or not isinstance(dc,dict):
            errors.append(f'Unknown/invalid device: {ident}')
            continue
        for key in ('cap_mib','backend_mib'):
            number(dc.get(key,0),0,1048576,ident+'.'+key)
        if not isinstance(dc.get('enabled'),bool):
            errors.append(f'{ident}.enabled must be boolean')
    groupids = {g['id'] for g in model['groups']}
    if not isinstance(cfg['locked'],list) or any(not isinstance(g,str) or g not in groupids for g in cfg['locked']):
        errors.append('Unknown or invalid locked tensor group')
    for gid, ident in cfg['placements'].items():
        if gid not in groupids:
            errors.append('Unknown tensor group; arbitrary expert slices are unsupported: '+gid)
        if not isinstance(ident,str) or ident not in known:
            errors.append('Unknown placement device: '+str(ident))
            continue
        physical = 'RAM' if ident == 'MMAP' else ident
        if not known[ident].get('available',True) or not cfg['devices'].get(ident,{}).get('enabled',False) or not cfg['devices'].get(physical,{}).get('enabled',False):
            errors.append('Placement uses unavailable/disabled/alias device: '+ident)
        if ident == 'MMAP' and not cfg['mmap']:
            errors.append('MMAP placement requires mmap enabled')
    if not isinstance(cfg['mmap'],bool):
        errors.append('mmap must be boolean')
    if cfg['server']['host'] not in ('127.0.0.1','::1','localhost') and cfg['server']['allow_network'] is not True:
        errors.append('Non-loopback server requires explicit allow_network acknowledgment')
    number(cfg['server']['port'],1024,65535,'server.port',True)
    s = cfg['spec']
    if s['mode'] not in ('none','draft-simple','draft-mtp'):
        errors.append('Unsupported speculative mode')
    number(s['tokens'],1,256,'spec.tokens',True)
    number(s['p_min'],0,1,'spec.p_min')
    number(s['overhead_mib'],0,1048576,'spec.overhead_mib')
    if not isinstance(s['device'],str) or s['device'] not in known:
        errors.append('Unknown draft device')
    return errors


def plan(model: dict, caps: dict, config: dict, mode=None) -> dict:
    cfg = default_config(caps)
    merge_errors = []
    for key, value in copy.deepcopy(config).items():
        if key in ('context', 'spec', 'server', 'sampling', 'reasoning', 'vision', 'environment', 'split'):
            if not isinstance(value,dict):
                merge_errors.append(f'{key} must be an object')
            else:
                cfg[key].update(value)
        else:
            cfg[key] = value
    errors = merge_errors or _validate(cfg, model, caps, mode)
    if errors:
        return dict(config=cfg, allocation=dict(cfg.get('placements',{})) if isinstance(cfg.get('placements'),dict) else {},
                    memory=[], status='UNKNOWN', warnings=[], errors=errors, argv=[], command='', launchable=False, assumptions=[])
    warnings = list(caps.get('warnings', []))
    assumptions = ['KV offload follows standard layer devices; manual tensor overrides do not relocate layer KV (ngl=0).',
        'Compute/backend/overhead are reserved allowances, not measured peaks; actual fit must be verified at runtime.',
        'Context size is total server KV capacity, shared across parallel slots.',
        'CPU/MMAP weights are fully resident in RAM budget; MMAP is not an independent memory pool.']
    context = cfg['context']
    kv, modeled = _kv_bytes(model, context)
    if not modeled:
        warnings.append('Architecture/cache layout not fully modeled; KV counts identified attention layers only for hybrid models. Recurrent state/checkpoint memory is unknown; not a fit guarantee.')
    devices = {d['id']:d for d in caps['devices']}
    memory = {}
    for ident, d in devices.items():
        if d.get('alias_of'):
            continue
        dc = cfg['devices'].get(ident, {})
        cap = min(int(dc.get('cap_mib',0)*MIB), d.get('total_bytes',0), d.get('free_bytes',0))
        if ident == 'RAM' and 'MMAP' in cfg['placements'].values():
            cap = min(cap, int(cfg['devices'].get('MMAP',{}).get('cap_mib',0)*MIB))
        memory[ident] = dict(id=ident, name=d['name'], cap_bytes=max(0,cap), used_bytes=0,
             free_bytes=0, status='FIT', weights=0, kv=kv if ident == 'RAM' else 0,
             compute=int(cfg['compute_mib']*MIB), backend=int(dc.get('backend_mib',0)*MIB),
             draft=0, vision=0, overhead=int(cfg['overhead_mib']*MIB) if ident == 'RAM' else 0, mapped_bytes=0)
    vision, projector = cfg['vision'], None
    if vision['enabled']:
        target = vision['device']
        if not isinstance(target,str) or target not in memory or not cfg['devices'].get(target,{}).get('enabled',False):
            errors.append('Vision device unavailable/disabled')
        else:
            try:
                projector = read_model(vision['model_path'])
                if projector['architecture'] != 'clip' and projector['metadata'].get('general.type') != 'mmproj':
                    errors.append('Vision model must be a multimodal projector GGUF')
                memory[target]['vision'] = projector['weight_bytes']
            except (OSError,ValueError,TypeError) as exc:
                errors.append(f'Cannot parse vision projector: {exc}')
        modeled = False
        warnings.append('Vision projector weights are parsed; image compute peak and model compatibility are unknown.')
    spec, draft_model = cfg['spec'], None
    if spec['mode'] != 'none':
        target = 'RAM' if spec['device']=='MMAP' else spec['device']
        if target not in memory or not cfg['devices'].get(target,{}).get('enabled',False):
            errors.append('Draft device unavailable/disabled')
        else:
            if spec['model_path']:
                try:
                    draft_model = read_model(spec['model_path'])
                    draft_kv, _ = _kv_bytes(draft_model,dict(context,k_type='f16',v_type='f16'))
                    memory[target]['draft'] += draft_model['weight_bytes'] + draft_kv
                except (OSError,ValueError,TypeError) as exc:
                    errors.append(f'Cannot parse draft model: {exc}')
            elif spec['mode']=='draft-simple':
                errors.append('draft-simple requires a draft model path')
            memory[target]['draft'] += int(spec['overhead_mib']*MIB)
        modeled = False
        warnings.append('Draft tokenizer/model compatibility and extra runtime memory are not verified; draft allowance is not a measured peak.')
        if spec['mode']=='draft-mtp':
            candidate = draft_model or model
            if not any('nextn_predict_layers' in k and isinstance(v,int) and v>0 for k,v in candidate['metadata'].items()):
                errors.append('Selected model has no declared MTP nextn layers')
            else:
                warnings.append('Same-model MTP reuses embedded nextn weights; extra context/checkpoint memory remains unknown.')
    allocation = {g['id']:cfg['placements'].get(g['id'], 'RAM') for g in model['groups']}
    split = cfg['split']
    split_devices = split['devices'] or [i for i in memory if i != 'RAM' and cfg['devices'].get(i,{}).get('enabled',False)]
    ratios = split['ratios'] or [1]*len(split_devices)
    layer_devices = {}
    if cfg['placement_mode'] == 'layer':
        if len(ratios) != len(split_devices) or (split_devices and split['main_gpu'] >= len(split_devices)):
            errors.append('Split ratio count/main GPU must match selected devices')
        else:
            count = model['layer_count']
            start = max(count + 1 - split['gpu_layers'], 0)
            active = min(split['gpu_layers'], count + 1) if split_devices else 0
            cumulative, total = [], 0
            for ratio in ratios:
                total += ratio
                cumulative.append(total / sum(ratios))
            for layer in range(count + 1):
                layer_devices[layer] = (split_devices[min(bisect.bisect_right(cumulative,(layer-start)/active),len(split_devices)-1)]
                    if active and start <= layer < start+active else 'RAM')
            for g in model['groups']:
                layer = g['layer']
                if layer is None and all(n.startswith(('output.','output_norm.')) for n in g['tensor_names']):
                    layer = count
                allocation[g['id']] = layer_devices.get(layer,'RAM')
            assumptions.append('Standard split estimates whole-layer routing using installed output-inclusive ngl semantics; backend tensor fallbacks/duplicated tied weights may differ.')
    if mode in ('maximum', 'balance') and cfg['placement_mode'] == 'tensor':
        candidates = [i for i in memory if cfg['devices'].get(i,{}).get('enabled',False)
                      and devices[i].get('buffer_type') in caps.get('buffer_types', [])]
        reserved = {i:sum(row[k] for k in ('kv','compute','backend','draft','vision','overhead')) for i,row in memory.items()}
        remaining = {i:memory[i]['cap_bytes'] - reserved[i] for i in memory}
        for g in model['groups']:
            if g['id'] in cfg['locked']:
                target = allocation[g['id']]
                physical = 'RAM' if target == 'MMAP' else target
                if physical in remaining:
                    remaining[physical] -= g['nbytes']
        for g in sorted(model['groups'], key=lambda g:(-g['nbytes'],g['id'])):
            if g['id'] in cfg['locked']:
                continue
            fit = [i for i in candidates if remaining[i] >= g['nbytes']]
            gpu = [i for i in fit if i != 'RAM']
            if gpu:
                if mode == 'maximum':
                    target = min(gpu, key=lambda i:(0 if devices[i]['kind']=='cuda' else 1, i))
                else:
                    target = max(gpu, key=lambda i:remaining[i]/max(1,memory[i]['cap_bytes']))
            elif 'RAM' in fit:
                target = 'RAM'
            else:
                # Retain all weights and expose the overflow rather than dropping a group.
                target = 'RAM'
            allocation[g['id']] = target
            if target in remaining:
                remaining[target] -= g['nbytes']
    if context['kv_offload'] and layer_devices and model['layer_count']:
        memory['RAM']['kv'] = 0
        kv_layers = ({t['layer'] for t in model['tensors'] if '.attn_k.weight' in t['name']}
                     if any('.ssm.' in k for k in model['metadata']) else set(range(model['layer_count'])))
        for layer in kv_layers:
            memory[layer_devices[layer]]['kv'] += math.ceil(kv/len(kv_layers))
    for group in model['groups']:
        target = allocation[group['id']]
        physical = 'RAM' if target == 'MMAP' else target
        if physical not in memory:
            errors.append(f'Unavailable or unknown device: {target}')
            continue
        memory[physical]['weights'] += group['nbytes']
        if cfg['mmap'] and physical == 'RAM':
            memory['RAM']['mapped_bytes'] += group['nbytes']
    argv = [cfg['executable']]
    def flag(name, value=None):
        if name not in caps['flags']:
            errors.append(f'Installed executable does not support required {name}')
        else:
            argv.append(name)
            if value is not None:
                argv.append(str(value))
    flag('--model', model['path'])
    if projector:
        flag('--mmproj',projector['path'])
        if vision['device']=='RAM':
            flag('--no-mmproj-offload')
        else:
            flag('--mmproj-offload')
            flag('--mmproj-device',vision['device'])
    if spec['mode'] != 'none':
        flag('--spec-type',spec['mode'])
        flag('--spec-draft-n-max',spec['tokens'])
        flag('--spec-draft-p-min',spec['p_min'])
        if draft_model:
            flag('--spec-draft-model',draft_model['path'])
            flag('--spec-draft-device','none' if spec['device'] in ('RAM','MMAP') else spec['device'])
            flag('--spec-draft-ngl',0 if spec['device'] in ('RAM','MMAP') else 999)
    for key, name in [('temperature','--temp'), ('top_p','--top-p'), ('top_k','--top-k'),
                      ('min_p','--min-p'), ('presence_penalty','--presence-penalty'), ('repeat_penalty','--repeat-penalty')]:
        if key in cfg['sampling']:
            flag(name, cfg['sampling'][key])
    for key, positive, negative in [('jinja','--jinja','--no-jinja'),
                                     ('preserve','--reasoning-preserve','--no-reasoning-preserve')]:
        if key in cfg['reasoning']:
            flag(positive if cfg['reasoning'][key] else negative)
    if 'effort' in cfg['reasoning']:
        flag('--reasoning-effort', cfg['reasoning']['effort'])
    for name, value in [('--ctx-size',context['size']), ('--batch-size',context['batch']),
        ('--ubatch-size',context['ubatch']), ('--parallel',context['parallel']),
        ('--cache-type-k',context['k_type']), ('--cache-type-v',context['v_type']),
        ('--flash-attn',context['flash']), ('--host',cfg['server']['host']), ('--port',cfg['server']['port']),
        ('--load-mode','mmap' if cfg['mmap'] else 'none'), ('--fit','off'),
        ('--gpu-layers',split['gpu_layers'] if cfg['placement_mode']=='layer' else 0),
        ('--device',','.join(split_devices if cfg['placement_mode']=='layer' else [i for i in memory if i != 'RAM' and i in allocation.values()]) or 'none'), ('--cache-ram',0)]:
        flag(name,value)
    if cfg['placement_mode'] == 'layer':
        flag('--split-mode','layer')
        if ratios:
            flag('--tensor-split',','.join(map(str,ratios)))
        flag('--main-gpu',split['main_gpu'])
    flag('--kv-offload' if context['kv_offload'] else '--no-kv-offload')
    flag('--no-repack')
    flag('--no-op-offload')
    routes = []
    for group in model['groups'] if cfg['placement_mode']=='tensor' else []:
        ident = allocation[group['id']]
        physical = 'RAM' if ident == 'MMAP' else ident
        buffer_type = devices.get(physical, {}).get('buffer_type')
        if buffer_type not in caps.get('buffer_types', []):
            errors.append(f'No verified buffer type for {ident}')
            continue
        # One anchored alternation per indivisible group; no arbitrary expert slicing.
        pattern = '^(' + '|'.join(re.escape(name) for name in group['tensor_names']) + ')$'
        routes.append(pattern + '=' + buffer_type)
    if routes:
        flag('--override-tensor', ','.join(routes))
    status = 'FIT' if modeled and cfg['overhead_mib'] > 0 and cfg['compute_mib'] > 0 else 'UNKNOWN'
    for row in memory.values():
        row['used_bytes'] = sum(row[k] for k in ('weights','kv','compute','backend','draft','vision','overhead'))
        row['free_bytes'] = row['cap_bytes'] - row['used_bytes']
        row['status'] = 'OOM' if row['free_bytes'] < 0 else 'NEAR LIMIT' if row['free_bytes'] < row['cap_bytes']*.1 else 'FIT'
        if row['status'] == 'OOM':
            status = 'OOM'
        elif row['status'] == 'NEAR LIMIT' and status == 'FIT':
            status = 'NEAR LIMIT'
    if errors and status != 'OOM':
        status = 'UNKNOWN'
    if cfg['placement_mode'] == 'tensor':
        cfg['placements'] = allocation.copy()
    return dict(config=cfg, allocation=allocation, memory=list(memory.values()), status=status,
                warnings=warnings, errors=errors, argv=argv, command=shlex.join(argv),
                env={'GGML_CUDA_DISABLE_GRAPHS':'1'} if cfg['environment']['cuda_disable_graphs'] else {},
                launchable=not errors and (status in ('FIT','NEAR LIMIT') or (status == 'UNKNOWN' and cfg['acknowledge_estimate'])), assumptions=assumptions)


class _Reader:
    def __init__(self, stream):
        self.f = stream
        self.size = os.fstat(stream.fileno()).st_size

    def raw(self, n):
        if n < 0 or n > self.size - self.f.tell():
            raise ValueError('Truncated or invalid GGUF')
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError('Truncated GGUF')
        return data

    def number(self, fmt):
        return struct.unpack('<' + fmt, self.raw(struct.calcsize('<' + fmt)))[0]

    def string(self):
        n = self.number('Q')
        if n > 16 * MIB:
            raise ValueError('GGUF string exceeds safety limit')
        return self.raw(n).decode('utf-8', errors='replace')

    def value(self, typ, depth=0):
        formats = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
        if typ in formats:
            return self.number(formats[typ])
        if typ == 8:
            return self.string()
        if typ == 9 and depth < 2:
            element, count = self.number('I'), self.number('Q')
            if count > 2_000_000:
                raise ValueError('GGUF array exceeds safety limit')
            return [self.value(element, depth + 1) for _ in range(count)]
        raise ValueError(f'Unsupported GGUF metadata type {typ}')


def _shard(path):
    with path.open('rb') as f:
        r = _Reader(f)
        if r.raw(4) != b'GGUF' or r.number('I') not in (2, 3):
            raise ValueError('Expected little-endian GGUF v2/v3')
        nt, nm = r.number('Q'), r.number('Q')
        if nt > 1_000_000 or nm > 100_000:
            raise ValueError('GGUF count exceeds safety limit')
        meta = {}
        for _ in range(nm):
            key = r.string()
            if key in meta:
                raise ValueError('Duplicate GGUF metadata key')
            meta[key] = r.value(r.number('I'))
        tensors = []
        for _ in range(nt):
            name, nd = r.string(), r.number('I')
            if not 1 <= nd <= 4:
                raise ValueError('Invalid tensor dimension count')
            shape = [r.number('Q') for _ in range(nd)]
            typ, offset = r.number('I'), r.number('Q')
            if typ not in QUANTS:
                raise ValueError(f'Unsupported GGML type {typ} for {name}')
            label, block, size = QUANTS[typ]
            if not all(shape) or shape[0] % block:
                raise ValueError('Invalid quantized tensor shape')
            nbytes = math.prod(shape) // block * size
            match = re.match(r'blk\.(\d+)\.', name)
            layer = int(match[1]) if match else None
            kind = ('experts' if '_exps' in name else 'attention' if any(x in name for x in ('attn_', 'ssm_'))
                    else 'ffn' if 'ffn_' in name else 'other')
            group = f'blk.{layer}.{kind}' if layer is not None else 'global.' + name
            tensors.append(dict(name=name, shape=shape, type=label, nbytes=nbytes,
                                layer=layer, group=group, offset=offset, shard=str(path)))
        alignment = meta.get('general.alignment', 32)
        if not isinstance(alignment, int) or alignment < 1 or alignment > MIB or alignment & (alignment - 1):
            raise ValueError('Invalid GGUF alignment')
        start = (f.tell() + alignment - 1) // alignment * alignment
        end = 0
        for t in sorted(tensors, key=lambda x: x['offset']):
            if t['offset'] < end or start + t['offset'] + t['nbytes'] > r.size:
                raise ValueError('Overlapping or truncated GGUF tensor payload')
            end = t['offset'] + t['nbytes']
        return meta, tensors


def read_model(path: str) -> dict:
    path = Path(path).expanduser().resolve(strict=True)
    match = re.fullmatch(r'(.*)-(\d{5})-of-(\d{5})\.gguf', path.name)
    paths = [path]
    if match:
        count = int(match[3])
        if not 1 <= count <= 1000:
            raise ValueError('Invalid shard count')
        paths = [path.with_name(f'{match[1]}-{i:05d}-of-{count:05d}.gguf') for i in range(1, count + 1)]
    metadata, tensors, names = {}, [], set()
    for i, shard in enumerate(paths):
        meta, ts = _shard(shard)
        if meta.get('split.count', len(paths)) != len(paths) or meta.get('split.no', i) != i:
            raise ValueError('Missing or inconsistent GGUF shards')
        if i == 0:
            metadata = meta
        elif meta.get('general.architecture', metadata.get('general.architecture')) != metadata.get('general.architecture'):
            raise ValueError('Shard architecture mismatch')
        for t in ts:
            if t['name'] in names:
                raise ValueError('Duplicate tensor across shards: ' + t['name'])
            names.add(t['name'])
            tensors.append(t)
    if metadata.get('split.tensors.count', len(tensors)) != len(tensors):
        raise ValueError('Shard tensor count mismatch')
    groups = {}
    for t in tensors:
        g = groups.setdefault(t['group'], dict(id=t['group'], label=t['group'], layer=t['layer'],
                            kind=t['group'].rsplit('.',1)[-1] if t['layer'] is not None else 'global', tensor_names=[], nbytes=0))
        g['tensor_names'].append(t['name'])
        g['nbytes'] += t['nbytes']
    arch = metadata.get('general.architecture', 'unknown')
    # Tokenizer arrays are parsed/validated, but omitted from the browser payload.
    metadata = {k:v for k,v in metadata.items() if not k.startswith('tokenizer.')}
    return dict(path=str(paths[0]), name=metadata.get('general.name', path.stem), architecture=arch,
                metadata=metadata, tensors=tensors, groups=list(groups.values()), shards=list(map(str, paths)),
                weight_bytes=sum(t['nbytes'] for t in tensors), file_bytes=sum(p.stat().st_size for p in paths),
                layer_count=metadata.get(arch + '.block_count', max((t['layer'] for t in tensors if t['layer'] is not None), default=-1)+1),
                warnings=['Tokenizer metadata omitted from UI payload; tensor data was not loaded.'])
