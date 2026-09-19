import React from 'react';
import {Handle,Position} from '@xyflow/react';
import {SlidersHorizontal} from 'lucide-react';
import {NumberField,bytes} from './App';

export function FlagShell({title,id,children}){
 return <section className="graph-node flag-node"><div className="node-heading"><SlidersHorizontal size={16}/><b>{title}</b><span>CONFIG</span></div><Handle type="source" position={Position.Right} id={id}/><div className="node-form nodrag nowheel">{children}</div></section>;
}
export function FlagSupport({caps,flags,children}){
 const missing=flags.filter(flag=>!caps?.flags?.includes(flag));
 return <fieldset className="flag-support" disabled={missing.length>0}><legend className="sr-only">{flags.join(', ')}</legend>{children}{missing.length>0&&<p className="hint">{caps?.flags?'Unavailable in this executable: '+missing.join(', '):'Probe an executable to enable these controls.'}</p>}</fieldset>;
}
export function Toggle({label,value,onChange}){return <label className="check"><input type="checkbox" checked={!!value} onChange={e=>onChange(e.target.checked)}/>{label}</label>}
export function VisionNode({data}){
 const {config,caps,onChange}=data;const vision=config.vision;const update=patch=>onChange({...config,vision:{...vision,...patch}});
 return <FlagShell title="Vision projector" id="vision"><FlagSupport caps={caps} flags={['--mmproj']}><Toggle label="Enable vision projector" value={vision.enabled} onChange={v=>update({enabled:v})}/><fieldset className="flag-support" disabled={!vision.enabled}><label className="field">Projector model path<input value={vision.model_path} placeholder="/path/to/mmproj.gguf" onChange={e=>update({model_path:e.target.value})}/></label><label className="field">Projector device<select value={vision.device} onChange={e=>update({device:e.target.value})}><option value="RAM">RAM · CPU projector</option></select></label><div className="field">Parsed projector weights<output aria-label="Projector weight budget">{vision.enabled?bytes(data.plan?.memory?.find(m=>m.id===vision.device)?.vision):'Disabled · not budgeted'}</output></div></fieldset></FlagSupport><p className="hint">Optional. Parsed weights enter the memory ledger. Image compute and model compatibility remain unknown. GPU placement is not yet verified; CPU requires --no-mmproj-offload.</p></FlagShell>;
}

export function SamplingNode({data}){
 const {config,caps,onChange}=data;
 const update=(section,key,value)=>{const next={...config[section]};if(value===undefined||value==='')delete next[key];else next[key]=value;onChange({...config,[section]:next})};
 return <FlagShell title="Sampling & reasoning" id="sampling"><div className="form-grid">{[
  ['Temperature','temperature','--temp',0,undefined,0.05],['Top p','top_p','--top-p',0,1,0.01],['Top k','top_k','--top-k',0,undefined,1],['Min p','min_p','--min-p',0,1,0.01],['Presence penalty','presence_penalty','--presence-penalty',-2,2,0.1],['Repeat penalty','repeat_penalty','--repeat-penalty',0,undefined,0.05],
 ].map(([label,key,flag,min,max,step])=><FlagSupport key={key} caps={caps} flags={[flag]}><NumberField label={label} optional placeholder="Installed default" value={config.sampling[key]} min={min} max={max} step={step} onChange={v=>update('sampling',key,v)}/></FlagSupport>)}</div>
 <FlagSupport caps={caps} flags={['--jinja']}><Toggle label="Jinja chat template" value={config.reasoning.jinja} onChange={v=>update('reasoning','jinja',v)}/></FlagSupport>
 <FlagSupport caps={caps} flags={['--reasoning-effort']}><label className="field">Reasoning effort<select value={config.reasoning.effort??''} onChange={e=>update('reasoning','effort',e.target.value)}><option value="">Installed default</option>{['none','minimal','low','medium','high','xhigh','max'].map(v=><option key={v}>{v}</option>)}</select></label></FlagSupport>
 <FlagSupport caps={caps} flags={['--reasoning-preserve']}><Toggle label="Preserve reasoning in conversation" value={config.reasoning.preserve} onChange={v=>update('reasoning','preserve',v)}/></FlagSupport><p className="hint">Template and model support determine how reasoning is applied.</p>
 </FlagShell>;
}

export function SplitNode({data}){
 const {config,caps,onChange}=data; const split=config.split;
 const update=patch=>onChange({...config,split:{...split,...patch}});
 const devices=(caps?.devices||[]).filter(d=>!['RAM','MMAP'].includes(d.id));
 const available=d=>d.available!==false&&d.supported!==false&&!d.alias_of&&config.devices?.[d.id]?.enabled!==false;
 const order=split.devices?.length?split.devices:devices.filter(available).map(d=>d.id);const ratios=split.ratios||[];
 const include=(id,enabled)=>{const i=order.indexOf(id);update(enabled?{devices:[...order,id],ratios:[...order.map((_,i)=>ratios[i]??1),1]}:{devices:order.filter(x=>x!==id),ratios:order.flatMap((_,j)=>j===i?[]:[ratios[j]??1]),main_gpu:0})};
 const move=(i,delta)=>{const indices=order.map((_,j)=>j);[indices[i],indices[i+delta]]=[indices[i+delta],indices[i]];update({devices:indices.map(j=>order[j]),ratios:indices.map(j=>ratios[j]??1),main_gpu:0})};
 return <FlagShell title="Device split" id="split"><label className="field">Placement mode<select value={config.placement_mode} onChange={e=>onChange({...config,placement_mode:e.target.value})}><option value="layer">Standard layer split</option><option value="tensor">Manual tensor graph</option></select></label>
 <p className="hint">{config.placement_mode==='layer'?'llama.cpp distributes whole layers. Ratios are relative weights in device order. Empty lists use all available GPUs equally. Set GPU layers to 0 for CPU-only. Tensor graph assignments are retained but inactive.':'Wire tensor groups to device budgets. Standard split settings are retained but inactive.'}</p>
 <fieldset className="flag-support" disabled={config.placement_mode!=='layer'}><FlagSupport caps={caps} flags={['--split-mode','--tensor-split','--device']}>
 {devices.length?devices.map(d=><label className="check" key={d.id}><input aria-label={'Include '+d.id} type="checkbox" disabled={(!available(d)&&!order.includes(d.id))||(order.length===1&&order.includes(d.id))} checked={order.includes(d.id)} onChange={e=>include(d.id,e.target.checked)}/>{d.id}{!available(d)?' · unavailable / alias':''}</label>):<p className="hint">No probed GPU devices. Load or probe an executable above.</p>}
 {order.map((id,i)=><div className="split-row" key={id}><span className="split-index">{i}</span><NumberField label={id+' split ratio'} value={ratios[i]??1} step={0.1} onChange={v=>update({devices:order,ratios:order.map((_,j)=>j===i?v:ratios[j]??1)})}/><button aria-label={`Move ${id} earlier`} disabled={i===0} onClick={()=>move(i,-1)}>↑</button><button aria-label={`Move ${id} later`} disabled={i===order.length-1} onClick={()=>move(i,1)}>↓</button></div>)}
 </FlagSupport><div className="form-grid"><FlagSupport caps={caps} flags={['--main-gpu']}><label className="field">Main GPU index<select value={split.main_gpu} onChange={e=>update({main_gpu:Number(e.target.value)})}>{order.length?order.map((id,i)=><option value={i} key={id}>{i} · {id}</option>):<option value={0}>0 · no GPU selected</option>}</select></label></FlagSupport><FlagSupport caps={caps} flags={['--gpu-layers']}><NumberField label="GPU layers (ngl)" min={-1} value={split.gpu_layers} onChange={v=>update({gpu_layers:v})}/></FlagSupport></div></fieldset>
 </FlagShell>;
}
