// UI migration defaults only; device IDs and capacities always come from the probe.
export function hydrateConfig(config = {}) {
 const defaults = {
  executable:'', model_path:'', devices:{}, placements:{}, locked:[], positions:{}, edges:[],
  placement_mode:'tensor', split:{mode:'layer',ratios:[],main_gpu:0,gpu_layers:999,devices:[]},
  context:{size:4096,batch:512,ubatch:128,k_type:'f16',v_type:'f16',flash:'auto',parallel:1,kv_offload:false},
  spec:{mode:'none',model_path:'',tokens:4,p_min:0.75,device:'RAM',overhead_mib:512},
  sampling:{},
  reasoning:{},
  vision:{enabled:false,model_path:'',device:'RAM'},
  environment:{cuda_disable_graphs:false}, server:{host:'127.0.0.1',port:8096,allow_network:false},
  acknowledge_estimate:false, overhead_mib:512,compute_mib:512,mmap:true,
 };
 const result={...defaults,...config};
 for(const key of ['split','context','spec','sampling','reasoning','vision','environment','server']) result[key]={...defaults[key],...config[key]};
 return result;
}

export function launchAllowed(plan,config) {
 if(!plan||plan.errors?.length||plan.status==='OOM'||plan.memory?.some(m=>m.status==='OOM'))return false;
 return plan.launchable===true||(plan.status==='UNKNOWN'&&Array.isArray(plan.errors)&&Array.isArray(plan.memory)&&config?.acknowledge_estimate===true);
}

export function connectGraph(config, edge) {
 if (!['model','context','spec','split','sampling','vision'].includes(edge.source) || edge.target !== 'server') throw new Error('Connect model or configuration outputs to the server input.');
 const id = edge.source + ':' + edge.target;
 return {...config, edges:[...(config.edges||[]).filter(e=>e.id!==id),{...edge,id}],...(edge.source==='spec'?{spec:{...config.spec,mode:config.spec.mode==='none'?'draft-mtp':config.spec.mode}}:{})};
}
export function validateImport(text) {
 const c=JSON.parse(text);
 if (!c || typeof c!=='object' || typeof c.executable!=='string' || typeof c.model_path!=='string' || !['devices','placements','context','spec','server','positions'].every(k=>c[k] && typeof c[k]==='object' && !Array.isArray(c[k])) || !Array.isArray(c.locked) || !Array.isArray(c.edges)) throw new Error('Invalid workspace: expected a complete allocator configuration.');
 for (const p of Object.values(c.positions)) if (!Number.isFinite(p.x)||!Number.isFinite(p.y)) throw new Error('Invalid graph coordinates.');
 return c;
}
export function assignGroups(config, groups, device) {
 return {...config,placements:{...config.placements,...Object.fromEntries(groups.map(id=>[id,device]))},locked:[...new Set([...(config.locked||[]),...groups])]};
}
