import {expect,it} from 'vitest';
it('accepts typed graph connections and rejects nonsensical links',()=>{
 const config={edges:[],spec:{mode:'none'}};
 expect(connectGraph(config,{source:'context',target:'server'}).edges).toHaveLength(1);
 expect(connectGraph(config,{source:'sampling',target:'server'}).edges).toHaveLength(1);
 expect(connectGraph(config,{source:'split',target:'server'}).edges).toHaveLength(1);
 expect(connectGraph(config,{source:'vision',target:'server'}).edges).toHaveLength(1);
 expect(()=>connectGraph(config,{source:'server',target:'context'})).toThrow('Connect');
 expect(connectGraph(config,{source:'spec',target:'server'}).spec.mode).toBe('draft-mtp');
});
it('validates imported config without discarding positions or edges',()=>{
 const config={executable:'/bin/server',model_path:'/model.gguf',context:{},spec:{},server:{},devices:{},placements:{},locked:[],positions:{model:{x:22,y:15}},edges:[]};
 expect(validateImport(JSON.stringify(config))).toEqual(config);
 expect(()=>validateImport('{"placements":{}}')).toThrow();
});
import {assignGroups, connectGraph, validateImport, hydrateConfig, launchAllowed} from './workspace';
it('permits acknowledged UNKNOWN only without errors or OOM budgets',()=>{
 const unknown={launchable:false,status:'UNKNOWN',errors:[],memory:[{status:'UNKNOWN'}]};
 expect(launchAllowed(unknown,{acknowledge_estimate:false})).toBe(false);
 expect(launchAllowed(unknown,{acknowledge_estimate:true})).toBe(true);
 expect(launchAllowed({...unknown,errors:['unsupported']},{acknowledge_estimate:true})).toBe(false);
 expect(launchAllowed({...unknown,memory:[{status:'OOM'}]},{acknowledge_estimate:true})).toBe(false);
 expect(launchAllowed(null,{acknowledge_estimate:true})).toBe(false);
});
it('hydrates an unprobed workspace without inventing devices or overwriting deliberate settings',()=>{
 expect(hydrateConfig().sampling).toEqual({});
 expect(hydrateConfig().reasoning).toEqual({});
 expect(hydrateConfig().context.kv_offload).toBe(false);
 const result=hydrateConfig({context:{size:8192,kv_offload:false},sampling:{temperature:0},server:{host:'0.0.0.0'},placement_mode:'layer'});
 expect(result.devices).toEqual({});
 expect(result.split.devices).toEqual([]);
 expect(result.context.kv_offload).toBe(false);
 expect(result.context.size).toBe(8192);
 expect(result.sampling.temperature).toBe(0);
 expect(result.server.allow_network).toBe(false);
 expect(result.acknowledge_estimate).toBe(false);
 expect(result.vision).toMatchObject({enabled:false,model_path:'',device:'RAM'});
 expect(result.environment.cuda_disable_graphs).toBe(false);
 expect(result.placement_mode).toBe('layer');
});
it('manual group allocation locks only those groups and keeps graph layout',()=>{
 const original={placements:{b:'RAM'},locked:[],positions:{model:{x:1,y:2}},edges:[]};
 const result=assignGroups(original,['a'],'CUDA0');
 expect(result.placements).toEqual({a:'CUDA0',b:'RAM'}); expect(result.locked).toEqual(['a']); expect(result.positions).toEqual(original.positions); expect(original.locked).toEqual([]);
});
