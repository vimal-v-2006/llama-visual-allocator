import React from 'react';
import {it,expect,vi} from 'vitest';
import {render,screen,fireEvent} from '@testing-library/react';
import Graph,{ModelNode,DeviceNode,ServerNode,ContextNode,SpecNode} from './Graph';
import {hydrateConfig} from './workspace';
vi.mock('@xyflow/react',()=>({Handle:()=>null,Position:{Right:'right',Left:'left'},useUpdateNodeInternals:()=>()=>{},ReactFlow:({nodes,nodeTypes})=><div>{nodes.map(n=>{const Node=nodeTypes[n.type];return <Node key={n.id} id={n.id} data={n.data}/>})}</div>,Background:()=>null,Controls:()=>null,MiniMap:()=>null}));
it('renders all flag nodes safely before devices or new config sections arrive',()=>{
 render(<Graph config={{}} model={null} caps={null} selected={[]} onChange={vi.fn()}/>);
 for(const title of ['Device split','Sampling & reasoning','Vision projector','Context & KV cache']) expect(screen.getByText(title)).toBeVisible();
 expect(screen.getByRole('button',{name:'Focus context'})).toBeVisible();
 expect(screen.getByLabelText('Temperature')).toBeDisabled();
});
it('supports built-in MTP without requiring a separate draft path',()=>{
 const config=hydrateConfig({spec:{mode:'draft-mtp'}}),onChange=vi.fn();
 render(<SpecNode data={{config,caps:{flags:['--spec-type'],devices:[]},onChange}}/>);
 expect(screen.getByLabelText(/Draft model path/)).not.toBeRequired();
 expect(screen.getByText(/Leave empty for built-in MTP/)).toBeVisible();
 fireEvent.change(screen.getByLabelText('Draft mode'),{target:{value:'draft-simple'}});
 expect(onChange.mock.calls.at(-1)[0].spec.mode).toBe('draft-simple');
});
it('makes KV offload an explicit context choice',()=>{
 const onChange=vi.fn();render(<ContextNode data={{config:hydrateConfig(),caps:{flags:['--no-kv-offload']},onChange}}/>);
 expect(screen.getByLabelText('Offload KV cache to GPUs')).not.toBeChecked();
 fireEvent.click(screen.getByLabelText('Offload KV cache to GPUs'));
 expect(onChange.mock.calls.at(-1)[0].context.kv_offload).toBe(true);
});
it('allows explicit network binding without treating acknowledgement as safety',()=>{
 const onChange=vi.fn();render(<ServerNode data={{config:hydrateConfig(),onChange}}/>);
 fireEvent.change(screen.getByLabelText('Bind address'),{target:{value:'0.0.0.0'}});
 expect(onChange.mock.calls.at(-1)[0].server).toMatchObject({host:'0.0.0.0',allow_network:false});
 fireEvent.click(screen.getByLabelText(/Allow network access/));
 expect(onChange.mock.calls.at(-1)[0].server.allow_network).toBe(true);
 fireEvent.click(screen.getByLabelText(/unknown memory estimate/i));
 expect(onChange.mock.calls.at(-1)[0].acknowledge_estimate).toBe(true);
 expect(screen.getByText(/not a safety guarantee/i)).toBeVisible();
 fireEvent.click(screen.getByLabelText('Disable CUDA graphs'));
 expect(onChange.mock.calls.at(-1)[0].environment.cuda_disable_graphs).toBe(true);
});
it('expands real GGUF groups and selects group for assignment',()=>{
 const setSelected=vi.fn();
 render(<ModelNode id="model" data={{model:{name:'Fixture',architecture:'llama',weight_bytes:1024,layer_count:1,groups:[{id:'a',label:'Attention',layer:0,kind:'attention',nbytes:1024,tensor_names:['blk.0.attn_q.weight']}]},config:{placements:{a:'RAM'},locked:['a']},selected:[],setSelected,filter:''}}/>);
 fireEvent.click(screen.getByText('Layer 0'));
 fireEvent.click(screen.getByLabelText('Select Attention'));
 expect(setSelected).toHaveBeenCalledWith(['a']);
 expect(screen.getByText('Attention')).toBeVisible();
});
it('does not accept manual tensor drops while standard layer splitting is active',()=>{
 const onAssign=vi.fn();render(<DeviceNode data={{device:{id:'CUDA0',name:'GPU'},config:hydrateConfig({placement_mode:'layer'}),onChange:vi.fn(),onAssign}}/>);
 fireEvent.drop(screen.getByTestId('device-CUDA0'),{dataTransfer:{getData:()=>JSON.stringify(['a'])}});
 expect(onAssign).not.toHaveBeenCalled();
});
it('keeps mmap CPU-backed placement available without counting extra GPU capacity',()=>{
 const onAssign=vi.fn();render(<DeviceNode data={{device:{id:'MMAP',name:'File backing',alias_of:'RAM',available:true},config:hydrateConfig({devices:{MMAP:{cap_mib:100,backend_mib:0,enabled:true}}}),onChange:vi.fn(),onAssign}}/>);
 fireEvent.drop(screen.getByTestId('device-MMAP'),{dataTransfer:{getData:()=>JSON.stringify(['a'])}});
 expect(onAssign).toHaveBeenCalledWith(['a'],'MMAP');
});
it('edits device cap and assigns a dragged group',()=>{
 const onChange=vi.fn(),onAssign=vi.fn();
 render(<DeviceNode data={{device:{id:'CUDA0',name:'GPU',total_bytes:1024},config:{devices:{CUDA0:{cap_mib:100,backend_mib:20,enabled:true}}},onChange,onAssign}}/>);
 fireEvent.change(screen.getByLabelText('CUDA0 cap / MiB'),{target:{value:'80'}});
 expect(onChange.mock.calls[0][0].devices.CUDA0.cap_mib).toBe(80);
 fireEvent.drop(screen.getByTestId('device-CUDA0'),{dataTransfer:{getData:()=>JSON.stringify(['a'])}});
 expect(onAssign).toHaveBeenCalledWith(['a'],'CUDA0');
});
