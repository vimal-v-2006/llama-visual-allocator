import React from 'react';
import {it,expect,vi} from 'vitest';
import {render,screen,fireEvent} from '@testing-library/react';
import {SplitNode,SamplingNode,VisionNode} from './FlagNodes';
import {hydrateConfig} from './workspace';
vi.mock('@xyflow/react',()=>({Handle:()=>null,Position:{Right:'right',Left:'left'},useUpdateNodeInternals:()=>()=>{},ReactFlow:()=>null,Background:()=>null,Controls:()=>null,MiniMap:()=>null}));
const caps={flags:['--split-mode','--tensor-split','--main-gpu','--gpu-layers','--device'],devices:[{id:'CUDA0',name:'GPU A'},{id:'Vulkan0',alias_of:'CUDA0'},{id:'Vulkan1',name:'GPU B'}]};
it('keeps the vision projector optional and budgets it only after enabling',()=>{
 const onChange=vi.fn();const config=hydrateConfig();
 const {rerender}=render(<VisionNode data={{config,caps:{flags:['--mmproj','--no-mmproj-offload']},onChange}}/>);
 expect(screen.getByLabelText('Projector model path')).toBeDisabled();
 fireEvent.click(screen.getByLabelText('Enable vision projector'));
 expect(onChange.mock.calls.at(-1)[0].vision.enabled).toBe(true);
 rerender(<VisionNode data={{config:{...config,vision:{...config.vision,enabled:true}},caps:{flags:['--mmproj','--no-mmproj-offload']},onChange}}/>);
 fireEvent.change(screen.getByLabelText('Projector model path'),{target:{value:'/vision.gguf'}});
 expect(onChange.mock.calls.at(-1)[0].vision.model_path).toBe('/vision.gguf');
 expect(screen.getByLabelText('Projector weight budget')).toHaveTextContent('unavailable');
 expect(screen.queryByLabelText('Projector overhead / MiB')).not.toBeInTheDocument();
 expect(screen.getByLabelText('Projector device')).toHaveValue('RAM');
});
it('keeps installed sampling defaults until explicitly edited and permits reset',()=>{
 const onChange=vi.fn();render(<SamplingNode data={{config:hydrateConfig({sampling:{temperature:0.4}}),caps:{flags:['--temp','--reasoning-effort']},onChange}}/>);
 expect(screen.getByLabelText('Reasoning effort')).toHaveValue('');
 expect(screen.getByLabelText('Top p')).toBeDisabled();
 fireEvent.change(screen.getByLabelText('Temperature'),{target:{value:''}});
 expect(onChange.mock.calls.at(-1)[0].sampling).not.toHaveProperty('temperature');
});
it('maps sampling and reasoning inputs to their distinct config sections',()=>{
 const onChange=vi.fn(),config=hydrateConfig();
 const flags=['--temp','--top-p','--top-k','--min-p','--presence-penalty','--repeat-penalty','--jinja','--reasoning-effort','--reasoning-preserve'];
 render(<SamplingNode data={{config,caps:{flags},onChange}}/>);
 for(const [label,key,value] of [['Temperature','temperature',0.7],['Top p','top_p',0.8],['Top k','top_k',30],['Min p','min_p',0.1],['Presence penalty','presence_penalty',1.2],['Repeat penalty','repeat_penalty',1.1]]){
  fireEvent.change(screen.getByLabelText(label),{target:{value:String(value)}});
  expect(onChange.mock.calls.at(-1)[0].sampling[key]).toBe(value);
 }
 fireEvent.click(screen.getByLabelText('Jinja chat template'));
 expect(onChange.mock.calls.at(-1)[0].reasoning.jinja).toBe(true);
 fireEvent.change(screen.getByLabelText('Reasoning effort'),{target:{value:'high'}});
 expect(onChange.mock.calls.at(-1)[0].reasoning.effort).toBe('high');
 fireEvent.click(screen.getByLabelText('Preserve reasoning in conversation'));
 expect(onChange.mock.calls.at(-1)[0].reasoning.preserve).toBe(true);
});
it('shows the engine automatic GPU order when split lists are empty',()=>{
 const onChange=vi.fn();render(<SplitNode data={{config:hydrateConfig({placement_mode:'layer'}),caps,onChange}}/>);
 expect(screen.getByLabelText('Include CUDA0')).toBeChecked();
 expect(screen.getByLabelText('CUDA0 split ratio')).toHaveValue(1);
 fireEvent.change(screen.getByLabelText('CUDA0 split ratio'),{target:{value:'5'}});
 expect(onChange.mock.calls.at(-1)[0].split).toMatchObject({devices:['CUDA0','Vulkan1'],ratios:[5,1]});
});
it('edits ordered standard layer split without using duplicate physical devices',()=>{
 const onChange=vi.fn();const config=hydrateConfig({placement_mode:'layer',split:{devices:['CUDA0','Vulkan1'],ratios:[3,2]}});
 render(<SplitNode data={{config,caps,onChange}}/>);
 fireEvent.change(screen.getByLabelText('Vulkan1 split ratio'),{target:{value:'4'}});
 expect(onChange.mock.calls.at(-1)[0].split.ratios).toEqual([3,4]);
 fireEvent.click(screen.getByRole('button',{name:'Move Vulkan1 earlier'}));
 expect(onChange.mock.calls.at(-1)[0].split).toMatchObject({devices:['Vulkan1','CUDA0'],ratios:[2,3]});
 expect(screen.getByLabelText('Include Vulkan0')).toBeDisabled();
 fireEvent.change(screen.getByLabelText('Placement mode'),{target:{value:'tensor'}});
 expect(onChange.mock.calls.at(-1)[0].placement_mode).toBe('tensor');
});
