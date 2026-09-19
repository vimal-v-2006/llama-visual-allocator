"""Real browser + HTTP + mixed-GPU inference acceptance check. No mock API."""
from pathlib import Path
import json
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from allocator.app import create_app
from playwright.sync_api import sync_playwright
import uvicorn

PORT = 18095
BASE = f'http://127.0.0.1:{PORT}'
MODEL = ROOT / 'tests/models/SmolLM2-135M-Instruct-Q4_K_M.gguf'
OUT = ROOT/'verification'

def main():
    app = create_app(port=PORT)
    nonce = app.state.launch_nonce
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=PORT, log_level='error', access_log=False))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic()+15
    while not server.started and thread.is_alive() and time.monotonic()<deadline:
        time.sleep(.05)
    assert server.started, 'Control plane failed to start'
    results = {}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
            page = browser.new_page(viewport={'width':1600,'height':1000})
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            assert page.request.get(BASE+'/api/state').status == 403
            assert page.request.get(BASE+'/api/state',headers={'Host':'evil.test'}).status == 403
            page.goto(BASE+'/#launch='+nonce)
            page.get_by_label('Model path',exact=True).wait_for()
            page.wait_for_function("() => document.querySelector('#model-path') && !document.querySelector('#model-path').disabled")
            assert '#launch' not in page.url
            assert browser.contexts[0].cookies() == [], 'Authentication must not create host cookies'
            def api(path,body=None):
                return page.evaluate('''async ({path,body}) => {
                    const r=await fetch('/api'+path,{method:body===null?'GET':'POST',headers:{'Content-Type':'application/json',Authorization:'Bearer '+sessionStorage.getItem('allocator-token')},...(body===null?{}:{body:JSON.stringify(body)})});
                    const data=await r.json();return {status:r.status,data};
                }''', {'path':path,'body':body})
            assert page.request.post(BASE+'/api/bootstrap',data={'nonce':nonce}).status == 403
            page.get_by_label('Model path',exact=True).fill(str(MODEL))
            with page.expect_response(lambda r:r.url.endswith('/api/model'),timeout=90000) as model_response:
                page.get_by_label('Load model',exact=True).click()
            loaded=model_response.value.json()
            assert model_response.value.status==200, loaded
            print('Model load:', loaded.get('plan',{}).get('errors'), flush=True)
            page.get_by_role('button',name='Auto Maximum Fit',exact=True).wait_for()
            page.wait_for_function("() => [...document.querySelectorAll('button')].some(b=>b.textContent==='Auto Maximum Fit'&&!b.disabled)",timeout=90000)
            state = api('/state')['data']
            assert state['model']['weight_bytes']==103668480
            assert state['model']['layer_count']==30
            assert state['config']['devices'], 'Load must initialize detected devices'
            page.get_by_role('button',name='Auto Maximum Fit',exact=True).click()
            page.wait_for_function("() => document.querySelector('[aria-label=\"Start server\"]') && !document.querySelector('[aria-label=\"Start server\"]').disabled",timeout=90000)
            state=api('/state')['data']
            oldkv=sum(m.get('kv',0) for m in state['plan']['memory'])
            page.get_by_role('button',name='Focus context',exact=True).click()
            page.get_by_label('Context tokens',exact=True).fill('8192',timeout=3000)
            page.get_by_label('Context tokens',exact=True).blur()
            deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                state=api('/state')['data']
                if state['config']['context']['size']==8192:break
                time.sleep(.1)
            assert sum(m.get('kv',0) for m in state['plan']['memory'])>oldkv, 'Context must update real KV budget'
            page.get_by_label('Focus model',exact=True).click()
            page.get_by_role('button',name='Expand all',exact=True).click()
            page.locator('.group-row input[type=checkbox]').first.check()
            page.get_by_label('Place selected groups',exact=True).select_option('Vulkan1')
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                manual=api('/state')['data']['config']
                if manual['locked']: break
                time.sleep(.1)
            assert manual['locked'] and manual['placements'][manual['locked'][0]]=='Vulkan1'
            pinned=manual['locked'][0]
            with page.expect_response(lambda r:r.url.endswith('/api/plan'),timeout=30000):
                page.get_by_role('button',name='Auto Balance',exact=True).click()
            state=api('/state')['data']
            assert state['config']['placements'][pinned]=='Vulkan1', 'Auto allocation must preserve manual pins'
            page.get_by_role('button',name='Collapse',exact=True).click()
            # Exercise save/download; the downloaded file is a real graph config.
            with page.expect_download() as d:
                page.get_by_label('Save workspace',exact=True).click()
            d.value.save_as(OUT/'browser-workspace.json')
            saved=json.loads((OUT/'browser-workspace.json').read_text())
            assert saved['model_path']==str(MODEL)
            # Compile a real mixed-device plan through the same authenticated control API.
            cfg=state['config']
            cfg['server']={'host':'127.0.0.1','port':18096,'allow_network':False}
            cfg['context'].update(size=1024,batch=128,ubatch=64,parallel=1)
            cfg['placement_mode']='tensor'
            groups=state['model']['groups']
            for g in groups:
                cfg['placements'][g['id']]='Vulkan1' if g['layer'] is not None and g['layer']>=15 else 'CUDA0'
            cfg['locked']=list(cfg['placements'])
            plan=api('/plan',{'config':cfg})
            assert plan['status']==200, plan
            assert plan['data']['launchable'], plan['data'].get('errors')
            imported_path=OUT/'mixed-workspace.json'
            imported_path.write_text(json.dumps(plan['data']['config']))
            with page.expect_response(lambda r:r.url.endswith('/api/plan'),timeout=60000):
                page.locator('input[type=file]').set_input_files(str(imported_path))
            page.wait_for_function("() => !document.querySelector('[aria-label=\"Start server\"]').disabled",timeout=30000)
            with page.expect_response(lambda r:r.url.endswith('/api/start'),timeout=60000) as started:
                page.get_by_label('Start server',exact=True).click()
            start={'status':started.value.status,'data':started.value.json()}
            assert start['status']==200, start
            deadline=time.monotonic()+90
            runtime={}
            while time.monotonic()<deadline:
                runtime=api('/runtime')['data']
                if runtime['health']=='healthy':break
                if not runtime['running']:raise AssertionError(runtime['logs'][-20:])
                time.sleep(.25)
            assert runtime['health']=='healthy', runtime
            req=urllib.request.Request('http://127.0.0.1:18096/completion',json.dumps({'prompt':'The capital of France is','n_predict':12,'temperature':0}).encode(),{'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=45) as r: completion=json.load(r)
            assert completion.get('content') and completion.get('timings'), completion
            runtime=api('/runtime')['data']
            results.update(completion=completion['content'],timings=completion['timings'],runtime={k:runtime[k] for k in ['health','rss_bytes','prompt_tps','generation_tps','gpu_process_bytes','gpu_global']})
            with page.expect_response(lambda r:r.url.endswith('/api/stop'),timeout=30000) as stopped:
                page.get_by_label('Stop server',exact=True).click()
            stop={'status':stopped.value.status,'data':stopped.value.json()}
            assert stop['status']==200 and not stop['data']['running'],stop
            assert stop['data']['rss_bytes']==0
            page.get_by_label('Close runtime drawer',exact=True).click()
            page.get_by_role('button',name='Overview',exact=True).click()
            page.wait_for_timeout(300)
            page.screenshot(path=str(OUT/'workspace-desktop.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Desktop horizontal overflow'
            page.set_viewport_size({'width':900,'height':900})
            page.screenshot(path=str(OUT/'workspace-narrow.png'),full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Narrow horizontal overflow'
            assert not errors, errors
            results.update(browser_errors=errors,security='unauthenticated and hostile Host rejected; nonce one-use; no cookies',model_tensors=state['model']['tensors'].__len__(),passed=True)
            browser.close()
    finally:
        app.state.runtime.stop()
        server.should_exit=True
        thread.join(timeout=15)
    (OUT/'workspace-acceptance.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
