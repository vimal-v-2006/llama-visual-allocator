"""Loopback-only authenticated control plane."""
import secrets
import copy
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from .runtime import Runtime, validate_executable


def create_app(port=8095, engine_module=None, static_dir=None):
    runtime_manager = Runtime()
    @asynccontextmanager
    async def lifespan(app):
        yield
        runtime_manager.stop()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.runtime = runtime_manager
    operation_lock = threading.RLock()
    state = {'config': None, 'capabilities': None, 'model': None, 'plan': None}
    def engine():
        nonlocal engine_module
        if engine_module is None:
            from . import engine as loaded
            engine_module = loaded
        return engine_module
    def config_init():
        if state['config'] is None:
            state['config'] = engine().default_config()
    def check_config(config):
        if not isinstance(config, dict):
            raise ValueError('config must be an object')
        allowed = {'executable','model_path','devices','placements','locked','context','spec','server','overhead_mib','compute_mib','mmap','positions','edges','placement_mode','split','sampling','reasoning','vision','environment','acknowledge_estimate'}
        if set(config)-allowed:
            raise ValueError('Unknown configuration fields; arbitrary arguments are forbidden')
        server = config.get('server', {})
        if (not isinstance(server, dict) or server.get('host') not in ('127.0.0.1','0.0.0.0') or
                (server.get('host') == '0.0.0.0' and server.get('allow_network') is not True)):
            raise ValueError('Network binding requires explicit server.allow_network true')
        target_port = server.get('port')
        if type(target_port) is not int or not 1024 <= target_port <= 65535 or target_port == port:
            raise ValueError('Invalid or conflicting managed server port')
        validate_executable(config.get('executable'))
        return config
    cache = {'caps': None, 'model': None}
    def identity(paths):
        try:
            return tuple((str(p), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                         for p in paths for s in [Path(p).stat()])
        except OSError:
            return None  # Missing inputs are never cacheable; engine reports the error.
    def cached(kind, path, loader, fresh=False):
        entry = cache[kind]
        if not fresh and entry and entry[0] == path and entry[2] is not None and identity(entry[1]) == entry[2]:
            return copy.deepcopy(entry[3])
        value = loader(path)
        paths = value.get('shards', [path]) if kind == 'model' else [path]
        cache[kind] = (path, paths, identity(paths), copy.deepcopy(value))
        return value
    def evaluate(config, mode=None, fresh=False, initialize_devices=False):
        config = copy.deepcopy(check_config(config))
        caps = cached('caps', config['executable'], engine().probe, fresh)
        if initialize_devices:
            defaults = engine().default_config(caps).get('devices', {})
            existing = config.get('devices', {})
            config['devices'] = {key: {**value, **existing.get(key, {})} for key, value in defaults.items()}
        model = cached('model', config['model_path'], engine().read_model, fresh)
        result = engine().plan(model, caps, config, mode=mode)
        if config['server']['host'] == '0.0.0.0':
            result.setdefault('warnings', []).append('Network exposure: managed inference server listens on all IPv4 interfaces; protect it with a firewall/authentication. Control plane remains loopback-only.')
        state.update(config=result.get('config',config), capabilities=caps, model=model, plan=result)
        return result
    app.state.launch_nonce = secrets.token_urlsafe(32)
    tokens = set()
    lock = threading.Lock()

    @app.middleware('http')
    async def boundary(request: Request, call_next):
        headers = request.headers
        if (headers.get('host') != f'127.0.0.1:{port}' or
                headers.get('origin') not in (None, f'http://127.0.0.1:{port}') or
                headers.get('sec-fetch-site') not in (None, 'same-origin', 'none')):
            return JSONResponse({'detail': 'Invalid request origin'}, status_code=403)
        # Reject ambiguous duplicate security headers instead of trusting a proxy/parser choice.
        for key in ('host', 'origin', 'authorization', 'content-length', 'sec-fetch-site'):
            if len(headers.getlist(key)) > 1:
                return JSONResponse({'detail': 'Duplicate header'}, status_code=400)
        if request.method in ('POST', 'PUT', 'PATCH', 'DELETE'):
            if headers.get('content-type', '').split(';')[0].strip().lower() != 'application/json':
                return JSONResponse({'detail': 'JSON required'}, status_code=415)
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 65536:
                    return JSONResponse({'detail': 'Body too large'}, status_code=413)
            import json
            try:
                data = json.loads(body, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
                if not isinstance(data, dict):
                    raise ValueError()
            except (ValueError, RecursionError):
                return JSONResponse({'detail': 'Invalid JSON object'}, status_code=400)
            request.state.data = data
        if request.url.path.startswith('/api/') and request.url.path != '/api/bootstrap':
            with lock:
                allowed = request.headers.get('authorization', '').removeprefix('Bearer ') in tokens
            if not allowed:
                return JSONResponse({'detail': 'Unauthorized'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
        return response

    @app.post('/api/bootstrap')
    async def bootstrap(request: Request):
        data = request.state.data
        with lock:
            if not app.state.launch_nonce or data.get('nonce') != app.state.launch_nonce:
                return JSONResponse({'detail': 'Invalid launch nonce'}, status_code=403)
            token = secrets.token_urlsafe(32)
            tokens.add(token)
            app.state.launch_nonce = None
        return {'token': token}

    @app.get('/api/runtime')
    def runtime():
        return runtime_manager.snapshot()

    @app.get('/api/state')
    def get_state():
        with operation_lock:
            config_init()
            return {**state, 'runtime':runtime_manager.snapshot()}

    @app.post('/api/{action}')
    def mutate(action: str, request: Request):
        data = request.state.data
        with operation_lock:
            try:
                config_init()
                if action == 'stop':
                    return runtime_manager.stop()
                if action == 'probe':
                    path = validate_executable(data.get('executable'))
                    caps = engine().probe(path)
                    config = engine().default_config(caps)
                    config['executable'] = path
                    state.update(capabilities=caps, config=config, model=None, plan=None)
                    return {'capabilities':caps,'config':config}
                if action == 'model':
                    config = copy.deepcopy(state['config'])
                    if config.get('model_path') != data['path']:
                        config.update(placements={}, locked=[], positions={}, edges=[])
                    config['model_path'] = data['path']
                    if data.get('executable'):
                        config['executable'] = data['executable']
                    evaluate(config, fresh=True, initialize_devices=True)
                    return {key:state[key] for key in ('model','capabilities','config','plan')}
                if action in ('plan','start'):
                    if data.get('mode') not in (None,'maximum','balance'):
                        raise ValueError('Invalid allocation mode')
                    result = evaluate(data['config'], data.get('mode') if action == 'plan' else None, fresh=action == 'start')
                    if action == 'plan':
                        return result
                    acknowledged = result.get('status') == 'UNKNOWN' and data['config'].get('acknowledge_estimate') is True
                    if not result.get('launchable') or (result.get('status') not in ('FIT','NEAR LIMIT') and not acknowledged) or result.get('errors') != []:
                        raise HTTPException(409, 'Plan is not safe to launch; inspect plan errors and memory status')
                    return runtime_manager.start(result['argv'], state['config']['server'], env=result.get('env', {}))
                raise HTTPException(404, 'Unknown API endpoint')
            except HTTPException:
                raise
            except (ValueError, TypeError, KeyError, OSError) as exc:
                raise HTTPException(400, str(exc)[:300]) from None

    root = Path(static_dir or Path(__file__).resolve().parent.parent/'frontend'/'dist').resolve()
    @app.get('/{asset:path}')
    def static(asset: str):
        target = root/(asset or 'index.html')
        if '..' in Path(asset).parts or target.is_symlink():
            raise HTTPException(404, 'Not found')
        target = target.resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise HTTPException(404, 'UI not built or asset not found')
        return FileResponse(target)

    return app


def main():
    import argparse
    import socket
    import webbrowser
    import uvicorn
    parser = argparse.ArgumentParser(description='Local llama.cpp Visual Allocator')
    parser.add_argument('--port', type=int, default=8095)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be between 1024 and 65535')
    # Bind before displaying authority or opening a browser: no race to a foreign listener.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', args.port))
    sock.listen(128)
    app = create_app(port=args.port)
    launch_url = f'http://127.0.0.1:{args.port}/#launch={app.state.launch_nonce}'
    print(launch_url, flush=True)  # Deliberate one-time authority handoff, never generic logs.
    if not args.no_browser:
        webbrowser.open(launch_url)
    config = uvicorn.Config(app, host='127.0.0.1', port=args.port, access_log=False, proxy_headers=False, log_level='warning')
    try:
        uvicorn.Server(config).run(sockets=[sock])
    finally:
        app.state.runtime.stop()
        sock.close()


if __name__ == '__main__':
    main()

