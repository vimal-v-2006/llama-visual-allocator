"""Stable-handle lifecycle management; no shell or user-supplied freeform args."""
from collections import deque
import json
import os
from pathlib import Path
import re
import selectors
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.request


from .supervisor import pidfd_open


def executable_fd(path):
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ValueError('Executable must be an absolute path')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    info = os.fstat(fd)
    private_group = False
    if info.st_mode & 0o020 and info.st_uid == os.getuid():
        import grp
        import pwd
        try:
            owner = pwd.getpwuid(info.st_uid).pw_name
            group = grp.getgrgid(info.st_gid)
            members = set(group.gr_mem) | {u.pw_name for u in pwd.getpwall() if u.pw_gid == info.st_gid}
            private_group = group.gr_name == owner and members <= {owner, 'root'}
        except (KeyError, OSError):
            pass
    if (not stat.S_ISREG(info.st_mode) or info.st_uid not in (0, os.getuid()) or
            info.st_mode & 0o002 or (info.st_mode & 0o020 and not private_group) or not os.access(path, os.X_OK)):
        os.close(fd)
        raise ValueError('Executable must be owned by you/root, regular, executable, and not writable by untrusted users')
    return fd


def validate_executable(path):
    fd = executable_fd(path)
    os.close(fd)
    return path


def clean_environment(overrides=None):
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict) or any(k != 'GGML_CUDA_DISABLE_GRAPHS' or v != '1' for k, v in overrides.items()):
        raise ValueError('Unsupported managed environment override')
    return {**{k: v for k, v in os.environ.items() if k in ('PATH','HOME','USER','LANG','LC_ALL','TMPDIR','XDG_RUNTIME_DIR')}, **overrides}


def parse_nvidia_telemetry(global_rows, process_rows, pid):
    import csv
    result = {'gpu_process_bytes': {}, 'gpu_global': []}
    for row in csv.reader(global_rows.splitlines()):
        try:
            uuid, name, used, total = [s.strip() for s in row]
            result['gpu_global'].append(dict(id=uuid, name=name, used_bytes=int(used)*1024**2, total_bytes=int(total)*1024**2))
        except (ValueError, TypeError):
            continue
    for row in csv.reader(process_rows.splitlines()):
        try:
            target, uuid, used = [s.strip() for s in row]
            if pid is not None and int(target) == pid:
                result['gpu_process_bytes'][uuid] = result['gpu_process_bytes'].get(uuid, 0)+int(used)*1024**2
        except (ValueError, TypeError):
            continue
    return result


def parse_drm_telemetry(texts):
    result, seen = {}, set()
    for text in texts:
        fields = dict(line.split(':',1) for line in text.splitlines() if ':' in line)
        fields = {k:v.strip() for k,v in fields.items()}
        device, client = fields.get('drm-pdev'), fields.get('drm-client-id')
        match = re.fullmatch(r'(\d+)\s+(KiB|MiB|bytes)', fields.get('drm-memory-vram',''))
        if not device or not client or not match or (device,client) in seen:
            continue
        seen.add((device,client))
        result[device] = result.get(device,0)+int(match[1])*{'KiB':1024,'MiB':1024**2,'bytes':1}[match[2]]
    return result


def sample_gpu_telemetry(pid):
    result = {'gpu_process_bytes':{}, 'gpu_global':[], 'gpu_telemetry_warnings':[]}
    try:
        def query(fields):
            return subprocess.run(['nvidia-smi', fields, '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=1, check=True, env=clean_environment()).stdout
        global_rows = query('--query-gpu=uuid,name,memory.used,memory.total')
        rows = query('--query-compute-apps=pid,gpu_uuid,used_gpu_memory') if pid else ''
        result.update(parse_nvidia_telemetry(global_rows, rows, pid))
    except (OSError, subprocess.SubprocessError):
        result['gpu_telemetry_warnings'].append('NVIDIA telemetry unavailable')
    if pid:
        try:
            texts = []
            for fd in Path(f'/proc/{pid}/fdinfo').iterdir():
                try: texts.append(fd.read_text())
                except OSError: pass
            drm = parse_drm_telemetry(texts)
            result['gpu_process_bytes'].update(drm)
            if not drm:
                result['gpu_telemetry_warnings'].append('DRM/AMD process VRAM unavailable (no fdinfo counters)')
        except OSError:
            result['gpu_telemetry_warnings'].append('DRM/AMD process VRAM unavailable')
    return result


class Runtime:
    def __init__(self):
        self.lock = threading.RLock()
        self.proc = None
        self.pidfd = None
        self.control = None
        self.reader = None
        self.pid = None
        self.returncode = None
        self.started = None
        self.logs = deque(maxlen=400)
        self.server = None
        self._gpu_cache = None
        self._gpu_time = 0
        self._gpu_identity = None
        self.prompt_tps = self.generation_tps = self.acceptance = None

    def start(self, argv, server, env=None):
        with self.lock:
            self._refresh()
            if self.proc is not None:
                raise ValueError('A server is already running')
            if server.get('host') not in ('127.0.0.1','0.0.0.0') or (server.get('host') == '0.0.0.0' and server.get('allow_network') is not True):
                raise ValueError('Network binding requires explicit allow_network true')
            if type(server.get('port')) is not int or not 1024 <= server['port'] <= 65535:
                raise ValueError('Invalid managed port')
            environment = clean_environment(env)
            # Preflight stable handles before any managed process exists.
            test_fd = pidfd_open(os.getpid())
            os.close(test_fd)
            # llama-server has no inherited-listener contract: fail early on busy ports.
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind((server['host'], server['port']))
            fd = executable_fd(argv[0])
            read_control, write_control = os.pipe()
            read_status, write_status = os.pipe()
            try:
                self.proc = subprocess.Popen([sys.executable, '-I', str(Path(__file__).with_name('supervisor.py')),
                    str(read_control), str(write_status), str(fd), *argv], pass_fds=(read_control,write_status,fd),
                    env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, close_fds=True)
                self.pidfd = pidfd_open(self.proc.pid)
                os.write(write_control, b'G')
            except Exception:
                os.close(write_control)
                os.close(read_status)
                if self.proc is not None:
                    # Popen still owns an unreaped direct child, so kill is anchored.
                    self.proc.kill()
                    self.proc.wait()
                    self.proc.stdout.close()
                    self.proc = None
                raise
            finally:
                os.close(fd)
                os.close(read_control)
                os.close(write_status)
            self.control = write_control
            self.pid = None
            self.returncode = None
            self.started = time.monotonic()
            self.server = dict(server)
            self.logs.clear()
            self.prompt_tps = self.generation_tps = self.acceptance = None
            self.reader = threading.Thread(target=self._read, args=(self.proc,read_status), daemon=True)
            self.reader.start()
            return self.snapshot()

    def _read(self, process, status):
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, 'log')
        selector.register(status, selectors.EVENT_READ, 'status')
        buffers = {'log':b'', 'status':b''}
        try:
            while selector.get_map():
                for key, _ in selector.select(.2):
                    chunk = os.read(key.fd, 4096)
                    kind = key.data
                    if not chunk:
                        selector.unregister(key.fileobj)
                        if buffers[kind] and kind == 'log':
                            self._log(buffers[kind].decode(errors='replace'))
                        continue
                    buffers[kind] += chunk
                    while b'\n' in buffers[kind]:
                        line, buffers[kind] = buffers[kind].split(b'\n', 1)
                        if kind == 'log':
                            self._log(line.decode(errors='replace'))
                        else:
                            data = json.loads(line)
                            if 'pid' in data:
                                self.pid = data['pid']
                            if 'returncode' in data:
                                self.returncode = data['returncode']
                            if 'error' in data:
                                self._log(data['error'])
                    if len(buffers[kind]) > 8192:
                        if kind == 'log':
                            self._log(buffers[kind][:8192].decode(errors='replace'))
                        buffers[kind] = b''
        finally:
            selector.close()
            os.close(status)
            process.stdout.close()

    def _log(self, line):
        line = re.sub(r'(?i)(bearer\s+|(?:api[_-]?key|token|password|secret|nonce)\s*[=:]\s*)\S+', r'\1[REDACTED]', line)
        self.logs.append(line[:2048])
        match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s+tokens per second', line)
        if match:
            if 'prompt eval' in line:
                self.prompt_tps = float(match[1])
            elif 'eval time' in line:
                self.generation_tps = float(match[1])
        match = re.search(r'acceptance(?: rate)?\s*[=:]\s*([0-9.]+)%', line, re.I)
        if match:
            self.acceptance = float(match[1])/100

    def _refresh(self):
        if self.proc is not None and self.proc.poll() is not None:
            self.reader.join(timeout=1)
            if self.returncode is None:
                self.returncode = self.proc.returncode
            os.close(self.pidfd)
            os.close(self.control)
            self.proc = self.pidfd = self.control = self.pid = None

    def stop(self):
        with self.lock:
            self._refresh()
            if self.proc is not None:
                try:
                    os.write(self.control, b'S')
                except BrokenPipeError:
                    pass
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Do not kill the subreaper and orphan descendants: retain ownership.
                    raise RuntimeError('Supervisor cleanup did not complete; ownership retained')
                self._refresh()
            return self.snapshot()

    def _owns_listener(self):
        if not self.pid or not self.proc:
            return False
        try:
            fields = Path(f'/proc/{self.pid}/stat').read_text().rsplit(')',1)[1].split()
            if int(fields[1]) != self.proc.pid:
                return False
            sockets = set()
            for fd in Path(f'/proc/{self.pid}/fd').iterdir():
                try: sockets.add(os.readlink(fd))
                except OSError: pass
            for line in Path(f'/proc/{self.pid}/net/tcp').read_text().splitlines()[1:]:
                row = line.split()
                address, port = row[1].split(':')
                if row[3] == '0A' and int(port,16) == self.server['port'] and address in ('0100007F','00000000') and f'socket:[{row[9]}]' in sockets:
                    return True
        except (OSError, ValueError, IndexError):
            pass
        return False

    def snapshot(self):
        with self.lock:
            self._refresh()
            running = self.proc is not None
            rss = 0
            if running and self.pid:
                try:
                    rss = int(Path(f'/proc/{self.pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                except (OSError, ValueError, IndexError):
                    pass
            health = 'stopped'
            if running:
                health = 'starting'
                try:
                    if not self._owns_listener():
                        raise OSError('Managed process has no owned listener')
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(f"http://127.0.0.1:{self.server['port']}/health", timeout=.3) as response:
                        health = 'healthy' if response.status == 200 and self._owns_listener() else 'unhealthy'
                except Exception:
                    health = 'unreachable' if time.monotonic()-self.started > 10 else 'starting'
            # Only attribute telemetry while /proc proves this is our supervisor's child.
            def owned_identity():
                if not running or not self.pid:
                    return None
                try:
                    fields = Path(f'/proc/{self.pid}/stat').read_text().rsplit(')',1)[1].split()
                    return (self.pid, fields[19]) if int(fields[1]) == self.proc.pid else None
                except (OSError, ValueError, IndexError):
                    return None
            identity = owned_identity()
            if self._gpu_cache is None or identity != self._gpu_identity or time.monotonic()-self._gpu_time >= 2:
                self._gpu_cache = sample_gpu_telemetry(identity[0] if identity else None)
                if owned_identity() != identity:
                    self._gpu_cache['gpu_process_bytes'] = {}
                self._gpu_time = time.monotonic()
                self._gpu_identity = identity
            gpu = dict(self._gpu_cache)
            if not identity:
                gpu['gpu_process_bytes'] = {}
            return {'running':running, 'pid':self.pid if running else None,
                    'returncode':self.returncode, 'health':health,
                    'logs':list(self.logs), 'rss_bytes':rss, 'prompt_tps':self.prompt_tps,
                    'generation_tps':self.generation_tps, 'acceptance':self.acceptance,
                    **gpu,
                    'uptime':time.monotonic()-self.started if running else 0}
