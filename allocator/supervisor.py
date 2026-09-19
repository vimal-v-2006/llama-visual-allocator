"""Private Linux subreaper. Control-pipe EOF also means stop.

Only signal currently owned direct children via pidfds; killing parents causes
setsid/double-fork descendants to be adopted here, independent of process groups.
"""
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


def pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    return _syscall(434, pid, 0)


def pidfd_signal(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    return _syscall(424, fd, sig, 0, 0)


def _syscall(number, *args):
    import platform
    if platform.machine() not in ('x86_64', 'aarch64'):
        raise RuntimeError('Stable process handles unsupported on this architecture')
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(x) for x in args))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def children():
    return [int(x) for x in Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()]


def signal_children(sig):
    for pid in children():
        fd = None
        try:
            fd = pidfd_open(pid)
            # Recheck ownership after obtaining the stable handle. An unreaped
            # child cannot have its PID reused, and we alone reap our children.
            if pid in children():
                pidfd_signal(fd, sig)
        except ProcessLookupError:
            pass
        finally:
            if fd is not None:
                os.close(fd)


def reap():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if not pid:
                return
        except ChildProcessError:
            return


def main():
    control, status, executable_fd = map(int, sys.argv[1:4])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'subreaper unavailable')
    def emit(data):
        os.write(status, (json.dumps(data)+'\n').encode())
    child = None
    # Parent grants spawn only after stable supervisor ownership is established.
    if os.read(control, 1) != b'G':
        return
    try:
        child = subprocess.Popen(sys.argv[4:], executable=f'/proc/self/fd/{executable_fd}',
                                 pass_fds=(executable_fd,), stdin=subprocess.DEVNULL,
                                 start_new_session=True, close_fds=True)
        os.close(executable_fd)
        emit({'pid':child.pid})
        while child.poll() is None:
            ready, _, _ = select.select([control], [], [], .1)
            if ready:
                os.read(control, 1)
                break
    except Exception:
        emit({'error':'Server spawn failed'})
    finally:
        # TERM all current children, including descendants adopted after parent exit.
        deadline = time.monotonic()+2
        while children():
            signal_children(signal.SIGTERM if time.monotonic()<deadline else signal.SIGKILL)
            if child is not None:
                child.poll()
            reap()
            time.sleep(.03)
        emit({'returncode': child.returncode if child is not None else -1})
        os.close(status)
        os.close(control)


if __name__ == '__main__':
    main()
