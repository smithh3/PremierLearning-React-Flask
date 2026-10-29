"""Spawn child worker processes that reliably die with their parent.

Windows has no process groups that survive a hard kill, so a parent that is
taskkill'd, or stopped from the IDE, leaves its children running. The fix is a
job object with KILL_ON_JOB_CLOSE: this process joins it, descendants inherit
membership, and the job kills whatever is left as soon as our last handle
closes, however this process goes away.

simulation/sweep.py keeps its own copy of this logic because its workers spawn
grandchildren (cbc.exe) and it patches CreateProcess around the spawn; its tests
also patch these names in that module. This module is the plain version, for
workers that are just Python.
"""
import os
import subprocess
import sys
from pathlib import Path


JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

_JOB_HANDLE = None
_JOB_HAS_SELF = False
# Children we spawned, so Ctrl+C or a clean exit does not strand them.
LIVE_WORKERS = set()


def venv_python():
    """The venv interpreter, not the system Python that created it."""
    prefix = Path(sys.prefix)
    if os.name == 'nt':
        candidates = (prefix / 'Scripts' / 'python.exe', prefix / 'python.exe')
    else:
        candidates = (prefix / 'bin' / 'python', prefix / 'bin' / 'python3')
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _looks_like_debugger(executable):
    text = str(executable or '').replace('\\', '/').lower()
    return 'debugpy' in text or 'pydevd' in text


def worker_python():
    """Interpreter for children: never the debugger stub, never system Python."""
    found = venv_python()
    if found:
        return found
    executable = sys.executable
    if executable and not _looks_like_debugger(executable):
        return executable
    return getattr(sys, '_base_executable', None) or executable


def worker_env(extra=None):
    """A clean environment: no debugger hooks inherited into the child.

    Launching the parent under the IDE debugger otherwise drags debugpy into
    every child, which slows them down and can make them hang on exit.
    """
    env = os.environ.copy()
    for key in list(env):
        upper = key.upper()
        if upper.startswith('PYDEVD') or upper.startswith('DEBUGPY'):
            env.pop(key, None)
    env.pop('PYTHONHOME', None)
    pythonpath = [
        part
        for part in env.get('PYTHONPATH', '').split(os.pathsep)
        if part
        and 'debugpy' not in part.lower()
        and 'pydevd' not in part.lower()
    ]
    if pythonpath:
        env['PYTHONPATH'] = os.pathsep.join(pythonpath)
    else:
        env.pop('PYTHONPATH', None)
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    env['VIRTUAL_ENV'] = sys.prefix
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def kill_with_this_process():
    """Join a job object whose members are killed when this process exits."""
    global _JOB_HANDLE, _JOB_HAS_SELF
    if os.name != 'nt' or _JOB_HANDLE is not None:
        return _JOB_HANDLE
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    # Explicit prototypes: without them handles pass as 32-bit ints on x64 and
    # the GetCurrentProcess pseudo-handle arrives truncated, so joining the job
    # fails with ERROR_INVALID_HANDLE.
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ('ReadOperationCount', ctypes.c_ulonglong),
            ('WriteOperationCount', ctypes.c_ulonglong),
            ('OtherOperationCount', ctypes.c_ulonglong),
            ('ReadTransferCount', ctypes.c_ulonglong),
            ('WriteTransferCount', ctypes.c_ulonglong),
            ('OtherTransferCount', ctypes.c_ulonglong),
        ]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', wintypes.LARGE_INTEGER),
            ('PerJobUserTimeLimit', wintypes.LARGE_INTEGER),
            ('LimitFlags', wintypes.DWORD),
            ('MinimumWorkingSetSize', ctypes.c_size_t),
            ('MaximumWorkingSetSize', ctypes.c_size_t),
            ('ActiveProcessLimit', wintypes.DWORD),
            ('Affinity', ctypes.c_size_t),
            ('PriorityClass', wintypes.DWORD),
            ('SchedulingClass', wintypes.DWORD),
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', _BasicLimits),
            ('IoInfo', _IoCounters),
            ('ProcessMemoryLimit', ctypes.c_size_t),
            ('JobMemoryLimit', ctypes.c_size_t),
            ('PeakProcessMemoryUsed', ctypes.c_size_t),
            ('PeakJobMemoryUsed', ctypes.c_size_t),
        ]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    extended_limit_information = 9
    if not kernel32.SetInformationJobObject(
        job, extended_limit_information, ctypes.byref(limits), ctypes.sizeof(limits)
    ):
        kernel32.CloseHandle(job)
        return None
    _JOB_HANDLE = job
    # Joining ourselves means descendants inherit membership for free. If the
    # surrounding job forbids nesting, fall back to adding each worker.
    _JOB_HAS_SELF = bool(
        kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())
    )
    return job


def add_worker_to_job(proc):
    if os.name != 'nt' or _JOB_HANDLE is None or _JOB_HAS_SELF:
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    process_set_quota_and_terminate = 0x0100 | 0x0001
    handle = kernel32.OpenProcess(process_set_quota_and_terminate, False, proc.pid)
    if not handle:
        return
    try:
        kernel32.AssignProcessToJobObject(_JOB_HANDLE, handle)
    finally:
        kernel32.CloseHandle(handle)


def kill_live_workers():
    """Stop tracked children on Ctrl+C or a clean exit.

    The job object covers hard kills of this process; this covers the paths
    where we still get to run code. taskkill takes the tree, since a child may
    have started under a launcher stub that proc.kill() would not reach past.
    """
    for proc in list(LIVE_WORKERS):
        LIVE_WORKERS.discard(proc)
        try:
            if proc.poll() is not None:
                continue
            if os.name == 'nt':
                subprocess.run(
                    ['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                    capture_output=True,
                )
            else:
                proc.kill()
        except Exception:
            continue


def popen_worker(cmd, env=None, cwd=None, stdout=None):
    """Start a child that will not outlive us."""
    kwargs = dict(env=env or worker_env(), cwd=cwd)
    if stdout is not None:
        kwargs['stdout'] = stdout
        kwargs['stderr'] = subprocess.STDOUT
    if os.name != 'nt':
        proc = subprocess.Popen(cmd, **kwargs)
        LIVE_WORKERS.add(proc)
        return proc
    kill_with_this_process()
    # No CREATE_BREAKAWAY_FROM_JOB: leaving the job is what strands children.
    new_group = getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x00000200)
    proc = subprocess.Popen(cmd, creationflags=new_group, **kwargs)
    add_worker_to_job(proc)
    LIVE_WORKERS.add(proc)
    return proc


def thread_limit_env(workers):
    """Keep N TensorFlow processes from each grabbing every core.

    Left alone, every worker sizes its thread pools to the whole machine and
    they spend their time fighting over cores rather than fitting. Splitting the
    cores evenly is the useful default; reserving one for the parent is not
    worth it, since the parent only waits on children.
    """
    workers = max(1, int(workers))
    try:
        cores = os.cpu_count() or workers
    except NotImplementedError:
        cores = workers
    per_worker = max(1, cores // workers)
    return {
        'OMP_NUM_THREADS': per_worker,
        'TF_NUM_INTRAOP_THREADS': per_worker,
        'TF_NUM_INTEROP_THREADS': 1,
    }
