"""Run a competitor under a wall-clock and memory guard, and record what happened.

Every comparison in this project has to survive a baseline that dies. Twice now a
competitor's failure took down the harness that was supposed to report it: an
out-of-memory kill ended an SCS ladder with no row written, and a dense solve
had to be bounded by hand so it would not thrash the machine.

The rule this enforces: a competitor failing is a measurement, not an accident.
It runs the work in a child process so a kill cannot reach the parent, bounds it
in time and memory, and always returns a row.

    r = run_guarded(solve_it, args=(A, b), timeout_s=600, mem_gb=32, label="SCS")
    r["status"]   ok | timeout | out_of_memory | killed | failed:<Exc>
    r["seconds"]  wall time in the parent, INCLUDING spawn and argument pickling;
                  use it for bounding, not for a competitor's reported solve time.
                  Time the solve inside `target` and return it if the number will
                  be published, or large inputs will inflate the competitor.
    r["value"]    whatever the callable returned, on success

`mem_gb` sets RLIMIT_AS in the child, so allocation past it raises MemoryError
rather than inviting the kernel's OOM killer. Leave it None for GPU work, whose
runtimes reserve large virtual ranges and will trip an address-space cap.

The child is started with `spawn`, so it does not inherit a CUDA context from
the parent. That also means `target` must be importable in a fresh interpreter:
a module-level function, not a lambda or a closure. And the calling script must
protect its top level with `if __name__ == "__main__":`, or the spawned child
re-executes the script and the run reports `killed` instead of a result.

`mem_gb` caps the whole child, imports included, so it must leave room for the
interpreter and its libraries. A cap under about 2 GB makes NumPy fail to
import and you get `failed:ImportError` rather than `out_of_memory`. Size it as
the working set you want to allow plus that baseline.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import resource
import signal
import time
import traceback

__all__ = ["run_guarded", "GuardResult"]

GuardResult = dict


def _child(conn, target, args, kwargs, mem_gb):
    try:
        if mem_gb:
            nbytes = int(mem_gb * (1 << 30))
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        value = target(*args, **(kwargs or {}))
        conn.send(("ok", value))
    except MemoryError:
        conn.send(("out_of_memory", None))
    except Exception as exc:                       # noqa: BLE001
        conn.send((f"failed:{type(exc).__name__}", traceback.format_exc(limit=4)))
    finally:
        conn.close()


def run_guarded(target, args=(), kwargs=None, *, timeout_s, mem_gb=None,
                label="", verbose=True):
    """Run `target` in a child process under a time and memory guard."""
    ctx = mp.get_context("spawn")        # do not inherit a CUDA context
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, target, args, kwargs, mem_gb))
    t0 = time.perf_counter()
    proc.start()
    child.close()

    status, value = "timeout", None
    got = parent.poll(timeout_s)
    if got:
        try:
            status, value = parent.recv()
        except EOFError:
            status = "killed"

    if not got:
        # Out of time: stop it now rather than waiting on a join first.
        proc.terminate(); proc.join(timeout=10)
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL); proc.join()
    else:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.terminate(); proc.join(timeout=5)

    # A negative exit code means a signal. SIGKILL here is the kernel's
    # out-of-memory killer often enough that it is worth naming separately.
    if status in ("killed", "timeout") and proc.exitcode is not None:
        if proc.exitcode == -signal.SIGKILL:
            status = "killed_out_of_memory"

    seconds = time.perf_counter() - t0
    if verbose:
        note = "" if status == "ok" else f"  <- {status}"
        print(f"  [{label or target.__name__}] {seconds:8.3f}s {status}{note}",
              flush=True)
    return {"label": label or getattr(target, "__name__", "?"),
            "status": status, "seconds": round(seconds, 4),
            "value": value if status == "ok" else None,
            "detail": value if str(status).startswith("failed") else None,
            "timeout_s": timeout_s, "mem_gb": mem_gb}


def _t_fine(n):
    import numpy as np
    return float(np.ones(n).sum())


def _t_slow():
    time.sleep(30)


def _t_hog():
    import numpy as np
    return np.zeros((1 << 30,), dtype=np.float64)      # 8 GB


def _t_boom():
    return 1 / 0


if __name__ == "__main__":
    print("self-test: each case must return a row, never crash the parent")
    for r in (run_guarded(_t_fine, args=(1000,), timeout_s=30, label="ok-case"),
              run_guarded(_t_slow, timeout_s=2, label="timeout-case"),
              run_guarded(_t_hog, timeout_s=120, mem_gb=4, label="oom-case"),
              run_guarded(_t_boom, timeout_s=10, label="raise-case")):
        assert r["status"], r
    print("all four returned a row")
