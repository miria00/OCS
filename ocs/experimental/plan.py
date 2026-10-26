"""Optional, measured compile planning above the unchanged solver.

A plan chooses data layout and fixed-step unrolling, not a numerical answer.
The state/workspace estimate omits backend and equilibration temporaries: it
is a screening estimate, never a memory bound. GPU probes enforce a budget
against the device allocator's *process-lifetime* high-water mark. JAX does
not expose a portable reset for that mark; it is not a candidate-only peak.
Prior allocations can therefore conservatively reject an otherwise viable
candidate. Use a fresh process to remove that ambiguity.

Cache keys include source/runtime, placement, block/candidate policy, and
instance identity. Immutable JAX arrays have process-scoped weak-reference
tokens; their contents and operator entries are never read for hashing.
Mutable host arrays and opaque callbacks disable cache reuse. Consequently a
new process or new data objects may reprobe: safety takes priority over cache
hits. A cached choice is only a performance hint, rechecked against current
memory accounting. Streaming, residency and tiling are not implemented here.
"""
from __future__ import annotations

import dataclasses
import functools
import gc
import hashlib
import json
import marshal
import os
from pathlib import Path
import platform
import tempfile
import time
import types
import uuid
import weakref

import numpy as np

_STATE_ARRAYS_PRIMAL = 8
_STATE_ARRAYS_DUAL = 6
_HEADROOM = 0.85
_SCHEMA = 2
_PROCESS_TOKEN = uuid.uuid4().hex
_ARRAY_TOKENS = {}
_HARDWARE_KEYS = frozenset(("data_layout", "dual_layout", "loop_unroll"))


class PlanningError(RuntimeError):
    """No measured candidate can safely be recommended; inspect candidates."""

    def __init__(self, message, candidates=()):
        super().__init__(message)
        self.candidates = list(candidates)


class _Uncacheable(ValueError):
    pass


@dataclasses.dataclass
class Plan:
    device_kind: str
    device_memory_bytes: int
    key: str
    compile_kwargs: dict
    estimated_bytes: int
    fits_estimate: bool
    measured_peak_bytes: int | None = None
    seconds_per_block: float | None = None
    candidates: list = dataclasses.field(default_factory=list)
    note: str = ""
    created: str = ""
    ocs_version: str = ""
    schema: int = _SCHEMA
    device_identity: dict = dataclasses.field(default_factory=dict)
    cache_scope: str = "disabled"
    memory_scope: str = "unmeasured estimate"

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=1, allow_nan=False)

    @classmethod
    def from_json(cls, text):
        d = json.loads(text)
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: d[k] for k in d if k in fields})


def device_info(device=None):
    """Resolve a local JAX device; GPU memory never falls back to host RAM."""
    import jax
    dev = device if device is not None else jax.devices()[0]
    if dev not in jax.local_devices():
        raise ValueError("planner device must be a local JAX device")
    kind = str(getattr(dev, "device_kind", dev.platform))
    stats = _memory_stats(dev)
    limit = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
    if not limit and dev.platform == "cpu":
        try:
            limit = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            limit = 0
    return dev, kind, int(limit or 0)


def _memory_stats(dev):
    try:
        return dev.memory_stats() or {}
    except Exception:
        return {}


def _device_identity(dev):
    return {k: str(getattr(dev, k, "")) for k in
            ("platform", "device_kind", "id", "process_index", "uuid")}


def _runtime_identity():
    import jax
    import jaxlib
    # Runtime settings can affect compilation, precision, timing and memory.
    settings = {k: str(v) for k, v in jax.config.values.items()}
    env = {k: v for k, v in os.environ.items()
           if k.startswith(("JAX_", "XLA_", "CUDA_", "NVIDIA_"))}
    return {"python": platform.python_version(), "jax": jax.__version__,
            "jaxlib": jaxlib.__version__, "numpy": np.__version__,
            "config": settings, "env": env, "process": _PROCESS_TOKEN}


def _source_identity():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def _array_token(value):
    ident = id(value)
    old = _ARRAY_TOKENS.get(ident)
    if old is None or old[0]() is not value:
        token = uuid.uuid4().hex
        def remove(ref, ident=ident):
            entry = _ARRAY_TOKENS.get(ident)
            if entry is not None and entry[0] is ref:
                _ARRAY_TOKENS.pop(ident, None)
        _ARRAY_TOKENS[ident] = (weakref.ref(value, remove), token)
    return _ARRAY_TOKENS[ident][1]


def _fingerprint(value, active=None):
    """Metadata/identity only: never evaluate an oracle or read array values."""
    import jax
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return (str(value.dtype), value.item())
    if isinstance(value, jax.Array):
        return {"jax_array": [_PROCESS_TOKEN, _array_token(value)],
                "shape": list(value.shape), "dtype": str(value.dtype),
                "sharding": str(value.sharding)}
    if isinstance(value, np.ndarray):
        raise _Uncacheable("mutable NumPy array: no content hash or cache reuse")
    active = set() if active is None else active
    if id(value) in active:
        raise _Uncacheable("recursive callback or model state")
    active.add(id(value))
    try:
        child = lambda v: _fingerprint(v, active)
        if isinstance(value, (tuple, list)):
            return [type(value).__name__, [child(v) for v in value]]
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise _Uncacheable("non-string mapping keys")
            return {k: child(v) for k, v in sorted(value.items())}
        if isinstance(value, functools.partial):
            return [child(value.func), child(value.args), child(value.keywords)]
        if isinstance(value, types.FunctionType):
            # Non-OCS callbacks may read mutable globals, files, services, etc.
            # Do not infer a durable contract from their bytecode.
            if not value.__module__.startswith("ocs."):
                raise _Uncacheable("custom callback has no immutable cache contract")
            closure = [c.cell_contents for c in value.__closure__ or ()]
            return {"function": value.__qualname__, "module": value.__module__,
                    "code": hashlib.sha256(marshal.dumps(value.__code__)).hexdigest(),
                    "defaults": child(value.__defaults__),
                    "kwdefaults": child(value.__kwdefaults__), "closure": child(closure)}
        if type(value).__module__.startswith("ocs.") and hasattr(value, "__dict__"):
            methods = {k: getattr(type(value), k, None) for k in
                       ("matvec", "rmatvec", "prox", "prox_matrix", "smooth_grad",
                        "value", "conj_value")}
            if any(callable(v) and not getattr(v, "__module__", "").startswith("ocs.")
                   for v in methods.values()):
                raise _Uncacheable("custom class oracle has no immutable cache contract")
            return {"class": type(value).__module__ + "." + type(value).__qualname__,
                    "state": child(vars(value)),
                    "method_identities": {k: _array_token(v) if callable(v) else None
                                          for k, v in methods.items()},
                    "contracts": {k: child(getattr(value, k, None)) for k in
                                  ("scaling_class", "eval_kind", "coordinate_separable",
                                   "well_conditioned", "workspace_bytes")}}
        raise _Uncacheable(f"opaque {type(value).__name__}: cache reuse disabled")
    finally:
        active.remove(id(value))


def family_signature(family, fixed_kwargs):
    """Fingerprint known immutable model state without reading operator entries.

    Opaque or mutable models raise ValueError; plan_family still probes them,
    but deliberately does not persist/reuse their plans.
    """
    return json.dumps([_fingerprint(family), _fingerprint(fixed_kwargs)],
                      sort_keys=True, allow_nan=False)


def _array_metadata(value, dev, seen=None):
    """Check captured placements and conservatively discover dtype widths."""
    import jax
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 4
    seen.add(id(value))
    if isinstance(value, (jax.Array, np.ndarray)):
        if isinstance(value, jax.Array) and value.committed and value.devices() != {dev}:
            raise ValueError("family contains arrays on another device; rebuild it inside "
                             "jax.default_device(the_planned_device)")
        return max(4, np.dtype(value.dtype).itemsize)
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (tuple, list)):
        children = value
    elif isinstance(value, types.FunctionType):
        children = [value.__defaults__, value.__kwdefaults__]
        children += [c.cell_contents for c in value.__closure__ or ()]
    elif isinstance(value, functools.partial):
        children = [value.func, value.args, value.keywords]
    elif hasattr(value, "__dict__") and not isinstance(value, (type, types.ModuleType)):
        children = vars(value).values()
    else:
        children = ()
    return max([4] + [_array_metadata(v, dev, seen) for v in children])


def _term_bytes(term, itemsize=4):
    op = term.op
    m, n = (int(op.shape[0]), int(op.shape[1])) if getattr(op, "shape", None) else (0, 0)
    extra = getattr(op, "workspace_bytes", None)
    if extra is not None:
        try:
            return max(0, int(extra))
        except (TypeError, ValueError):
            pass
    name = type(op).__name__
    if "FFT" in name:
        return 2 * itemsize * (n + m)
    if hasattr(op, "A") and hasattr(op.A, "nbytes"):
        return int(op.A.nbytes) + (int(op.B.nbytes) if hasattr(op, "B") and hasattr(op.B, "nbytes") else 0)
    return itemsize * (m + n)


def estimate_bytes(family, block=100, data_layout="dense", itemsize=4):
    """Screening estimate only; not a conservative bound on peak memory."""
    n = int(family.n)
    m = sum(int(t.op.shape[0]) for t in family.terms if getattr(t.op, "shape", None))
    state = itemsize * (_STATE_ARRAYS_PRIMAL * n + _STATE_ARRAYS_DUAL * m)
    if data_layout == "dense":
        state += itemsize * m
    return int(state + sum(_term_bytes(t, itemsize) for t in family.terms))


def _cache_dir(cache_dir=None):
    d = cache_dir or os.environ.get("OCS_PLAN_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "ocs", "plans")
    os.makedirs(d, exist_ok=True)
    return d


def _probe(family, data, cand, fixed, block, dev):
    """Two-block pilot; peak is device process lifetime, not candidate-local."""
    import jax
    kwargs = dict(fixed, **cand)
    kwargs["block"] = block
    kwargs["max_blocks"] = 2
    comp = sol = placed = None
    try:
        with jax.default_device(dev):
            placed = jax.device_put(data, dev)
            comp = family.compile(**kwargs)
            for repeat in range(2):
                # Drop the first result before timing the second allocation.
                sol = None
                t0 = time.perf_counter()
                sol = comp.solve_instance(placed)
                jax.block_until_ready((sol.x, sol.y))
                if any(a.devices() != {dev} for a in (sol.x, sol.y)):
                    raise ValueError("probe output did not execute on the planned device")
                elapsed = time.perf_counter() - t0
            if not np.isfinite(float(sol.residual)) or str(sol.status) in (
                    "numerical_error", "error", "invalid"):
                raise ValueError("probe returned a nonfinite or numerical-error endpoint")
            seconds = elapsed / max(1.0, int(sol.iters) / block)
            peak = int(_memory_stats(dev).get("peak_bytes_in_use") or 0) or None
            return seconds, peak, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"[:300]
    finally:
        comp = sol = placed = None
        gc.collect()


def _memory_admissible(dev, budget, peak):
    if dev.platform == "cpu":
        return peak is None or budget is None or peak <= budget
    # Missing GPU memory accounting cannot establish budget admissibility.
    return budget is not None and peak is not None and peak <= budget


def plan_family(family, data=None, *, mode="auto", cache_dir=None, block=100,
                candidates=None, device=None, probe=True, **fixed_kwargs):
    """Estimate/probe layouts; never fall back to an unsuccessful candidate.

    Estimates alone are available with probe=False or data=None. They carry no
    measured-fit promise. Hardware candidates cannot override fixed choices.
    """
    import jax
    import ocs
    if mode not in ("auto", "cached"):
        raise ValueError("mode must be 'auto' or 'cached'")
    if isinstance(block, bool) or not isinstance(block, (int, np.integer)) or block <= 0:
        raise ValueError("block must be a positive integer")
    block = int(block)
    dev, kind, limit = device_info(device)
    itemsize = _array_metadata(family, dev)
    # Instance data is explicitly transferred by the probe; it need not already
    # reside on the planned device. Metadata inspection does not copy it.
    for leaf in jax.tree_util.tree_leaves(data):
        if hasattr(leaf, "dtype"):
            itemsize = max(itemsize, np.dtype(leaf.dtype).itemsize)
    fixed = dict(fixed_kwargs, block=block)
    method, variant = fixed.get("method", "pdhg"), fixed.get("pdhg_variant", "halpern")
    if candidates is None:
        candidates = [{"data_layout": "dense"}]
        if method == "pdhg":
            candidates.append({"data_layout": "compact"})
        if method == "pdhg" and variant == "standard" and not fixed.get("adaptive_omega", False):
            candidates += [{"data_layout": "dense", "loop_unroll": 4},
                           {"data_layout": "compact", "loop_unroll": 4}]
        candidates = [{k: v for k, v in c.items() if k not in fixed} for c in candidates]
    unique = []
    for c in candidates:
        if not isinstance(c, dict) or set(c) - _HARDWARE_KEYS:
            raise ValueError("candidates may choose only data_layout, dual_layout and loop_unroll")
        if any(k in fixed and fixed[k] != v for k, v in c.items()):
            raise ValueError("candidate conflicts with a fixed compile keyword")
        if c not in unique:
            unique.append(dict(c))
    if not unique:
        raise ValueError("at least one candidate is required")
    candidates = unique
    budget = _HEADROOM * limit if limit else None
    identity = _device_identity(dev)
    cache_note = ""
    try:
        sig = family_signature(family, fixed)
        data_id = _fingerprint(data)
        json.dumps(fixed, allow_nan=False)
    except _Uncacheable as exc:
        sig, data_id, cache_note = uuid.uuid4().hex, None, str(exc)
    except TypeError:
        sig, data_id, cache_note = uuid.uuid4().hex, None, "non-JSON compile settings"
    key_payload = [sig, data_id, _runtime_identity(), _source_identity(), identity,
                   limit, candidates, bool(probe), data is not None, itemsize, _SCHEMA, _HEADROOM]
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]
    path = os.path.join(_cache_dir(cache_dir), key + ".json")
    if not cache_note and mode == "cached" and os.path.exists(path):
        try:
            cached = Plan.from_json(Path(path).read_text())
            current_peak = int(_memory_stats(dev).get("peak_bytes_in_use") or 0) or None
            observed = max(cached.measured_peak_bytes or 0, current_peak or 0) or None
            expected = [dict(fixed, **c) for c in candidates]
            if (cached.schema == _SCHEMA and cached.key == key and cached.compile_kwargs in expected
                    and cached.device_identity == identity and cached.fits_estimate
                    and (not probe or data is None or _memory_admissible(dev, budget, observed))):
                return cached
        except (ValueError, TypeError, KeyError, OSError):
            pass  # corrupt, obsolete or incompatible entries are misses

    ranked = []
    for c in candidates:
        layout = c.get("data_layout", fixed.get("data_layout", "dense"))
        e = estimate_bytes(family, block, layout, itemsize=itemsize)
        ranked.append({"candidate": c, "estimated_bytes": e,
                       "fits_estimate": bool(budget is None or e <= budget)})
    fitting = [r for r in ranked if r["fits_estimate"]]
    if not fitting:
        raise PlanningError("all screening estimates exceed the device budget; "
                            "no package streaming plan is implemented", ranked)
    chosen = None
    if probe and data is not None:
        for r in fitting:
            s, p, err = _probe(family, data, r["candidate"], fixed, block, dev)
            finite_time = s is not None and np.isfinite(s) and s >= 0
            admissible = _memory_admissible(dev, budget, p)
            r.update(seconds_per_block=s, measured_peak_bytes=p, error=err,
                     memory_scope="device process-lifetime allocator high-water mark",
                     fits_measured_budget=bool(admissible))
            if err is None and finite_time and admissible:
                if chosen is None or s < chosen["seconds_per_block"]:
                    chosen = r
        final_peak = int(_memory_stats(dev).get("peak_bytes_in_use") or 0) or None
        if chosen is None or not _memory_admissible(dev, budget, final_peak):
            raise PlanningError("no measured candidate meets the device budget and runtime checks; "
                                "allocator peaks include prior allocations (retry in a fresh process)", ranked)
        peak = max(chosen.get("measured_peak_bytes") or 0, final_peak or 0) or None
        seconds = chosen["seconds_per_block"]
        note = "two-block pilot only; peak is process-lifetime allocator use, not candidate-only or total CUDA memory"
        memory_scope = "device process-lifetime allocator high-water mark"
    else:
        chosen, seconds, peak = fitting[0], None, None
        note, memory_scope = "chosen from the estimate only (no probe); measured fit is unknown", "unmeasured estimate"
    if cache_note:
        note += "; cache disabled: " + cache_note
    result = Plan(kind, limit, key, dict(fixed, **chosen["candidate"]),
                  chosen["estimated_bytes"], chosen["fits_estimate"], peak, seconds,
                  ranked, note, time.strftime("%Y-%m-%d %H:%M:%S"), ocs.__version__,
                  device_identity=identity,
                  cache_scope="disabled" if cache_note else "process-scoped immutable object identity",
                  memory_scope=memory_scope)
    if not cache_note:
        # An interrupted write must not become a usable cache entry.
        name = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=os.path.dirname(path), delete=False) as fh:
                name = fh.name
                fh.write(result.to_json())
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(name, path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)
    return result


def plan(family, data=None, *, mode="auto", **kwargs):
    """Public optional planner, also exposed as ocs.plan."""
    return plan_family(family, data, mode=mode, **kwargs)
