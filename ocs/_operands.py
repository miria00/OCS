"""Runtime operands and 64-bit-safe windows for the solve path (patches C and L).

Candidate 0.3.4+patchb.c1 (patch C), extended in 0.3.4+patchb.c1.l1 (patch L: host-resident
host constants, donation, a build record).  Two mechanisms:

``operand_jit(f)``
    A drop-in for ``jax.jit(f)`` on the solve path.  A jitted function that
    reads a concrete array through a Python closure or an object attribute
    (a step vector, a sampling index set, a mask, a preconditioner symbol, a
    measurement vector) records that array as a jaxpr constant, and with the
    jax 0.11.1 defaults the lowering copies it to the host and embeds it in
    the executable.  The executable then owns a second device copy of the
    data, and an embedded constant of 2**31 or more elements crashes the GPU
    compiler (SIGSEGV after buffer assignment, B200 diagnosis 2026-10-23).

    ``operand_jit`` traces ``f`` once per input signature, finds every array
    constant larger than ``EMBED_MAX_BYTES`` at any depth of the traced
    program (nested ``jit`` calls, ``while``/``cond``/``scan`` bodies), turns
    each into an explicit input of the program, and compiles it with those
    arrays passed as runtime operands.  The operands are the device buffers
    the closures already hold, passed by reference on every call.  (Patch C put a
    host numpy constant on the device once; patch L keeps it on the host between
    calls and uploads it for each call, see `_HostOperand`.)
    Only shapes, dtypes and the traced structure are static.  The traced
    operations themselves are unchanged: no operation is added, removed or
    reordered; constants become parameters.

    Called under an enclosing trace (inside another jit, vmap, ...), it
    behaves as ``jax.jit(f)``; the enclosing ``operand_jit`` hoists the
    constants of the nested program.

``set_window(y, a, b, v)``
    Replaces ``y.at[a:b].set(v)`` for static Python bounds.  With x64 off,
    jax lowers the latter to a FILL_OR_DROP scatter whose bound
    ``dim - window`` is int32, so on a vector above 2**31 - 1 elements the
    update is dropped silently whenever ``dim - window`` > 2**31 - 1
    (measured on the 3.8e9-element stacked dual).  ``lax.dynamic_update_slice``
    is not a safe substitute: under ``vmap`` it batches into a CLIP-mode
    scatter whose clamp has the same int32 bound.  ``set_window`` builds the
    result from static slices and one concatenation, which carry no index
    operand at all (slice bounds are int64 attributes of the HLO), so it is
    exact at every size, batched or not.  The values are the same as the
    scatter's: a copy of ``v`` (cast to ``y``'s dtype) in the window and of
    ``y`` elsewhere.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax._src import core as _core
from jax._src import pjit as _pjit
from jax._src.interpreters import partial_eval as _pe
from jax._src.lax.control_flow import conditionals as _conditionals
from jax._src.lax.control_flow import loops as _loops
from jax._src.sharding_impls import UNSPECIFIED as _UNSPECIFIED

# The same size rule as jax's own `jax_embedded_constants_max_bytes` default:
# an array constant of more than 32 bytes is an operand; scalars and tiny
# literals stay in the program.
EMBED_MAX_BYTES = 32

_CONTEXT_FLAGS = ("jax_enable_x64", "jax_default_matmul_precision", "jax_default_device",
                  "jax_default_prng_impl", "jax_threefry_partitionable",
                  "jax_numpy_dtype_promotion", "jax_numpy_rank_promotion")


def _context():
    return tuple(str(getattr(jax.config, name, None)) for name in _CONTEXT_FLAGS)


def _hoistable(value):
    """A concrete numeric array constant above the embedding limit."""
    if isinstance(value, _core.Tracer):
        return False
    if isinstance(value, jax.Array):
        if jnp.issubdtype(value.dtype, jax.dtypes.extended):
            return False
        return value.ndim > 0 and value.nbytes > EMBED_MAX_BYTES
    if isinstance(value, np.ndarray):
        return (value.dtype != object and value.ndim > 0
                and value.nbytes > EMBED_MAX_BYTES)
    return False


def _sub_jaxprs(params):
    for value in params.values():
        if isinstance(value, _core.Jaxpr):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                if isinstance(item, _core.Jaxpr):
                    yield item


def _subst(atom, mapping):
    return mapping.get(atom, atom) if isinstance(atom, _core.Var) else atom


class _Hoister:
    """Rewrites a closed jaxpr so that its large constants are leading inputs."""

    def __init__(self):
        self._has = {}
        self._done = {}

    def has_large(self, jaxpr):
        key = id(jaxpr)
        if key not in self._has:
            self._has[key] = False                    # recursion guard
            self._has[key] = (any(_hoistable(c) for c in jaxpr.consts)
                              or any(self.eqn_has_large(e) for e in jaxpr.eqns))
        return self._has[key]

    def eqn_has_large(self, eqn):
        return any(self.has_large(j) for j in _sub_jaxprs(eqn.params))

    def hoist(self, jaxpr):
        """(new, values): ``new`` computes what ``jaxpr`` computes and takes
        ``values`` (the hoisted constants, deduplicated by identity) as its
        leading inputs, before the original inputs."""
        if not self.has_large(jaxpr):
            return jaxpr, []
        key = id(jaxpr)
        if key in self._done:
            return self._done[key]
        lifted_vars, lifted_vals = {}, {}

        def lift(values, avals):
            # Values a sub-program needs as inputs: new constants of THIS level,
            # which the partition below then moves to this level's inputs.
            out = []
            for value, aval in zip(values, avals):
                k = id(value)
                if k not in lifted_vars:
                    lifted_vars[k] = _core.Var(aval)
                    lifted_vals[k] = value
                out.append(lifted_vars[k])
            return out

        eqns = [self._rewrite(e, lift) if self.eqn_has_large(e) else e for e in jaxpr.eqns]
        kept_vars, kept_vals, moved, mapping = [], [], {}, {}
        for var, value in zip(jaxpr.constvars, jaxpr.consts):
            if _hoistable(value):
                if id(value) in moved:
                    mapping[var] = moved[id(value)][0]
                else:
                    moved[id(value)] = (var, value)
            else:
                kept_vars.append(var)
                kept_vals.append(value)
        for k, var in lifted_vars.items():
            if k in moved:
                mapping[var] = moved[k][0]
            else:
                moved[k] = (var, lifted_vals[k])
        outvars = list(jaxpr.outvars)
        if mapping:
            eqns = [e.replace(invars=[_subst(a, mapping) for a in e.invars]) for e in eqns]
            outvars = [_subst(a, mapping) for a in outvars]
        moved_vars = [v for v, _ in moved.values()]
        moved_vals = [c for _, c in moved.values()]
        invars = [*moved_vars, *jaxpr.invars]
        new = jaxpr.replace(constvars=kept_vars, invars=invars, outvars=outvars, eqns=eqns,
                            effects=_pe.make_jaxpr_effects(kept_vars, invars, outvars, eqns),
                            consts=kept_vals)
        self._done[key] = (new, moved_vals)
        return new, moved_vals

    def _rewrite(self, eqn, lift):
        prim, params = eqn.primitive, dict(eqn.params)
        if prim is _pjit.jit_p:
            inner, values = self.hoist(params["jaxpr"])
            k = len(values)
            extra = lift(values, [v.aval for v in inner.invars[:k]])
            params["jaxpr"] = inner
            params["in_shardings"] = (_UNSPECIFIED,) * k + tuple(params["in_shardings"])
            params["in_layouts"] = (None,) * k + tuple(params["in_layouts"])
            params["donated_invars"] = (False,) * k + tuple(params["donated_invars"])
            return eqn.replace(invars=[*extra, *eqn.invars], params=params)
        if prim is _loops.while_p:
            cj, cvals = self.hoist(params["cond_jaxpr"])
            bj, bvals = self.hoist(params["body_jaxpr"])
            nc, nb = params["cond_nconsts"], params["body_nconsts"]
            cext = lift(cvals, [v.aval for v in cj.invars[:len(cvals)]])
            bext = lift(bvals, [v.aval for v in bj.invars[:len(bvals)]])
            ins = list(eqn.invars)
            params.update(cond_jaxpr=cj, body_jaxpr=bj, cond_nconsts=nc + len(cvals),
                          body_nconsts=nb + len(bvals))
            return eqn.replace(invars=[*cext, *ins[:nc], *bext, *ins[nc:nc + nb], *ins[nc + nb:]],
                               params=params)
        if prim is _loops.scan_p:
            j, values = self.hoist(params["jaxpr"])
            k = len(values)
            extra = lift(values, [v.aval for v in j.invars[:k]])
            params.update(jaxpr=j, num_consts=params["num_consts"] + k,
                          linear=(False,) * k + tuple(params["linear"]))
            return eqn.replace(invars=[*extra, *eqn.invars], params=params)
        if prim is _conditionals.cond_p:
            hoisted = [self.hoist(b) for b in params["branches"]]
            union = {}
            for bj, values in hoisted:
                for i, value in enumerate(values):
                    union.setdefault(id(value), (value, bj.invars[i].aval))
            keys = list(union)
            branches = []
            for bj, values in hoisted:
                own = {id(value): bj.invars[i] for i, value in enumerate(values)}
                lead = [own[k] if k in own else _core.Var(union[k][1]) for k in keys]
                inv = [*lead, *bj.invars[len(values):]]
                branches.append(bj.replace(
                    invars=inv, effects=_pe.make_jaxpr_effects(bj.constvars, inv, bj.outvars, bj.eqns)))
            extra = lift([union[k][0] for k in keys], [union[k][1] for k in keys])
            params["branches"] = tuple(branches)
            ins = list(eqn.invars)
            return eqn.replace(invars=[ins[0], *extra, *ins[1:]], params=params)
        raise NotImplementedError(
            f"operand_jit: primitive {prim} carries an array constant above "
            f"{EMBED_MAX_BYTES} bytes in a sub-program; no operand rewrite is defined for it")



_PLAIN = object()

# Patch L (0.3.4+patchb.c1.l1): one record per executable built, for the controls and the gate's
# memory accounting (device operands by reference, host operands uploaded per call, donation).
BUILDS = []


class _HostOperand:
    """A host (numpy) constant, held on the host between calls (patch L).

    It is converted ONCE to the dtype the program was traced with -- the conversion the lowering
    would otherwise have applied when embedding it -- and uploaded for each call; the device copy
    is released when the call's outputs no longer need it.  A host constant therefore occupies no
    device memory while other executables (the drive) run."""
    __slots__ = ("host", "wide")

    def __init__(self, value, aval):
        dtype = np.dtype(aval.dtype)
        self.host = np.ascontiguousarray(np.asarray(value).astype(dtype, copy=False).reshape(aval.shape))
        self.wide = dtype.itemsize == 8 or dtype == np.complex128

    def upload(self):
        if self.wide:
            with jax.enable_x64(True):
                return jax.device_put(self.host)
        return jax.device_put(self.host)


def _as_operand(value, aval):
    """What an executable receives for one hoisted constant.

    A device array whose dtype and shape are the constant's own is passed as it is: the
    closure's buffer, by reference, on every call.  A device array recorded under another dtype
    is converted once, on the device.  A host (numpy) constant stays on the host (patch L)."""
    dtype = np.dtype(aval.dtype)
    if isinstance(value, jax.Array):
        if value.dtype == dtype and tuple(value.shape) == tuple(aval.shape):
            return value
        if dtype.itemsize == 8 or dtype == np.complex128:
            with jax.enable_x64(True):
                return value.astype(dtype)
        return value.astype(dtype)
    return _HostOperand(value, aval)


def _materialize(operands):
    return [o.upload() if isinstance(o, _HostOperand) else o for o in operands]


def _signature(leaf):
    aval = _core.typeof(leaf)
    return (tuple(getattr(aval, "shape", ())), str(getattr(aval, "dtype", type(leaf))),
            bool(getattr(aval, "weak_type", False)))


class operand_jit:
    """``jax.jit`` with every large captured array as a runtime operand.

    ``donate_argnums`` (patch L) donates those positional arguments' buffers to the executable,
    as ``jax.jit(..., donate_argnums=...)`` does; the caller must not use them afterwards."""

    def __init__(self, fun, *, donate_argnums=()):
        self._fun = fun
        self._donate = tuple(sorted({int(i) for i in (
            (donate_argnums,) if isinstance(donate_argnums, int) else donate_argnums)}))
        self._plain = jax.jit(fun, donate_argnums=self._donate)
        self._entries = {}
        self.__name__ = getattr(fun, "__name__", "operand_jit")
        self.__qualname__ = getattr(fun, "__qualname__", self.__name__)
        self.__doc__ = getattr(fun, "__doc__", None)
        self.__wrapped__ = fun

    def _key(self, args, kwargs):
        leaves, tree = jax.tree_util.tree_flatten((args, kwargs))
        return leaves, (tree, tuple(_signature(a) for a in leaves), _context())

    def __call__(self, *args, **kwargs):
        leaves, key = self._key(args, kwargs)
        if any(isinstance(a, _core.Tracer) for a in leaves):
            return self._plain(*args, **kwargs)
        entry = self._entries.get(key)
        if entry is None:
            entry = self._entries[key] = self._build(args, kwargs)
        if entry is _PLAIN:
            return self._plain(*args, **kwargs)
        executable, operands, out_tree = entry
        return jax.tree_util.tree_unflatten(out_tree, executable(_materialize(operands), *leaves))

    def _build(self, args, kwargs):
        fun = self._fun
        closed, out_shape = jax.make_jaxpr(lambda a, k: fun(*a, **k), return_shape=True)(args, kwargs)
        if any(isinstance(c, _core.Tracer) for c in closed.consts):
            return _PLAIN
        program, values = _Hoister().hoist(closed)
        avals = [v.aval for v in program.invars[:len(values)]]
        operands = [_as_operand(v, a) for v, a in zip(values, avals)]
        kept = list(program.consts)

        def run(operand_values, *flat):
            return _core.eval_jaxpr(program, kept, *operand_values, *flat)

        run.__name__ = self.__name__
        donate = ()
        if self._donate:
            # flat positions of the donated positional arguments, after the operand list (arg 0)
            counts = [len(jax.tree_util.tree_leaves(a)) for a in args]
            starts = np.cumsum([0] + counts)
            donate = tuple(1 + j for i in self._donate if i < len(args)
                           for j in range(int(starts[i]), int(starts[i + 1])))
        BUILDS.append({
            "name": self.__name__,
            "device_operands": sum(1 for o in operands if not isinstance(o, _HostOperand)),
            "device_operand_bytes": sum(int(o.nbytes) for o in operands if not isinstance(o, _HostOperand)),
            "host_operands": sum(1 for o in operands if isinstance(o, _HostOperand)),
            "host_operand_bytes": sum(int(o.host.nbytes) for o in operands if isinstance(o, _HostOperand)),
            "donated_flat_args": len(donate)})
        return jax.jit(run, donate_argnums=donate), operands, jax.tree_util.tree_structure(out_shape)

    # ---- inspection (controls) -------------------------------------------
    def _entry(self, args, kwargs):
        leaves, key = self._key(args, kwargs)
        entry = self._entries.get(key)
        if entry is None:
            entry = self._entries[key] = self._build(args, kwargs)
        return leaves, entry

    def operands_for(self, *args, **kwargs):
        """The hoisted operands of the executable for these arguments: device arrays (the
        closures' own buffers) and, for host constants, their host arrays."""
        _leaves, entry = self._entry(args, kwargs)
        if entry is _PLAIN:
            return None
        return [o.host if isinstance(o, _HostOperand) else o for o in entry[1]]

    def call_with_operands(self, operands, *args, **kwargs):
        """Run the SAME executable with replacement operand values (controls)."""
        leaves, entry = self._entry(args, kwargs)
        executable, own, out_tree = entry
        if len(operands) != len(own):
            raise ValueError("operand count differs from the executable's")
        return jax.tree_util.tree_unflatten(out_tree, executable(list(operands), *leaves))

    def lower_with_operands(self, *args, **kwargs):
        """Lowered program of the executable for these arguments (controls)."""
        leaves, entry = self._entry(args, kwargs)
        if entry is _PLAIN:
            return self._plain.lower(*args, **kwargs)
        executable, operands, _ = entry
        return executable.lower(_materialize(operands), *leaves)

def set_window(y, a, b, v):
    """``y`` with ``y[a:b]`` replaced by ``v``: static bounds, any size, any batching."""
    a, b = int(a), int(b)
    size = int(y.shape[0])
    if not 0 <= a <= b <= size:
        raise ValueError(f"window [{a}, {b}) outside [0, {size})")
    v = jnp.broadcast_to(jnp.asarray(v).astype(y.dtype), (b - a,) + tuple(y.shape[1:]))
    parts = [lax.slice_in_dim(y, 0, a, axis=0)] if a > 0 else []
    parts.append(v)
    if b < size:
        parts.append(lax.slice_in_dim(y, b, size, axis=0))
    return parts[0] if len(parts) == 1 else lax.concatenate(parts, 0)
