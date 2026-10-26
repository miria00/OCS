"""Compact, orbit-tied mixed-metric construction; experimental, not public OCS.

SharedShapeGroup stores one SPD matrix A for ``count`` identically declared
blocks. This is the exact all-amplitudes-tied subclass A_i = alpha S,
det(S)=1. It is not the more general free-amplitude shared-shape class: that
class needs a coupled spectral-box projection and is deliberately rejected.

The intrinsic metric of a shared group has multiplicity ``count``. Its
sample Gram is pooled directly into k by k storage, and divided by count.
No per-block k by k Gram or metric is allocated. Scalar/diagonal/untied-full
groups reuse the previously audited mixed_metric_gpu implementation.
"""
from dataclasses import dataclass
import math
import numbers

import jax
import jax.numpy as jnp

from mixed_metric_gpu import (
    BlockGroup, initial_state as _initial_state, apply_power as _apply_power,
    _update_side, average_side as _average_side, _sym, _mm, _spectral,
    _spectral_many,
)


@dataclass(frozen=True)
class SharedShapeGroup:
    size: int
    count: int = 1
    amplitudes: str = 'tied'

    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, numbers.Integral) or x <= 0
               for x in (self.size, self.count)):
            raise ValueError('positive integer size/count required')
        if self.amplitudes != 'tied':
            raise NotImplementedError('free amplitudes require joint box projection')

    @property
    def kind(self):
        return 'shared_full'

    @property
    def dimension(self):
        return self.size * self.count


def initial_state(groups, dtype=jnp.float32):
    return tuple(jnp.eye(g.size, dtype=dtype) if isinstance(g, SharedShapeGroup)
                 else _initial_state((g,), dtype)[0] for g in groups)


def shared_shape_parts(matrix):
    """Return log(alpha), S with det(S)=1; diagnostics, not iterative state."""
    values, vectors = jnp.linalg.eigh(_sym(matrix))
    log_alpha = jnp.mean(jnp.log(values))
    shape = _sym(_mm(vectors * (values / jnp.exp(log_alpha))[None, :], vectors.T))
    return log_alpha, shape


def apply_power(groups, state, vector, power=1., batch_size=256):
    pieces = []
    start = 0
    for group, value in zip(groups, state):
        v = vector[start:start + group.dimension]
        if isinstance(group, SharedShapeGroup):
            matrix = value if power == 1 else _spectral(value, lambda d: d**power)
            result = _mm(v.reshape(group.count, group.size), matrix.T).reshape(-1)
        else:
            result = _apply_power((group,), (value,), v, power, batch_size)
        pieces.append(result)
        start += group.dimension
    return jnp.concatenate(pieces)


def pooled_gram(samples, group):
    """Direct Q*N by k contraction: output k by k, never N by k by k."""
    w = samples.reshape(-1, group.size)
    return _sym(_mm(w.T, w) / w.shape[0])


def update_side(groups, state, samples, target, mu, box, eta, batch_size=256):
    result = []
    finite = jnp.asarray(True)
    start = 0
    for group, value in zip(groups, state):
        w = samples[:, start:start + group.dimension]
        if isinstance(group, SharedShapeGroup):
            half, log = _spectral_many(value, (jnp.sqrt, jnp.log))
            gram = pooled_gram(w, group)
            tangent = (.5 * _mm(_mm(half, gram), half)
                       - .5 * target * jnp.eye(group.size, dtype=value.dtype)
                       + mu / 4 * log)
            exp = _spectral(-eta * tangent, jnp.exp)
            unconstrained = _sym(_mm(_mm(half, exp), half))
            valid = jnp.all(jnp.stack([jnp.all(jnp.isfinite(z)) for z in
                (half, log, gram, tangent, exp, unconstrained)]))
            # Overflow never becomes a successful clipped answer. Keep the
            # previous valid metric on failure and propagate finite=False.
            new = jax.lax.cond(valid,
                lambda a: _spectral(a, lambda d: jnp.clip(d, jnp.exp(-2*box),
                                                           jnp.exp(2*box))),
                lambda _: value, unconstrained)
        else:
            updated, valid = _update_side((group,), (value,), w, target,
                                           mu, box, eta, batch_size)
            new = updated[0]
        result.append(new)
        finite = finite & valid
        start += group.dimension
    return tuple(result), finite


def average_side(groups, old, new, ratio, batch_size=256):
    result = []
    for group, a, b in zip(groups, old, new):
        if isinstance(group, SharedShapeGroup):
            half, invhalf = _spectral_many(a, (jnp.sqrt, lambda d: 1/jnp.sqrt(d)))
            power = _spectral(_mm(_mm(invhalf, b), invhalf), lambda d: d**ratio)
            result.append(_sym(_mm(_mm(half, power), half)))
        else:
            result.append(_average_side((group,), (a,), (b,), ratio, batch_size)[0])
    return tuple(result)


def _valid_state(groups, state, box):
    valid = jnp.asarray(True)
    slack = 64 * jnp.finfo(jnp.float32).eps
    for group, value in zip(groups, state):
        if group.kind in ('full', 'shared_full'):
            eigenvalues = jnp.linalg.eigvalsh(_sym(value))
            scale = jnp.maximum(1., jnp.max(jnp.abs(value)))
            valid = (valid & jnp.all(jnp.isfinite(value))
                     & (jnp.max(jnp.abs(value-jnp.swapaxes(value, -1, -2))) <= slack*scale)
                     & jnp.all(eigenvalues >= jnp.exp(-2*box)*(1-slack))
                     & jnp.all(eigenvalues <= jnp.exp(2*box)*(1+slack)))
        else:
            valid = valid & jnp.all(jnp.isfinite(value)) & jnp.all(jnp.abs(value) <= 2*box+slack)
    return valid


def make_constructor(matvec, rmatvec, rows, columns, *, iterations=80, probes=2,
                     mu=1., box=2., shift=20, target_row=None, target_col=None,
                     batch_size=256):
    """Simultaneous fresh-probe projected geodesic SGD with pre-update mean.

    Status: 0 completed numerically; 1 invalid initial state; 2 nonfinite
    arithmetic. Status 0 is not an optimality or float32 convergence claim.
    Same schedule as mixed_metric_gpu; finite stochastic paths differ under
    orbit tying even when symmetry proves that their exact optima agree.
    """
    rows, columns = tuple(rows), tuple(columns)
    if not rows or not columns:
        raise ValueError('nonempty row and column declarations required')
    for group in rows + columns:
        if not isinstance(group, (BlockGroup, SharedShapeGroup)):
            raise TypeError('unsupported metric declaration')
    m, n = sum(g.dimension for g in rows), sum(g.dimension for g in columns)
    for name, value in [('iterations', iterations), ('probes', probes),
                        ('shift', shift), ('batch_size', batch_size)]:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if not math.isfinite(mu) or mu <= 0 or not math.isfinite(box) or box < 0:
        raise ValueError('positive finite mu and nonnegative finite box required')
    r = math.sqrt(n/m) if target_row is None else float(target_row)
    s = math.sqrt(m/n) if target_col is None else float(target_col)
    if any(not math.isfinite(v) or v <= 0 for v in (r, s)):
        raise ValueError('positive finite targets required')

    def run(key, a0, b0):
        for groups, state in ((rows, a0), (columns, b0)):
            if len(groups) != len(state):
                raise ValueError('one state array per metric declaration required')
            for group, value in zip(groups, state):
                expected = ((group.size, group.size) if isinstance(group, SharedShapeGroup)
                    else (group.count, group.size, group.size) if group.kind == 'full'
                    else (group.count,) if group.kind == 'scalar' else (group.count, group.size))
                if value.shape != expected or value.dtype != jnp.float32:
                    raise ValueError('float32 state with declared shape required')
        initial_valid = _valid_state(rows, a0, box) & _valid_state(columns, b0, box)

        def body(index, carry):
            key, a, b, avg_a, avg_b, total_weight, finite = carry
            key, row_key, col_key = jax.random.split(key, 3)
            xi = jax.random.rademacher(row_key, (probes, n), dtype=jnp.float32)
            zeta = jax.random.rademacher(col_key, (probes, m), dtype=jnp.float32)
            forward = jax.vmap(lambda v: matvec(apply_power(columns, b, v, .5, batch_size)))(xi)
            reverse = jax.vmap(lambda v: rmatvec(apply_power(rows, a, v, .5, batch_size)))(zeta)
            t = index + 1
            weight = t + shift - 1
            next_total = total_weight + weight
            ratio = weight / next_total
            avg_a = jax.lax.cond(index == 0, lambda _: a,
                lambda _: average_side(rows, avg_a, a, ratio, batch_size), None)
            avg_b = jax.lax.cond(index == 0, lambda _: b,
                lambda _: average_side(columns, avg_b, b, ratio, batch_size), None)
            eta = 2 / ((mu / 4) * (t + shift))
            next_a, fa = update_side(rows, a, forward, r, mu, box, eta, batch_size)
            next_b, fb = update_side(columns, b, reverse, s, mu, box, eta, batch_size)
            finite = finite & fa & fb & jnp.isfinite(eta) & jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(z)) for z in jax.tree_util.tree_leaves(
                    (next_a, next_b, avg_a, avg_b))]))
            return key, next_a, next_b, avg_a, avg_b, next_total, finite

        _, a, b, avg_a, avg_b, weight, finite = jax.lax.fori_loop(
            0, iterations, body, (key, a0, b0, a0, b0, jnp.asarray(0., jnp.float32), initial_valid))
        status = jnp.where(~initial_valid, 1, jnp.where(finite, 0, 2))
        return {'A': avg_a, 'B': avg_b, 'last_A': a, 'last_B': b,
                'weight_sum': weight, 'finite': finite, 'status': status,
                'operator_calls': jnp.asarray(2*iterations*probes, jnp.int32)}
    return jax.jit(run)


def state_storage_bytes(groups, dtype_bytes=4):
    """One compact metric state, excluding probes, copies and workspaces."""
    return dtype_bytes * sum(g.size*g.size if isinstance(g, SharedShapeGroup) else
        g.count*(g.size*g.size if g.kind == 'full' else g.size if g.kind == 'diagonal' else 1)
        for g in groups)
