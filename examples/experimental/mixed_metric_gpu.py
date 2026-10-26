"""Operator-callback geodesic SGD, isolated from the public OCS engine.

Squared metrics use fixed scalar/diagonal/FULL block declarations. Scalar and
diagonal states contain log eigenvalues; FULL states contain SPD matrices.
This implements exact exponential/projection/averaging formulas in floating
point, without a condition cap or an approximate retraction. It does not
provide a coupling-norm bound or establish a downstream PDHG speed guarantee.
"""
from dataclasses import dataclass
import math
import numbers

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class BlockGroup:
    kind: str
    size: int
    count: int = 1

    def __post_init__(self):
        if self.kind not in ('scalar', 'diagonal', 'full'):
            raise ValueError('kind must be scalar, diagonal or full')
        if any(isinstance(x, bool) or not isinstance(x, numbers.Integral) or x <= 0
               for x in (self.size, self.count)):
            raise ValueError('positive integer size/count required')

    @property
    def dimension(self):
        return self.size * self.count


def initial_state(groups, dtype=jnp.float32):
    return tuple(jnp.broadcast_to(jnp.eye(g.size, dtype=dtype), (g.count, g.size, g.size))
                 if g.kind == 'full' else
                 jnp.zeros((g.count,) if g.kind == 'scalar' else (g.count, g.size), dtype)
                 for g in groups)


def _sym(a):
    return (a + jnp.swapaxes(a, -1, -2)) * .5


def _mm(a, b):
    return jnp.matmul(a, b, precision=jax.lax.Precision.HIGHEST)


def _spectral(a, transform):
    values, vectors = jnp.linalg.eigh(_sym(a))
    return _sym(_mm(vectors * transform(values)[..., None, :], jnp.swapaxes(vectors, -1, -2)))


def _spectral_many(a, transforms):
    values, vectors = jnp.linalg.eigh(_sym(a))
    transpose = jnp.swapaxes(vectors, -1, -2)
    return tuple(_sym(_mm(vectors * fn(values)[..., None, :], transpose)) for fn in transforms)


def full_functions(a, transforms, batch_size=256):
    """Reuse one eigendecomposition for several functions of the same matrix."""
    if a.shape[0] <= batch_size:
        return _spectral_many(a, transforms)
    return jax.lax.map(lambda x: _spectral_many(x, transforms), a, batch_size=batch_size)


def full_function(a, transform, batch_size=256):
    if a.shape[0] <= batch_size:
        return _spectral(a, transform)
    # Bounded eigensolver batches; all metric state remains explicit and owned.
    return jax.lax.map(lambda x: _spectral(x, transform), a, batch_size=batch_size)


def apply_power(groups, state, vector, power=1., batch_size=256):
    pieces = []
    start = 0
    for group, value in zip(groups, state):
        v = vector[start:start+group.dimension].reshape(group.count, group.size)
        if group.kind == 'full':
            matrix = value if power == 1 else full_function(value, lambda d: d**power, batch_size)
            result = _mm(matrix, v[..., None])[..., 0]
        elif group.kind == 'scalar':
            result = jnp.exp(power*value)[:, None] * v
        else:
            result = jnp.exp(power*value) * v
        pieces.append(result.reshape(-1))
        start += group.dimension
    return jnp.concatenate(pieces)


def _update_side(groups, state, samples, target, mu, box, eta, batch_size):
    result = []
    finite = jnp.asarray(True)
    start = 0
    for group, value in zip(groups, state):
        w = samples[:, start:start+group.dimension].reshape(-1, group.count, group.size)
        if group.kind == 'full':
            half, log = full_functions(value, (jnp.sqrt, jnp.log), batch_size)
            gram = jnp.einsum('qbi,qbj->bij', w, w, precision=jax.lax.Precision.HIGHEST)/w.shape[0]
            tangent = .5*_mm(_mm(half, gram), half) - .5*target*jnp.eye(group.size) + mu/4*log
            # Congruence of the exact exponential, followed by spectral box
            # projection. Overflow is an explicit numerical failure, not a cap.
            exp = full_function(-eta*tangent, jnp.exp, batch_size)
            unconstrained = _mm(_mm(half, exp), half)
            finite = finite & jnp.all(jnp.stack([jnp.all(jnp.isfinite(z)) for z in
                (half, log, gram, tangent, exp, unconstrained)]))
            new = full_function(unconstrained,
                                lambda d: jnp.clip(d, jnp.exp(-2*box), jnp.exp(2*box)),
                                batch_size)
        else:
            energy = jnp.mean(w*w, axis=0)
            if group.kind == 'scalar':
                # Intrinsic scalar metric carries the block dimension: average
                # row energy, never use the unnormalized sum.
                energy = jnp.mean(energy, axis=1)
            tangent = .5*jnp.exp(value)*energy - .5*target + mu/4*value
            unconstrained = value-eta*tangent
            finite = (finite & jnp.all(jnp.isfinite(energy)) & jnp.all(jnp.isfinite(tangent))
                      & jnp.all(jnp.isfinite(unconstrained)))
            new = jnp.clip(unconstrained, -2*box, 2*box)
        result.append(new)
        start += group.dimension
    return tuple(result), finite


def average_side(groups, old, new, ratio, batch_size=256):
    result = []
    for group, a, b in zip(groups, old, new):
        if group.kind == 'full':
            half, invhalf = full_functions(a, (jnp.sqrt, lambda d: 1/jnp.sqrt(d)), batch_size)
            relative = _mm(_mm(invhalf, b), invhalf)
            power = full_function(relative, lambda d: d**ratio, batch_size)
            value = _sym(_mm(_mm(half, power), half))
        else:
            value = (1-ratio)*a + ratio*b
        result.append(value)
    return tuple(result)


def update_from_samples(rows, columns, a, b, forward_samples, reverse_samples,
                        *, target_row, target_col, mu, box, eta, batch_size=256,
                        return_finite=False):
    """One simultaneous step; samples are K Bhalf xi and K* Ahalf zeta."""
    next_a, fa = _update_side(rows, a, forward_samples, target_row, mu, box, eta, batch_size)
    next_b, fb = _update_side(columns, b, reverse_samples, target_col, mu, box, eta, batch_size)
    return (next_a, next_b, fa & fb) if return_finite else (next_a, next_b)


def make_constructor(matvec, rmatvec, rows, columns, *, iterations=160, probes=2,
                     mu=1., box=2., shift=20, target_row=None, target_col=None,
                     batch_size=256):
    rows, columns = tuple(rows), tuple(columns)
    m, n = sum(g.dimension for g in rows), sum(g.dimension for g in columns)
    if not rows or not columns:
        raise ValueError('nonempty row and column declarations required')
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
        def body(index, carry):
            key, a, b, avg_a, avg_b, total_weight, finite = carry
            key, row_key, col_key = jax.random.split(key, 3)
            xi = jax.random.rademacher(row_key, (probes, n), dtype=jnp.float32)
            zeta = jax.random.rademacher(col_key, (probes, m), dtype=jnp.float32)
            # Both oracle estimates see exactly the same pre-update (A,B).
            forward = jax.vmap(lambda v: matvec(apply_power(columns, b, v, .5, batch_size)))(xi)
            reverse = jax.vmap(lambda v: rmatvec(apply_power(rows, a, v, .5, batch_size)))(zeta)
            t = index + 1
            weight = t + shift - 1
            next_total = total_weight + weight
            ratio = weight/next_total
            next_avg_a = jax.lax.cond(index == 0, lambda _: a,
                lambda _: average_side(rows, avg_a, a, ratio, batch_size), None)
            next_avg_b = jax.lax.cond(index == 0, lambda _: b,
                lambda _: average_side(columns, avg_b, b, ratio, batch_size), None)
            eta = 2/((mu/4)*(t+shift))
            next_a, next_b, intermediate_finite = update_from_samples(rows, columns, a, b, forward, reverse,
                target_row=r, target_col=s, mu=mu, box=box, eta=eta, batch_size=batch_size,
                return_finite=True)
            next_finite = finite & intermediate_finite & jnp.isfinite(eta) & jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(z)) for z in
                jax.tree_util.tree_leaves((next_a, next_b, next_avg_a, next_avg_b))]))
            return key, next_a, next_b, next_avg_a, next_avg_b, next_total, next_finite

        _, a, b, avg_a, avg_b, weight, finite = jax.lax.fori_loop(
            0, iterations, body, (key, a0, b0, a0, b0, jnp.asarray(0., jnp.float32), jnp.asarray(True)))
        return {'A': avg_a, 'B': avg_b, 'last_A': a, 'last_B': b,
                'weight_sum': weight, 'finite': finite,
                'operator_calls': jnp.asarray(2*iterations*probes, jnp.int32)}
    return jax.jit(run)


def state_storage_bytes(groups):
    """One float32 metric state, excluding copies, probes and workspaces."""
    return 4*sum(g.count*(g.size*g.size if g.kind == 'full' else
                         g.size if g.kind == 'diagonal' else 1) for g in groups)
