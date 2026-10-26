"""Quadratically regularized optimal transport, using a reusable native family.

Minimize <C,X> + (EPS/2)||X||_F^2 subject to X>=0 and prescribed marginals.
The proximal calculus combines the nonnegative domain and quadratic term.
MarginalOperator applies row/column sums without storing the constraint matrix.
Changed costs/marginals reuse the family and its GPU executables.

Run: python examples/killer_ot.py --size 64 --batch 64
POT, when installed, compares the same quadratic objective on CPU. First-use
and warm public timings are distinct. Relative KKT tolerance alone is not an
objective-error bound; independent marginal and objective checks are printed.
"""
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from _quadratic_ot import family, instance, data, diagnostics, pot_reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--tol', type=float, default=2e-6)
    p.add_argument('--max-blocks', type=int, default=2000)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    t = time.perf_counter(); compiled = family(a.size,tol=a.tol,max_blocks=a.max_blocks)
    print(f'{jax.devices()[0]}: family setup {time.perf_counter()-t:.3f}s')
    c,pm,qm = instance(a.size,rng)
    one = data(c,pm,qm)
    t = time.perf_counter(); sol = compiled.solve_instance(one); jax.block_until_ready(sol.x)
    print(f'first public solve {time.perf_counter()-t:.3f}s; {sol.status}, residual {sol.residual:.2e}')
    checked = diagnostics(c,pm,qm,sol.x,sol.y)
    print(f'quadratic objective {checked["objective"]:.9g}; marginal relative {checked["marginal_relative"]:.2e}; min(X) {checked["min_x"]:.2e}')
    try:
        ref = pot_reference(c,pm,qm)
        difference = abs(checked['objective']-ref['objective'])/max(abs(checked['objective']),abs(ref['objective']),np.finfo(float).tiny)
        print(f'POT quadratic CPU: {ref["seconds"]:.4f}s, success={ref["success"]}, marginal relative {ref["marginal_relative"]:.2e}; relative objective difference {difference:.2e}')
    except ImportError:
        print('POT reference skipped: optional POT package is not installed.')
    instances = [instance(a.size,rng) for _ in range(a.batch)]
    batch = {'c':jnp.asarray(np.stack([c.reshape(-1) for c,_,_ in instances])),
             'm':jnp.asarray(np.stack([np.r_[pm,qm] for _,pm,qm in instances]))}
    t = time.perf_counter(); result = compiled.solve_many_results(batch); jax.block_until_ready(tuple(s.x for s in result))
    print(f'first public batch {time.perf_counter()-t:.3f}s (new batch shape)')
    samples = []
    for _ in range(3):
        t = time.perf_counter(); result = compiled.solve_many_results(batch)
        jax.block_until_ready(tuple(s.x for s in result)); samples.append(time.perf_counter()-t)
    elapsed = float(np.median(samples))
    residual = max(s.residual for s in result)
    marginal = max(diagnostics(c,pm,qm,s.x,s.y)['marginal_relative']
                   for (c,pm,qm),s in zip(instances,result))
    solved = sum(s.status=='solved' for s in result)
    print(f'batch {a.batch}: {solved}/{a.batch} solved, max residual {residual:.2e}, max marginal relative {marginal:.2e}; warm {elapsed:.4f}s; {a.batch/elapsed:.1f} attempted instances/s, {solved/elapsed:.1f} solved instances/s')


if __name__=='__main__':
    main()
