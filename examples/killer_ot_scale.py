"""Scale the native quadratic-OT family with explicit status and accuracy checks.

The transport plan and costs have N² entries. The explicit marginal matrix has
2N² nonzeros, so this example does not claim that competing sparse methods
cannot form it. The OCS operator applies row/column sums directly.

Run tested modest sizes: python examples/killer_ot_scale.py
Request larger sizes explicitly: --sizes 1024 4096 10000
A requested size is an experiment, not a capacity or convergence guarantee.
POT comparisons use the same quadratic objective, not Sinkhorn's entropic one.
"""
import argparse
import time
import numpy as np
import jax
from _quadratic_ot import family, instance, data, diagnostics, pot_reference


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sizes', nargs='+', type=int, default=[64,256])
    p.add_argument('--tol', type=float, default=2e-6)
    p.add_argument('--max-blocks', type=int, default=2000)
    p.add_argument('--reference-max-size', type=int, default=256)
    a = p.parse_args()
    print('N variables status residual marginal_relative quadratic_objective setup_s first_public_s warm_public_s')
    for n in a.sizes:
        c,pm,qm = instance(n,np.random.default_rng(0),geometric=True)
        d = data(c,pm,qm)
        t = time.perf_counter(); comp = family(n,tol=a.tol,max_blocks=a.max_blocks)
        setup = time.perf_counter()-t
        t = time.perf_counter(); sol = comp.solve_instance(d); jax.block_until_ready(sol.x)
        first = time.perf_counter()-t
        t = time.perf_counter(); sol = comp.solve_instance(d); jax.block_until_ready(sol.x)
        elapsed = time.perf_counter()-t
        checked = diagnostics(c,pm,qm,sol.x,sol.y)
        print(f'{n} {n*n} {sol.status} {sol.residual:.3e} {checked["marginal_relative"]:.3e} {checked["objective"]:.9g} {setup:.3f} {first:.3f} {elapsed:.4f}')
        if n<=a.reference_max_size:
            try:
                ref = pot_reference(c,pm,qm)
                difference = abs(checked['objective']-ref['objective'])/max(abs(checked['objective']),abs(ref['objective']))
                print(f'  POT quadratic CPU: {ref["seconds"]:.4f}s, success={ref["success"]}, marginal relative {ref["marginal_relative"]:.2e}; relative objective difference {difference:.2e}')
            except ImportError:
                print('  POT reference skipped: optional POT package is not installed.')
        # Release the preceding problem before allocating the next size.
        del sol, comp, d, c, pm, qm


if __name__=='__main__':
    main()
