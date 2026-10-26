"""Does the equilibration clip box ever bind, and does the projection order matter?

The paper describes a projection onto a box intersected with a tied-coordinate
subspace. The Euclidean projection of a tentative point w onto
C_B = {a*1 : -M <= a <= M} is clip(mean(w), -M, M) * 1: average first, clip the
scalar second. The implementation clips each coordinate first and then averages
the clipped values, which is a different map.

Whether that changes any reported number depends on whether the box binds at
all. If no tentative coordinate ever leaves [-M, M], clip is the identity there
and the two orders agree exactly.

This tests it as a black box, with no instrumentation of the solver: run the
same equilibration at the default M and at a much larger M. If the box never
binds, the two runs must agree bit for bit.
"""
import numpy as np, jax, jax.numpy as jnp
import ocs
from ocs.solver.equilibrate_sgd import smfe_equilibration

M_DEFAULT, M_LOOSE = 6.0, 600.0


def families():
    rng = np.random.default_rng(0)
    out = []

    # a badly scaled dense operator, the case equilibration exists for
    m, n = 300, 200
    A = rng.normal(size=(m, n)).astype("f")
    A *= np.exp(rng.uniform(-4, 4, size=(m, 1))).astype("f")     # row scaling spread
    Ad = jnp.asarray(A)
    out.append(("dense, rows spread 1e4", lambda x: Ad @ x, lambda y: Ad.T @ y,
                n, m, ()))

    # with tied blocks, which is where the projection order applies at all
    out.append(("dense + tied blocks", lambda x: Ad @ x, lambda y: Ad.T @ y,
                n, m, ((0, 60), (60, 140))))

    # an operator family the paper actually runs
    L = 32; nn = L * L
    F = ocs.FFTMask.random(nn, 0.35, seed=0)
    Gr = ocs.Gradient2D(L, L)
    mv = jax.jit(lambda x: jnp.concatenate([F.matvec(x), Gr.matvec(x)]))
    mF = F.shape[0]
    rmv = jax.jit(lambda y: F.rmatvec(y[:mF]) + Gr.rmatvec(y[mF:]))
    mtot = mF + Gr.shape[0]
    out.append(("fft + tv, tied gradient", mv, rmv, nn, mtot, ((mF, mtot),)))
    return out


def dump(path):
    """Save the scalings at the default M, so the same code path can be re-run
    after the projection order is corrected and the two compared directly."""
    out = {}
    for name, mv, rmv, n, m, spans in families():
        a = smfe_equilibration(mv, rmv, n, m, spans, M=M_DEFAULT, seed=0)
        out[name + "|tau"] = np.asarray(a[0])
        out[name + "|sigma"] = np.asarray(a[1])
        out[name + "|nrm"] = np.asarray(float(a[2]))
    np.savez(path, **out)
    print(f"wrote {path} with {len(out)} arrays")


def compare(path):
    ref = np.load(path)
    print(f"{'family':26s} {'max|d tau|':>12s} {'max|d sigma|':>13s} "
          f"{'rel nrm':>10s}")
    for name, mv, rmv, n, m, spans in families():
        a = smfe_equilibration(mv, rmv, n, m, spans, M=M_DEFAULT, seed=0)
        dt = float(np.abs(np.asarray(a[0]) - ref[name + "|tau"]).max())
        ds = float(np.abs(np.asarray(a[1]) - ref[name + "|sigma"]).max())
        r0 = float(ref[name + "|nrm"]); r1 = float(a[2])
        print(f"{name:26s} {dt:12.3e} {ds:13.3e} "
              f"{abs(r1-r0)/max(abs(r0),1e-30):10.3e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--dump":
        dump(sys.argv[2]); sys.exit(0)
    if len(sys.argv) > 2 and sys.argv[1] == "--compare":
        compare(sys.argv[2]); sys.exit(0)
    print(f"clip box: default M={M_DEFAULT}, loosened M={M_LOOSE}")
    print("if the box never binds, the two runs agree exactly\n")
    print(f"{'family':26s} {'max|tau diff|':>14s} {'max|sigma diff|':>16s} "
          f"{'binds?':>8s}")
    for name, mv, rmv, n, m, spans in families():
        a = smfe_equilibration(mv, rmv, n, m, spans, M=M_DEFAULT, seed=0)
        b = smfe_equilibration(mv, rmv, n, m, spans, M=M_LOOSE, seed=0)
        dt = float(np.abs(np.asarray(a[0]) - np.asarray(b[0])).max())
        ds = float(np.abs(np.asarray(a[1]) - np.asarray(b[1])).max())
        binds = "YES" if max(dt, ds) > 0 else "no"
        print(f"{name:26s} {dt:14.3e} {ds:16.3e} {binds:>8s}")
    print("\nYES means the clip is active, so the projection order can change "
          "the reported scalings.\nno means clip is the identity there and the "
          "two orders coincide exactly.")
