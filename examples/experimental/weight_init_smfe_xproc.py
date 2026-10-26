"""Diagnose a cross-process hash mismatch of the SMFE output (step 2 hard stop).

The step-2 gate compared the hoisted SMFE copy against the shipped function IN ONE
PROCESS and found them bitwise identical.  The first timing process then found its
hoisted output's sha256 different from the gate's.  Two explanations predict
different things:
  (a) the copy differs from the shipped function   -> shipped and hoisted differ
      when computed in the SAME process under the timing process's conditions;
  (b) SMFE itself is not bitwise reproducible across processes (GPU kernel or
      autotuning choices differ per process) -> the SHIPPED function's own output
      differs between processes, and shipped == hoisted within each process.
Each invocation runs in a fresh process and saves the shipped and hoisted outputs
with its variant label; `--compare` (numpy only, no device) reports hashes and
maximum absolute differences across all saved variants.
  --variant plain         shipped call, then hoisted call
  --variant cert_first    the timing process's order: certificate compile, then
                          hoisted, then shipped
"""
import argparse, glob, hashlib, json, os, sys, time
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--variant", choices=["plain", "cert_first"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.dir, exist_ok=True)
    if a.compare:
        files = sorted(glob.glob(os.path.join(a.dir, "*_tau.npy")))
        names = [os.path.basename(f)[:-8] for f in files]
        taus = {nm: np.load(os.path.join(a.dir, nm + "_tau.npy")) for nm in names}
        sigs = {nm: np.load(os.path.join(a.dir, nm + "_sigma.npy")) for nm in names}
        ref = names[0]
        out = {}
        for nm in names:
            dt = np.abs(taus[nm].astype(np.float64) - taus[ref].astype(np.float64))
            ds = np.abs(sigs[nm].astype(np.float64) - sigs[ref].astype(np.float64))
            out[nm] = dict(tau_sha=sha(taus[nm]), sigma_sha=sha(sigs[nm]),
                           tau_max_abs_diff_vs_ref=float(dt.max()),
                           tau_max_rel_diff_vs_ref=float((dt / np.abs(taus[ref])).max()),
                           tau_n_diff_vs_ref=int(np.count_nonzero(dt)),
                           sigma_max_abs_diff_vs_ref=float(ds.max()),
                           sigma_n_diff_vs_ref=int(np.count_nonzero(ds)))
            print(nm, json.dumps(out[nm]))
        json.dump(dict(reference=ref, entries=out), open(os.path.join(a.dir, "compare.json"), "w"), indent=1)
        return
    import jax, jax.numpy as jnp
    import weight_init_twostage_timing as TT
    from ocs.solver.equilibrate_sgd import smfe_equilibration
    import smfe_hoisted as SH
    _ = float(jnp.sum(jnp.ones(8)))
    prob, spec = TT.build(a.L)
    mv, rmv, slices, m, spans = TT.stacked_like_shipped(spec)
    label = f"{a.variant}{a.tag}"
    if a.variant == "cert_first":
        prob.compile(tol=1e-5, max_blocks=1, block=100)
        th, sh_, nh, _d, _t = TT.smfe_call(SH.smfe_equilibration_hoisted, mv, rmv, spec, m, spans)
        ts, ss, ns, _d, _t = TT.smfe_call(smfe_equilibration, mv, rmv, spec, m, spans)
    else:
        ts, ss, ns, _d, _t = TT.smfe_call(smfe_equilibration, mv, rmv, spec, m, spans)
        th, sh_, nh, _d, _t = TT.smfe_call(SH.smfe_equilibration_hoisted, mv, rmv, spec, m, spans)
    for nm, t, s in ((f"{label}_shipped", ts, ss), (f"{label}_hoisted", th, sh_)):
        np.save(os.path.join(a.dir, nm + "_tau.npy"), np.asarray(t))
        np.save(os.path.join(a.dir, nm + "_sigma.npy"), np.asarray(s))
    same = np.array_equal(np.asarray(ts), np.asarray(th)) and np.array_equal(np.asarray(ss), np.asarray(sh_))
    print(f"{label}: within-process shipped == hoisted bitwise: {same}; norms {ns} {nh}; "
          f"shipped sha {sha(np.asarray(ts))} hoisted sha {sha(np.asarray(th))}")


if __name__ == "__main__":
    main()
