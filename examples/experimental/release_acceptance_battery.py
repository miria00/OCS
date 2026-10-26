"""One native returned-point acceptance case; run only on the authorized host.

This is a new strict view, not a relabeling of reliability_battery.py. Input
arrays are rounded once to float32; independent references evaluate those same
arrays in float64. Raw and optional refined pairs never share certificates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import jax
import jax.numpy as jnp
import cvxpy as cp
import ocs


def relative(value, scale):
    return float(value / scale) if scale > 0 else (0. if value == 0 else float("inf"))


def independent_reference(problem, variable, objective, feasibility, zero_size=None):
    """Require an accurate reference status; retain failed reference attempts."""
    attempts = []
    point = None
    for solver, settings in (
        ("CLARABEL", dict(tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=1000)),
        ("SCS", dict(eps_abs=1e-9,eps_rel=1e-9,max_iters=200000)),
    ):
        start = time.perf_counter()
        try:
            problem.solve(solver=solver, warm_start=False, **settings)
            point = np.asarray(variable.value,dtype=float).reshape(-1) if variable.value is not None else None
            if zero_size is not None: point = np.zeros(zero_size)
            value = objective(point) if point is not None else float("nan")
            feas = feasibility(point) if point is not None else float("inf")
            ok = problem.status == "optimal" and np.isfinite(value) and feas <= 1e-7
            row = dict(solver=solver,settings=settings,status=problem.status,objective=value,
                       reported_objective=problem.value,feasibility=feas,accepted_reference=bool(ok),
                       seconds=time.perf_counter()-start,x=None if point is None else point.tolist(),
                       zero_case_analytic_override=zero_size is not None)
        except Exception as error:
            row = dict(solver=solver,settings=settings,status="error",error=repr(error),
                       accepted_reference=False,seconds=time.perf_counter()-start)
            ok = False
        attempts.append(row)
        if ok: break
    selected = dict(attempts[-1])
    selected["attempts"] = attempts
    return selected


def build(kind, size, seed, alpha=.5):
    rng = np.random.default_rng(seed)
    # The non-dyadic power exponent is input data too. Both implementations
    # receive the exact float64 representation of this one float32 value.
    alpha = float(np.float32(alpha))
    arrays = {}
    n = 2 * size if kind == "lp" else size
    x = ocs.Variable(n if kind != "sdp" else n * n)
    z = cp.Variable(n if kind != "sdp" else (n, n), symmetric=(kind == "sdp"))
    if kind == "lp":
        A = rng.normal(size=(size, n)).astype("f")
        truth = np.abs(rng.normal(size=n)).astype("f")
        b = (A @ truth).astype("f")
        c = np.abs(rng.normal(size=n)).astype("f")
        family = ocs.minimize([ocs.nonneg(x), ocs.equality(ocs.Dense(A) @ x - ocs.Parameter("b"))], cost=ocs.Parameter("c"))
        data = {"b": b, "c": c}
        arrays.update(A=A, b=b, c=c)
        Ad, bd, cd = A.astype(float), b.astype(float), c.astype(float)
        ref = cp.Problem(cp.Minimize(cd @ z), [Ad @ z == bd, z >= 0])
        objective = lambda t: float(cd @ t)
        feasibility = lambda t: max(relative(np.linalg.norm(Ad @ t - bd), np.linalg.norm(Ad @ t) + np.linalg.norm(bd)), relative(max(0., -np.min(t)), np.linalg.norm(t, np.inf)))
    elif kind in ("qp", "zero_qp"):
        A = rng.normal(size=(2 * n, n)).astype("f")
        b = rng.normal(size=2 * n).astype("f") if kind == "qp" else np.zeros(2 * n, "f")
        family = ocs.minimize([ocs.box(x, 0, 1), ocs.least_squares(ocs.Dense(A) @ x - ocs.Parameter("b"))])
        data = {"b": b}; arrays.update(A=A, b=b)
        Ad, bd = A.astype(float), b.astype(float)
        ref = cp.Problem(cp.Minimize(.5 * cp.sum_squares(Ad @ z - bd)), [z >= 0, z <= 1])
        objective = lambda t: float(.5 * np.sum((Ad @ t - bd)**2))
        feasibility = lambda t: float(max(0., -np.min(t), np.max(t) - 1.))
    elif kind == "socp":
        c = rng.normal(size=n).astype("f")
        A = rng.normal(size=(n, n)).astype("f")
        ridge = np.full(n, np.sqrt(.01), "f")
        offset = np.r_[1., np.zeros(n)].astype("f")
        family = ocs.minimize([ocs.sum_squares(ocs.Diagonal(ridge) @ x), ocs.soc(ocs.Dense(np.vstack([np.zeros((1, n), "f"), A])) @ x + offset)], cost=ocs.Parameter("c"))
        data = {"c": c}; arrays.update(A=A, c=c, ridge=ridge, offset=offset)
        Ad, cd, qd = A.astype(float), c.astype(float), ridge.astype(float)
        ref = cp.Problem(cp.Minimize(cd @ z + .5 * cp.sum_squares(cp.multiply(qd, z))), [cp.norm(Ad @ z) <= 1])
        objective = lambda t: float(cd @ t + .5 * np.sum((qd * t)**2))
        feasibility = lambda t: float(max(0., np.linalg.norm(Ad @ t) - 1.))
    elif kind in ("quantile", "logistic"):
        A = rng.normal(size=(20 * n, n)).astype("f")
        truth = rng.normal(size=n).astype("f")
        if kind == "quantile":
            b = (A @ truth + .3 * rng.standard_t(3, size=20 * n)).astype("f")
            ridge = np.full(n, np.sqrt(.01), "f")
            family = ocs.minimize([ocs.sum_squares(ocs.Diagonal(ridge) @ x), ocs.pinball(ocs.Dense(-A) @ x + ocs.Parameter("b"), .75)])
            data = {"b": b}; Ad, bd = A.astype(float), b.astype(float)
            loss = bd - Ad @ z
            objective = lambda t: float(np.maximum(.75 * (bd - Ad @ t), -.25 * (bd - Ad @ t)).sum() + .5 * np.sum((ridge.astype(float) * t)**2))
            expression = cp.sum(cp.maximum(.75 * loss, -.25 * loss))
        else:
            labels = np.where(A @ truth + .5 * rng.normal(size=20 * n) > 0, 1., -1.).astype("f")
            A = (-labels[:, None] * A).astype("f"); b = labels
            ridge = np.full(n, np.sqrt(.1), "f")
            family = ocs.minimize([ocs.sum_squares(ocs.Diagonal(ridge) @ x), ocs.logistic(ocs.Dense(A) @ x)])
            data = {}; Ad = A.astype(float)
            expression = cp.sum(cp.logistic(Ad @ z))
            objective = lambda t: float(np.logaddexp(0., Ad @ t).sum() + .5 * np.sum((ridge.astype(float) * t)**2))
        arrays.update(A=A, b=b, ridge=ridge)
        ref = cp.Problem(cp.Minimize(expression + .5 * cp.sum_squares(cp.multiply(ridge.astype(float), z))))
        feasibility = lambda t: 0.
    elif kind in ("exp", "power"):
        # The affine-QP generator from the archived 45-program native corpus:
        # coverage/battery_device.py, same random draw order and float32 arrays.
        from ocs.ir import ProblemFamily, Term
        from ocs.prox.cones import ExpConeBatch, PowerConeBatch
        q=2 if n<=4 else 3
        A=rng.normal(size=(n+3,n)).astype(np.float32)/np.sqrt(n+3)
        A=np.r_[A,.5*np.eye(n,dtype=np.float32)].astype(np.float32)
        b=(2*rng.normal(size=A.shape[0])).astype(np.float32)
        c=(.1*rng.normal(size=n)).astype(np.float32)
        K=(.3*rng.normal(size=(3*q,n))).astype(np.float32)
        offset=(np.r_[np.zeros(q),np.ones(q),2*np.ones(q)] if kind=="exp" else np.r_[np.ones(2*q),.2*np.ones(q)]).astype(np.float32)
        cone=ExpConeBatch(q) if kind=="exp" else PowerConeBatch(q,alpha)
        family=ProblemFamily(n=n,terms=[Term(ocs.Dense(A),ocs.SumSquares(),lambda d:-d["b"]),Term(ocs.Dense(K),cone,lambda d:d["l"])],cost_fn=lambda d:d["c"])
        data={"b":b,"l":offset,"c":c};arrays.update(A=A,b=b,c=c,K=K,l=offset,alpha=np.asarray(alpha,dtype=np.float32))
        Ad,bd,cd,Kd,ld=A.astype(float),b.astype(float),c.astype(float),K.astype(float),offset.astype(float)
        def constraints(point):
            return [cp.ExpCone(point[:q],point[q:2*q],point[2*q:])] if kind=="exp" else [cp.PowCone3D(point[:q],point[q:2*q],point[2*q:],alpha)]
        ref=cp.Problem(cp.Minimize(.5*cp.sum_squares(Ad@z-bd)+cd@z),constraints(Kd@z+ld))
        objective=lambda t:float(.5*np.sum((Ad@t-bd)**2)+cd@t)
        def feasibility(t):
            point=Kd@t+ld
            # Independent Euclidean-distance reference on the same point;
            # no OCS projector is used to validate itself.
            projected=cp.Variable(3*q)
            problem=cp.Problem(cp.Minimize(cp.norm(projected-point)),constraints(projected))
            problem.solve(solver="CLARABEL",tol_gap_abs=1e-10,tol_gap_rel=1e-10,tol_feas=1e-10,max_iter=1000)
            if problem.status!="optimal" or projected.value is None:return float("inf")
            return relative(np.linalg.norm(np.asarray(projected.value)-point),max(np.linalg.norm(Kd@t),np.linalg.norm(ld)))
    elif kind == "sdp":
        W = rng.normal(size=(n, n)).astype("f"); W = (W + W.T) / 2
        np.fill_diagonal(W, 0)
        L = ((np.diag(W.sum(1)) - W) / 4).astype("f")
        op = ocs.LinearOperator.from_forward(lambda v: jnp.diag(v.reshape(n, n)), n=n*n, m=n)
        family = ocs.minimize([ocs.psd(x, n), ocs.equality(op @ x - ocs.Parameter("d"))], cost=(-L).reshape(-1))
        data = {"d": np.ones(n, "f")}; arrays.update(L=L)
        Ld = L.astype(float)
        ref = cp.Problem(cp.Minimize(-cp.sum(cp.multiply(Ld, z))), [z >> 0, cp.diag(z) == 1])
        objective = lambda t: float(-np.sum(Ld * t.reshape(n, n)))
        def feasibility(t):
            X = t.reshape(n, n); sym = (X + X.T) / 2
            return max(float(np.max(np.abs(np.diag(X) - 1))), relative(np.linalg.norm(X-X.T), np.linalg.norm(X)), relative(max(0., -np.linalg.eigvalsh(sym)[0]), np.linalg.norm(sym, 2)))
    else:
        raise ValueError(kind)
    return family, data, arrays, ref, z, objective, feasibility


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, (np.generic, np.ndarray)): return clean(value.tolist())
    if isinstance(value, float) and not np.isfinite(value): return str(value)
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["lp", "qp", "socp", "quantile", "logistic", "sdp", "zero_qp", "exp", "power"], required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--alpha", type=float, default=.5)
    p.add_argument("--solve-tol", type=float, default=1e-3)
    p.add_argument("--max-blocks", type=int, default=2500)
    p.add_argument("--refine", action="store_true")
    p.add_argument("--reference-cache", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert platform.node() == "Mazeppa" and all(d.platform == "gpu" for d in jax.devices())
    assert not args.output.exists(), "Preserve previous records"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {"config":vars(args) | {"output":str(args.output),"reference_cache":None if args.reference_cache is None else str(args.reference_cache)}, "host":platform.node(),
              "devices":[str(d) for d in jax.devices()], "jax":jax.__version__, "cvxpy":cp.__version__,
              "dtype":"float32 GPU", "reference_dtype":"float64 CPU on exact rounded input arrays",
              "source_file":ocs.__file__, "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "quality_tolerance":1e-3, "status":"running", "points":[]}
    def save(): args.output.write_text(json.dumps(clean(record), indent=2)+"\n")
    save()
    try:
        family,data,arrays,ref,z,objective,feasibility = build(args.family,args.size,args.seed,args.alpha)
        input_path=args.output.with_suffix(".inputs.npz")
        np.savez(input_path,**arrays)
        record["inputs_sha256"]=hashlib.sha256(input_path.read_bytes()).hexdigest()
        cache_path=(args.reference_cache/(record["inputs_sha256"]+".json")) if args.reference_cache else None
        if cache_path is not None and cache_path.exists():
            cached=json.loads(cache_path.read_text())
            assert cached["input_sha256"]==record["inputs_sha256"]
            assert cached["family"]==args.family and cached["size"]==args.size and cached["seed"]==args.seed
            value=dict(cached["reference"])
            assert value["accepted_reference"] and value["status"]=="optimal"
            check_point=np.asarray(value["x"],dtype=float)
            assert abs(objective(check_point)-value["objective"])<=1e-10*max(abs(value["objective"]),1.)
            assert feasibility(check_point)<=1e-7
            value.update(reused_reference_path=str(cache_path),reused_reference_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),original_seconds=value.get("seconds"),seconds=0.)
            record["reference"]=value
        else:
            record["reference"]=independent_reference(ref,z,objective,feasibility,
                                                       args.size if args.family=="zero_qp" else None)
        reference_objective=record["reference"].get("objective",float("nan"))
        reference_ok=record["reference"]["accepted_reference"]
        # Only the separately named zero fixture uses an absolute objective gate.
        objective_scale=1. if args.family=="zero_qp" else abs(reference_objective)
        record["objective_error_rule"]="absolute, scale=1 for explicit zero fixture" if args.family=="zero_qp" else "abs(f-f_ref)/abs(f_ref); no additive one"
        save()
        start=time.perf_counter();comp=family.compile(tol=args.solve_tol,max_blocks=args.max_blocks)
        record["compile_family_seconds"]=time.perf_counter()-start
        start=time.perf_counter();sol=comp.solve_instance(data)
        jax.block_until_ready((sol.x,sol.y))
        solve_seconds=time.perf_counter()-start
        def audit(label,xx,yy,status,reported_residual,seconds):
            x64=np.asarray(xx,dtype=float);y64=np.asarray(yy,dtype=float)
            certificate=comp.certificate(xx,yy,data)
            residual=max(float(certificate["r_primal"]),float(certificate["r_dual"]))
            value=objective(x64);error=relative(abs(value-reference_objective),objective_scale)
            violation=feasibility(x64)
            finite=bool(np.isfinite(x64).all() and np.isfinite(y64).all() and np.isfinite(residual))
            numerical_pass=finite and residual<=1e-3 and violation<=1e-3 and error<=1e-3 and reference_ok
            accepted=numerical_pass and (status=="solved" if label=="raw" else True)
            row={"point":label,"solver_status":status,"reported_residual":reported_residual,
                 "certificate":certificate,"recomputed_residual":residual,"objective":value,
                 "objective_error":error,"feasibility":violation,"finite":finite,
                 "same_point_quality_pass":bool(numerical_pass),"accepted":bool(accepted),
                 "seconds":seconds,"status_rule":"raw public status" if label=="raw" else "refine returns a pair, not a solver status; only independently audited quality is asserted"}
            endpoint=args.output.with_name(args.output.stem+"."+label+".npz")
            np.savez(endpoint,x=x64,y=y64)
            row["endpoint_sha256"]=hashlib.sha256(endpoint.read_bytes()).hexdigest()
            record["points"].append(row);save()
        audit("raw",sol.x,sol.y,sol.status,float(sol.residual),solve_seconds)
        record["iterations"]=sol.iters
        if args.refine:
            start=time.perf_counter()
            try:
                xr,yr=comp.refine(sol.x,sol.y,data,steps=3,cg_iters=80)
                jax.block_until_ready((xr,yr))
                audit("refined",xr,yr,None,None,time.perf_counter()-start)
            except Exception as error:
                record["refinement_error"]={"error":repr(error),"seconds":time.perf_counter()-start}
        record["status"]="complete";save()
        print(json.dumps(clean({"config":record["config"],"reference":record["reference"],"points":record["points"]})),flush=True)
    except Exception as error:
        record.update(status="error",error=repr(error),traceback=traceback.format_exc());save();raise


if __name__ == "__main__": main()
