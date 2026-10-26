"""Audit immutable power-cone endpoints by an explicit feasible-point bound.

Also solve the distance problem with independent SCS for diagnostic comparison.
The constructive bound does not need a norm-minimization subproblem to certify
its status; this avoids mistaking a reference optimizer's tiny-distance warning
for an OCS cone violation. All arithmetic is independent of OCS projectors.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

import cvxpy as cp
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--cases",nargs="+",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
args=p.parse_args()
assert platform.node()=="Mazeppa" and not args.output.exists()
rows=[]
for path in args.cases:
    raw=json.loads(path.read_text());assert raw["config"]["family"]=="power"
    inputs=path.with_suffix(".inputs.npz")
    assert hashlib.sha256(inputs.read_bytes()).hexdigest()==raw["inputs_sha256"]
    with np.load(inputs) as data:
        K,l=data["K"].astype(float),data["l"].astype(float)
        alpha=float(data["alpha"])
    q=len(l)//3
    entry={"case":str(path),"case_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
           "input_sha256":raw["inputs_sha256"],"alpha_exact_float32_value":alpha,"points":[]}
    for old in raw["points"]:
        endpoint=path.with_name(path.stem+"."+old["point"]+".npz")
        assert hashlib.sha256(endpoint.read_bytes()).hexdigest()==old["endpoint_sha256"]
        with np.load(endpoint) as data:xx=data["x"]
        point=K@xx+l
        u,v,w=np.split(point,3)
        up,vp=np.maximum(u,0.),np.maximum(v,0.)
        capacity=np.zeros(q)
        positive=(up>0)&(vp>0)
        capacity[positive]=np.exp(alpha*np.log(up[positive])+(1-alpha)*np.log(vp[positive]))
        # Move inside before rounding to a float64 candidate.
        capacity*=1-128*np.finfo(float).eps
        wp=np.clip(w,-capacity,capacity)
        feasible=np.r_[up,vp,wp]
        # A separate extended-precision inequality evaluation checks membership.
        ld=np.longdouble
        high_capacity=np.zeros(q,dtype=ld)
        high_capacity[positive]=np.exp(ld(alpha)*np.log(up[positive].astype(ld))+(1-ld(alpha))*np.log(vp[positive].astype(ld)))
        candidate_in_cone=bool(np.all(up>=0) and np.all(vp>=0) and np.all(np.abs(wp.astype(ld))<=high_capacity))
        scale=max(np.linalg.norm(K@xx),np.linalg.norm(l))
        distance=float(np.linalg.norm(feasible-point))
        bound=distance/scale if scale>0 else (0. if distance==0 else float("inf"))
        z=cp.Variable(3*q)
        problem=cp.Problem(cp.Minimize(cp.norm(z-point)),[cp.PowCone3D(z[:q],z[q:2*q],z[2*q:],alpha)])
        problem.solve(solver="SCS",eps_abs=1e-9,eps_rel=1e-9,max_iters=200000)
        scs_distance=float(np.linalg.norm(np.asarray(z.value)-point))/scale if z.value is not None else None
        accepted=(raw["reference"]["accepted_reference"] and old["solver_status"]=="solved"
                  and old["finite"] and old["recomputed_residual"]<=1e-3
                  and old["objective_error"]<=1e-3 and candidate_in_cone and bound<=1e-3)
        entry["points"].append({"point":old["point"],"endpoint_sha256":old["endpoint_sha256"],
                                "prior_accepted":old["accepted"],"prior_feasibility":old["feasibility"],
                                "cone_point":point.tolist(),"feasible_candidate":feasible.tolist(),
                                "candidate_membership_check_longdouble":candidate_in_cone,
                                "distance_upper_bound":distance,"normalization":float(scale),
                                "normalized_distance_upper_bound":bound,
                                "scs_projection_status":problem.status,"scs_projection_distance":scs_distance,
                                "objective_error":old["objective_error"],
                                "recomputed_residual_on_same_pair":old["recomputed_residual"],
                                "accepted_after_independent_feasibility_audit":bool(accepted)})
    rows.append(entry)
out={"host":platform.node(),"cvxpy":cp.__version__,
     "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     "scope":"same saved endpoints; independent numerical feasible-point upper bound, not an OCS projector or a new solve",
     "longdouble_mantissa_bits":int(np.finfo(np.longdouble).nmant),"rows":rows}
args.output.write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out))
