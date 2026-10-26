"""Installed wheel smoke test, executed outside the checkout on an authorized host."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import ocs

output = Path(sys.argv[1])
prefix = Path(sys.prefix).resolve()
module = Path(ocs.__file__).resolve()
assert prefix in module.parents, (prefix, module)
record = {"python":sys.version,"prefix":str(prefix),"ocs_file":str(module),
          "ocs_version":importlib.metadata.version("ocsolver"),"jax":jax.__version__,
          "devices":[str(d) for d in jax.devices()],"results":[],"status":"running",
          "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
weights = jnp.array([1.0, 2.0, 3.0])
K = ocs.LinearOperator.from_forward(lambda v: weights * v, n=3, m=3)
x = ocs.Variable(3)
b = ocs.Parameter("b")
family = ocs.minimize([ocs.nonneg(x), ocs.least_squares(K @ x - b)])
compiled = family.compile(tol=1e-3)

def check(label, comp, sol, bvalue, elapsed):
    expected=np.maximum(np.asarray(bvalue),0)/np.asarray(weights)
    xx=np.asarray(sol.x,dtype=np.float64)
    obj=float(.5*np.sum((np.asarray(weights)*xx-bvalue)**2))
    reference=float(.5*np.sum(np.minimum(bvalue,0)**2))
    cert=comp.certificate(sol.x,sol.y,{"b":jnp.asarray(bvalue)})
    residual=max(float(cert["r_primal"]),float(cert["r_dual"]))
    error=abs(obj-reference)/abs(reference)
    accepted=sol.status=="solved" and residual<=1e-3 and error<=1e-3 and np.min(xx)>=-1e-6
    record["results"].append({"label":label,"status":sol.status,"residual":residual,
                              "objective":obj,"reference":reference,"objective_error":error,
                              "x":xx.tolist(),"expected":expected.tolist(),"seconds":elapsed,
                              "accepted":bool(accepted)})
    output.write_text(json.dumps(record,indent=2)+"\n")
    assert accepted,record["results"][-1]

start=time.perf_counter();sol=compiled.solve_instance({"b":jnp.array([-1.,2.,3.])});jax.block_until_ready((sol.x,sol.y))
check("README hello world",compiled,sol,np.array([-1.,2.,3.]),time.perf_counter()-start)
batch=jnp.array([[-1.,2.,3.],[2.,-2.,6.]])
start=time.perf_counter();solutions=compiled.solve_many_results({"b":batch});jax.block_until_ready([(s.x,s.y) for s in solutions]);elapsed=time.perf_counter()-start
for i,sol in enumerate(solutions):check("README batch row "+str(i),compiled,sol,np.asarray(batch[i]),elapsed)
compact=family.compile(data_layout="compact")
start=time.perf_counter();sol=compact.solve_instance({"b":jnp.array([-1.,2.,3.])});jax.block_until_ready((sol.x,sol.y))
check("README compact",compact,sol,np.array([-1.,2.,3.]),time.perf_counter()-start)

import cvxpy as cp
from ocs.cvxpy_harness import validate_against_cvxpy
z=cp.Variable(2)
problem=cp.Problem(cp.Minimize(cp.sum_squares(z-[1,2])+7),[z>=3])
report=validate_against_cvxpy(problem,tol=1e-5)
record["cvxpy_readme"]={k:v for k,v in report.items() if isinstance(v,(str,bool,int,float,type(None)))}
assert report["objective_agreement"],report
record["passed"]=True
record["status"]="complete"
record["installed_ocs_hashes"]={str(p.relative_to(module.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(module.parent.rglob("*.py"))}
output.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record),flush=True)
