"""Independent float64 audits of archived cross-family GPU endpoints."""
from pathlib import Path
import json
import os


def main():
    if os.uname().nodename!="Mazeppa":raise RuntimeError("Audit stays on Mazeppa")
    import numpy as np
    root=Path.home()/"ocs_research_20260914/weight"
    report=[]
    for directory in sorted((root/"results").glob("feature128_s2_*_gate1e3")):
        result=json.loads((directory/"result.json").read_text())
        inputs=np.load(directory/"inputs.npz");endpoint=np.load(directory/"endpoint.npz")
        A=inputs["A"].astype(float);B=inputs["B"].astype(float)
        T=B.shape[1];X=endpoint["x"].astype(float).reshape(A.shape[1],T)
        Y=endpoint["y"].astype(float).reshape(B.shape);lam=result["model"]["lam"]
        primal=.5*np.sum((A@X-B)**2)+lam*np.linalg.norm(X,axis=1).sum()
        bounds=[]
        for label,dual in [("returned dual",Y),("smooth gradient",A@X-B)]:
            adj=A.T@dual;maximum=np.linalg.norm(adj,axis=1).max()
            scale=min(1.,lam/max(maximum,np.finfo(float).tiny))*(1-1e-12)
            dual=dual*scale;feas=np.linalg.norm(A.T@dual,axis=1).max()
            lower=-.5*np.sum(dual*dual)-np.sum(B*dual)
            gap=(primal-lower)/max(abs(primal),abs(lower))
            bounds.append(dict(source=label,scale=float(scale),dual_max_group_norm=float(feas),
                dual_objective=float(lower),relative_gap=float(gap),feasible=bool(feas<=lam)))
        valid=[v for v in bounds if v["feasible"]]
        best=min(valid,key=lambda v:v["relative_gap"])
        report.append(dict(run=directory.name,reference_status=result["model"]["reference_status"],
            primal_objective=float(primal),bounds=bounds,
            independent_gap_pass=bool(-1e-12<=best["relative_gap"]<=1e-3),
            best_relative_gap=best["relative_gap"]))
    out=root/"reports"/"cross_family_independent_audit.json"
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=="__main__":main()
