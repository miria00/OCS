"""Remote-only Gate-1 algebra, native fidelity and L16 objective references."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize

from blockmetric_models import blockmetric_f2
import ocs.solver.equilibrate_sgd as native
import shared_shape_gpu as shared
from mixed_metric_gpu import BlockGroup
from attribution_constructor import construct, diagonal_row_update, make_diagonal_constructor, tied_smfe_function


def reference(op, kind):
    """Deterministic stationary optimum of the reduced joint F2 objective.

    Rounded stored M,H are used. The optimum is interior to the declared box;
    symmetry also makes its free-column first derivatives constant per channel.
    This is a numerical reference, not an interval optimum certificate.
    """
    k = op.c
    mix = np.asarray(op.M, np.float64)
    h2 = float(np.mean(np.abs(np.asarray(op.H, np.complex128))**2))
    basis = [np.eye(k)] if kind == 'scalar' else [np.diag(np.eye(k)[i]) for i in range(k)]
    if kind == 'full':
        basis = []
        for i in range(k):
            for j in range(i, k):
                e = np.zeros((k,k)); e[i,j] = e[j,i] = 1.; basis.append(e)
    def objective(z):
        X = sum(v*e for v,e in zip(z[:len(basis)],basis)); blog = z[len(basis):]
        A = expm(X); b = np.exp(blog); G = h2*(mix*b[None,:])@mix.T
        value = .5*np.trace(A@G)-.5*(np.trace(X)+blog.sum())+(np.sum(X*X)+blog@blog)/8
        gx = .5*expm_frechet(X,G,compute_expm=False)-.5*np.eye(k)+X/4
        gb = .5*b*h2*np.diag(mix.T@A@mix)-.5+blog/4
        return value, np.r_[[np.sum(gx*e) for e in basis],gb]
    fit = minimize(objective, np.zeros(len(basis)+k), jac=True, method='BFGS',
                   options={'gtol':1e-10,'maxiter':500})
    X = sum(v*e for v,e in zip(fit.x[:len(basis)],basis))
    return fit, objective, basis, X


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    assert jax.devices()[0].platform=='gpu' and '5090' in jax.devices()[0].device_kind
    jax.config.update('jax_enable_x64',False)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    report={'checks':[],'scope':'GPU float32 and CPU float64 on the same authorized remote host'}
    arrays={};started=time.perf_counter()
    def check(name,passed,**details):
        report['checks'].append(dict(name=name,passed=bool(passed),**details))
        report['elapsed_seconds']=time.perf_counter()-started
        out.write_text(json.dumps(report,indent=2)+'\n');np.savez(out.with_suffix('.npz'),**arrays)
        print(json.dumps(report['checks'][-1]),flush=True)
        assert passed,name
    rng=np.random.default_rng(1515)
    # Explicit outside-box example prevents mean-after-clipping regression.
    tentative=jnp.asarray([[3.,-1.],[-5.,2.]])
    projected=jnp.repeat(jnp.clip(jnp.mean(tentative,axis=1),-1.,1.),2)
    check('exact tied column projection averages before clipping',
          np.allclose(np.asarray(projected),[1.,1.,-1.,-1.]))
    _,op_spec=blockmetric_f2(16,mixing='moderate',seed=0)
    op=op_spec.dual_blocks[0]['op'];k,count,n=op.c,op.N,op.shape[1]
    # One-channel grouping covers every individual column: must be native.
    matrix=jnp.asarray(rng.normal(size=(4,4))*.2,jnp.float32)
    mv=lambda x:matrix@x;rv=lambda x:matrix.T@x
    fn,_,source=tied_smfe_function(4,1)
    old=native.smfe_equilibration(mv,rv,4,4,[(0,4)],iters=12,probes=2,gamma=1.,M=2.)
    new=fn(mv,rv,4,4,[(0,4)],iters=12,probes=2,gamma=1.,M=2.)
    error=max(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(old,new))
    check('S1 no-tie case equals native S0 including norm estimator',error<2e-6,max_abs_error=error)
    (out.parent/'generated_tied_smfe.py').write_text(source)
    # Full 80-step native-state extraction matches independently invoked source.
    a,b,timing,contract,_=construct(op,'S0')
    tau,sigma,norm=native.smfe_equilibration(op.matvec,op.rmatvec,n,n,[(0,n)],
        iters=80,probes=2,gamma=1.,M=2.,safety=.9,seed=0)
    sc=.9/(norm+1e-30)
    err=max(np.max(np.abs(b-np.asarray(tau,np.float64)/sc)),abs(a[0,0]-float(np.asarray(sigma)[0]/sc)))
    check('S0 captured source is numerically identical to native 80-step call',err==0.,max_abs_error=float(err))
    # Same physical samples, independently evaluated pooled diagonal tangent.
    loga=np.array([.2,-.1,.3],np.float32);samples=rng.normal(size=(2,n)).astype(np.float32)
    eta=.17
    energy=np.mean(samples.astype(np.float64).reshape(2,count,k)**2,axis=(0,1))
    expected=np.clip(loga-eta*(.5*np.exp(loga.astype(np.float64))*energy-.5+loga/4),-4,4)
    actual,finite=diagonal_row_update(jnp.asarray(loga),jnp.asarray(samples),eta=eta)
    err=float(np.max(np.abs(np.asarray(actual)-expected)))
    check('D constrained tangent uses pooled multiplicity N',bool(finite) and err<2e-6,max_abs_error=err)
    # One-step common-stream S2 versus D projection at identity. Mean of the
    # unclipped D tangent agrees with S2's scalar tangent (before box clipping).
    key=jax.random.PRNGKey(0);_,rk,ck=jax.random.split(key,3)
    xi=jax.random.rademacher(rk,(2,n),dtype=jnp.float32)
    zeta=jax.random.rademacher(ck,(2,n),dtype=jnp.float32)
    f=jax.vmap(op.matvec)(xi);reverse=jax.vmap(op.rmatvec)(zeta)
    eta=8/21
    dnew,_=diagonal_row_update(jnp.zeros(k),f,eta=eta)
    scalar=-eta*(.5*float(np.mean(np.asarray(f,np.float64)**2))-.5)
    check('common physical probes scalar/diagonal gradient reduction',
          abs(float(jnp.mean(dnew))-scalar)<2e-6,difference=abs(float(jnp.mean(dnew))-scalar))
    columns=(BlockGroup('scalar',count,k),);rows=(shared.SharedShapeGroup(k,count),)
    fresult=shared.make_constructor(op.matvec,op.rmatvec,rows,columns,iterations=1)(
        key,shared.initial_state(rows),shared.initial_state(columns))
    dresult=make_diagonal_constructor(op,iterations=1)(key,jnp.zeros(k),jnp.zeros(k))
    berr=float(np.max(np.abs(np.asarray(fresult['last_B'][0])-np.asarray(dresult['last_B'][0]))))
    check('D/F simultaneous common-probe column update at identity',berr<2e-6,max_abs_error=berr)
    report['reference']={}
    for kind,arms in [('scalar',['S0','S1','S2']),('diagonal',['D']),('full',['F'])]:
        fit,obj,basis,X=reference(op,kind)
        grad=float(np.max(np.abs(fit.jac)));eigs=np.linalg.eigvalsh(X)
        interior=max(np.max(np.abs(eigs)),np.max(np.abs(fit.x[len(basis):])))<4-1e-6
        check(kind+' deterministic rounded-data reference',grad<2e-7 and interior,
              gradient_infinity=grad,objective_per_pixel=float(fit.fun),scipy_success=bool(fit.success),box_interior=bool(interior))
        report['reference'][kind]={'objective_per_pixel':float(fit.fun),'x':fit.x.tolist(),'gradient_infinity':grad,
            'scope':'numerical stationary optimum; S0/S1/S2 share exact optimum by translation symmetry'}
        arrays['reference_'+kind]=fit.x
        for arm in arms:
            A,B,timing,contract,_=construct(op,arm)
            eigen,Q=np.linalg.eigh(A);loga=(Q*np.log(eigen))@Q.T;bl=np.log(B).reshape(k,count)
            if arm=='S0':
                # Exact scalar-row/free-column F2 objective and free gradients.
                alpha=A[0,0];mix=np.asarray(op.M,np.float64);h2=float(np.mean(np.abs(np.asarray(op.H,np.complex128))**2))
                colcoef=alpha*h2*np.sum(mix*mix,axis=0)
                value=.5*np.sum(colcoef[:,None]*np.exp(bl))/count-.5*(np.trace(loga)+bl.sum()/count)+(np.sum(loga*loga)+np.sum(bl*bl)/count)/8
            else:
                coordinates=np.array([loga[0,0]]) if kind=='scalar' else np.diag(loga) if kind=='diagonal' else np.array([loga[i,j] for i in range(k) for j in range(i,k)])
                value=obj(np.r_[coordinates,bl[:,0]])[0]
            check(arm+' 80-step finite declared class and reference bound',
                  np.isfinite(value) and value>=fit.fun-2e-6 and np.all(eigen>0) and
                  (arm=='S0' or np.max(np.ptp(bl,axis=1))<1e-7),
                  objective_per_pixel=float(value),reference_gap=float(value-fit.fun),construction_seconds=timing,
                  column_log_spatial_range=float(np.max(np.ptp(bl,axis=1))))
            arrays[arm+'_A']=A;arrays[arm+'_B']=B
    report['all_passed']=True;out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
