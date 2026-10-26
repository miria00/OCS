"""Bounded remote-5090 controls for compact orbit-tied metric construction.

SciPy/NumPy float64 references run on the same authorized host; their times
are never solver performance comparisons. Every control writes its record
immediately, including failures. Tiny dense matrices are reference fixtures,
never the constructor interface.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize, brentq

import mixed_metric_gpu as old
import shared_shape_gpu as new


def symf(a, fn):
    d, q = np.linalg.eigh((a+a.T)/2)
    return (q*fn(d))@q.T


def error(a, b):
    return max(float(np.linalg.norm(np.asarray(x)-y)/max(1., np.linalg.norm(y)))
               for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    args = p.parse_args(); output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    assert jax.devices()[0].platform == 'gpu' and '5090' in jax.devices()[0].device_kind
    jax.config.update('jax_enable_x64', False)
    report = {'host': subprocess.check_output(['hostname'], text=True).strip(),
              'gpu': subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi', '--query-gpu=uuid,name',
                                               '--format=csv,noheader'], text=True).strip(),
              'jax': jax.__version__, 'main_dtype': 'float32 GPU',
              'reference_dtype': 'float64 CPU on same host', 'checks': [],
              'source_hashes': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in
                  [Path(__file__), Path(__file__).with_name('shared_shape_gpu.py'),
                   Path(__file__).with_name('mixed_metric_gpu.py')]}}
    archive = {}; started = time.perf_counter()

    def check(name, passed, **details):
        report['checks'].append(dict(name=name, passed=bool(passed), **details))
        report['elapsed_seconds'] = time.perf_counter()-started
        output.write_text(json.dumps(report, indent=2))
        np.savez(output.with_suffix('.npz'), **archive)
        print(json.dumps(report['checks'][-1]), flush=True)
        assert passed, name

    rng = np.random.default_rng(5109)
    rows = (new.SharedShapeGroup(3, 7), old.BlockGroup('scalar', 2, 2),
            old.BlockGroup('diagonal', 2, 2))
    m = sum(g.dimension for g in rows)
    w = rng.normal(size=(2, m)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    a = (q*np.exp([-1., .2, .7]))@q.T
    state = (jnp.asarray(a, jnp.float32), jnp.asarray([.1, -.2], jnp.float32),
             jnp.asarray([[.2, -.1], [.3, .4]], jnp.float32))
    archive.update(step_samples=w, initial_shared_matrix=a)
    ref_gram = np.einsum('qni,qnj->ij', w[:, :21].reshape(2, 7, 3).astype(np.float64),
                         w[:, :21].reshape(2, 7, 3).astype(np.float64))/(2*7)
    gram = new.pooled_gram(jnp.asarray(w[:, :21]), rows[0])
    check('pooled Gram without block Gram storage', error(gram, ref_gram) < 2e-6,
          relative_error=error(gram, ref_gram), output_shape=list(gram.shape))
    eta, target, mu, box = .17, .8, 1., 2.
    half = symf(a, np.sqrt); loga = symf(a, np.log)
    tangent = .5*half@ref_gram@half-.5*target*np.eye(3)+mu/4*loga
    unconstrained = half@symf(-eta*tangent, np.exp)@half
    reference = symf(unconstrained, lambda d: np.clip(d, np.exp(-4), np.exp(4)))
    result, finite = jax.jit(lambda st, sample: new.update_side(rows, st, sample, target, mu, box, eta))(
        state, jnp.asarray(w))
    check('shared update multiplicity and noncommuting float64 reference',
          bool(finite) and error(result[0], reference) < 2e-5, relative_error=error(result[0], reference))
    for power in (-1., .5, 1.):
        v = rng.normal(size=m).astype(np.float32)
        got = new.apply_power(rows, state, jnp.asarray(v), power)
        ref = np.concatenate([(v[:21].reshape(7, 3)@symf(a, lambda d: d**power).T).reshape(-1),
            (v[21:25].reshape(2, 2)*np.exp(power*np.asarray(state[1]))[:, None]).reshape(-1),
            (v[25:].reshape(2, 2)*np.exp(power*np.asarray(state[2]))).reshape(-1)])
        check(f'compact apply power {power}', error(got, ref)<2e-5, relative_error=error(got, ref))
    alpha, shape = new.shared_shape_parts(result[0])
    check('det-one decomposition', abs(np.linalg.det(np.asarray(shape, np.float64))-1)<2e-5
          and error(np.exp(float(alpha))*np.asarray(shape), result[0])<2e-5,
          determinant=float(np.linalg.det(np.asarray(shape, np.float64))))

    # Spectral box: compare to the exact scalar clipping problem in log eigenvalues.
    outside = symf(a, lambda d: d**8)
    projection = new._spectral(jnp.asarray(outside, jnp.float32),
                               lambda d: jnp.clip(d, jnp.exp(-4), jnp.exp(4)))
    pref = symf(outside, lambda d: np.clip(d, np.exp(-4), np.exp(4)))
    check('exact tied-class spectral box projection', error(projection, pref)<5e-5,
          relative_error=error(projection, pref))

    # Count-one compact representation must follow the old FULL trajectory.
    dense = (rng.normal(size=(3, 4))*.2).astype(np.float32); kj = jnp.asarray(dense)
    rr = (new.SharedShapeGroup(3, 1),); ro = (old.BlockGroup('full', 3, 1),)
    cc = (old.BlockGroup('diagonal', 1, 4),)
    cfg = dict(iterations=12, probes=2, mu=1., box=2., shift=20)
    key = jax.random.PRNGKey(8)
    out = new.make_constructor(lambda x: kj@x, lambda y: kj.T@y, rr, cc, **cfg)(
        key, new.initial_state(rr), new.initial_state(cc))
    oldout = old.make_constructor(lambda x: kj@x, lambda y: kj.T@y, ro, cc, **cfg)(
        key, old.initial_state(ro), old.initial_state(cc))
    jax.block_until_ready((out, oldout))
    ee = error((out['A'][0], out['B']), (oldout['A'][0][0], oldout['B']))
    check('count-one FULL reduction, identical fresh probes and pre-update averaging',
          ee<3e-5 and bool(out['finite']) and int(out['operator_calls'])==48,
          relative_error=ee, operator_calls=int(out['operator_calls']))

    # Scalar tied row is the k=1 compact class; N multiplicity survives exactly.
    dense2 = (rng.normal(size=(6, 4))*.2).astype(np.float32); k2=jnp.asarray(dense2)
    rn = (new.SharedShapeGroup(1, 6),); rs=(old.BlockGroup('scalar', 6, 1),)
    on = new.make_constructor(lambda x:k2@x,lambda y:k2.T@y,rn,cc,**cfg)(key,new.initial_state(rn),new.initial_state(cc))
    os = old.make_constructor(lambda x:k2@x,lambda y:k2.T@y,rs,cc,**cfg)(key,old.initial_state(rs),old.initial_state(cc))
    ee = error((jnp.log(on['A'][0]).reshape(1), on['B']), (os['A'][0],os['B']))
    check('k=1 exact scalar reduction including group multiplicity',ee<3e-5,relative_error=ee)

    # Exhaust every independent Rademacher coordinate to verify oracle unbiasedness.
    signs = np.asarray(list(itertools.product([-1., 1.], repeat=4)))
    samples = signs@dense.astype(np.float64).T
    sampled = new.pooled_gram(jnp.asarray(samples, jnp.float32), rr[0])
    exact = dense.astype(np.float64)@dense.astype(np.float64).T
    ee=error(sampled,exact)
    check('exact-enumeration unbiased forward-Gram oracle',ee<2e-6,relative_error=ee,probes=16)

    # Float64 objective minimization, independently parameterized by symmetric logs.
    N,k = 3,2
    blur=np.array([[.7,.2,.1],[.1,.7,.2],[.2,.1,.7]])
    mix=np.array([[1.,.4],[-.15,.7]])
    K=np.empty((N*k,N*k))
    for pixel in range(N):
        for channel in range(k):
            for c in range(k): K[pixel*k+channel,c*N:(c+1)*N]=mix[channel,c]*blur[pixel]
    basis=[]
    for i in range(k):
        for j in range(i,k):
            z=np.zeros((k,k));z[i,j]=1;z[j,i]=1;basis.append(z)
    def objective(z,tied):
        nr=1 if tied else N; nc=k if tied else N*k; t=len(basis)
        rowlogs=z[:nr*t].reshape(nr,t); blog=z[nr*t:]; bv=np.repeat(blog,N) if tied else blog
        G=(K*np.exp(bv)[None,:])@K.T
        value=0.;gradrow=[]; colenergy=np.zeros(N*k)
        for i in range(nr):
            X=sum(x*b for x,b in zip(rowlogs[i],basis)); A=expm(X)
            ids=range(N) if tied else [i]
            pooled=sum(G[p*k:(p+1)*k,p*k:(p+1)*k] for p in ids)
            mult=N if tied else 1
            value += .5*np.trace(A@pooled)-.5*mult*np.trace(X)+mult/8*np.sum(X*X)
            egrad=.5*expm_frechet(X,pooled,compute_expm=False)-.5*mult*np.eye(k)+mult/4*X
            gradrow.extend(np.sum(egrad*b) for b in basis)
            for pixel in ids:
                Ki=K[pixel*k:(pixel+1)*k]
                colenergy+=np.sum(Ki*(A@Ki),axis=0)
        value+=-.5*np.sum(bv)+np.sum(bv*bv)/8
        gb=.5*np.exp(bv)*colenergy-.5+bv/4
        if tied:gb=gb.reshape(k,N).sum(axis=1)
        return value,np.r_[gradrow,gb]
    fits=[]
    for tied in (False,True):
        dimension=(len(basis)+k) if tied else N*len(basis)+N*k
        fit=minimize(lambda z:objective(z,tied),np.zeros(dimension),jac=True,method='BFGS',
                     options={'gtol':1e-10,'maxiter':500})
        fits.append(fit)
        archive['optimum_tied' if tied else 'optimum_untied']=fit.x
        check('tiny tied optimum stationarity' if tied else 'tiny unrestricted optimum stationarity',
              np.max(np.abs(fit.jac))<2e-7, gradient_infinity=float(np.max(np.abs(fit.jac))),
              objective=float(fit.fun), scipy_success=bool(fit.success))
    check('F2 translation symmetry: identical independently optimized objective',
          abs(fits[0].fun-fits[1].fun)<1e-9,objective_difference=float(abs(fits[0].fun-fits[1].fun)))
    free_row=fits[0].x[:N*len(basis)].reshape(N,len(basis));free_b=fits[0].x[N*len(basis):].reshape(k,N)
    variation=max(np.max(np.ptp(free_row,axis=0)),np.max(np.ptp(free_b,axis=1)))
    check('unrestricted optimum itself has tied row and channel-constant column metrics',variation<2e-7,
          max_log_coordinate_variation=float(variation))
    # Fixed-B conditional minimizer: root equation uses average Gram, not sum.
    gg=(K@K.T).reshape(N,k,N,k);gbar=sum(gg[i,:,i,:] for i in range(N))/N
    eig,q=np.linalg.eigh(gbar)
    roots=np.array([brentq(lambda z:.5*v*np.exp(z)-.5+z/4,-4,4) for v in eig])
    stationary=.5*eig*np.exp(roots)-.5+roots/4
    check('full conditional spectral roots use pooled mean Gram',np.max(np.abs(stationary))<1e-12,
          stationarity=float(np.max(np.abs(stationary))))
    # Direct oracle construction on the tiny symmetry fixture, same configured
    # 80-step schedule as the performance campaign. Report construction error;
    # a finite SGD endpoint is not called the exact optimum.
    tiny=jnp.asarray(K,jnp.float32)
    tr=(new.SharedShapeGroup(k,N),);tc=(old.BlockGroup('scalar',N,k),)
    construct=new.make_constructor(lambda x:tiny@x,lambda y:tiny.T@y,tr,tc)
    answer=construct(jax.random.PRNGKey(0),new.initial_state(tr),new.initial_state(tc))
    X=symf(np.asarray(answer['A'][0],np.float64),np.log)
    coords=np.array([X[i,j] for i in range(k) for j in range(i,k)])
    z=np.r_[coords,np.asarray(answer['B'][0],np.float64)]
    fv=objective(z,True)[0];f0=objective(np.zeros_like(z),True)[0]
    check('80-step compact constructor improves tiny objective',bool(answer['finite']) and fv<f0,
          objective=float(fv),initial_objective=float(f0),exact_optimum=float(fits[1].fun),
          suboptimality=float(fv-fits[1].fun),operator_calls=int(answer['operator_calls']))
    archive.update(tiny_construct_A=np.asarray(answer['A'][0]),tiny_construct_B=np.asarray(answer['B'][0]))
    # Shared global signs need not imply perfectly correlated outputs. The
    # exact Rademacher trace variance depends on the whole operator Gram.
    signs6=np.asarray(list(itertools.product([-1.,1.],repeat=N*k)))
    pooled_trace=np.sum((signs6@K.T)**2,axis=1)/N
    C=K.T@K/N
    variance_formula=2*(np.sum(C*C)-np.sum(np.diag(C)**2))
    measured_variance=float(np.var(pooled_trace))
    effective_rank=float(np.trace(C)**2/np.sum(C*C))
    check('pooled oracle variance matches exact global-probe covariance',
          abs(measured_variance-variance_formula)<1e-12,variance=measured_variance,
          analytic_variance=float(variance_formula),effective_rank=effective_rank,
          comment='No universal 1/N or no-1/N variance assertion')
    archive.update(tiny_K=K, tiny_conditional_logs=roots, trajectory_A=np.asarray(out['A'][0]),
                   trajectory_B=np.asarray(out['B'][0]), exact_gram=exact)

    # A rank-deficient operator and zero operator remain well-defined with ridge.
    for label, mat in [('rank_one', np.ones((3,4),np.float32)*.1),('zero', np.zeros((3,4),np.float32))]:
        jj=jnp.asarray(mat)
        run=new.make_constructor(lambda x:jj@x,lambda y:jj.T@y,rr,cc,iterations=80)
        ans=run(key,new.initial_state(rr),new.initial_state(cc));jax.block_until_ready(ans)
        check(f'{label} operator positive-ridge well-posed numerical control',bool(ans['finite']) and int(ans['status'])==0,
              smallest_eigenvalue=float(np.linalg.eigvalsh(np.asarray(ans['A'][0]))[0]))

    # Explicit invalid/overflow guards. They must not silently yield success.
    run=new.make_constructor(lambda x:kj@x,lambda y:kj.T@y,rr,cc,iterations=2)
    invalid=run(key,(-jnp.eye(3,dtype=jnp.float32),),new.initial_state(cc))
    check('invalid initial SPD rejected via status',int(invalid['status'])==1 and not bool(invalid['finite']),
          status=int(invalid['status']))
    huge=new.make_constructor(lambda x:jnp.full((3,),jnp.inf),lambda y:jnp.full((4,),jnp.inf),rr,cc,iterations=2)
    bad=huge(key,new.initial_state(rr),new.initial_state(cc))
    check('nonfinite oracle is explicit failure',int(bad['status'])==2 and not bool(bad['finite']),status=int(bad['status']))
    zero=new.make_constructor(lambda x:kj@x,lambda y:kj.T@y,rr,cc,iterations=3,box=0.)(
        key,new.initial_state(rr),new.initial_state(cc))
    check('zero-width spectral box is identity',error(zero['A'],new.initial_state(rr))<2e-5
          and error(zero['B'],new.initial_state(cc))<2e-5)
    guards=0
    for action in [lambda:new.SharedShapeGroup(3,10,amplitudes='free'),lambda:new.SharedShapeGroup(0,1),
                   lambda:new.make_constructor(lambda x:x,lambda x:x,rr,cc,mu=0),
                   lambda:new.make_constructor(lambda x:x,lambda x:x,rr,cc,iterations=0),
                   lambda:run(key,(),new.initial_state(cc)),
                   lambda:run(key,(jnp.eye(2,dtype=jnp.float32),),new.initial_state(cc))]:
        try:action()
        except (ValueError,NotImplementedError):guards+=1
    check('declaration guards including unsupported free amplitudes',guards==6,guards=guards)
    report['passed']=all(c['passed'] for c in report['checks'])
    report['compact_f2_state_bytes_1024']=new.state_storage_bytes((new.SharedShapeGroup(3,1024**2),
                                                                  old.BlockGroup('scalar',1024**2,3)))
    report['limitations']=['Tied amplitudes only; free-amplitude joint projection not implemented.',
        'Numerical checks are not a proof or an interval-certified float32 convergence guarantee.',
        'No public full-metric solver integration or whole-solver speed claim.']
    report['elapsed_seconds']=time.perf_counter()-started
    output.write_text(json.dumps(report,indent=2));np.savez(output.with_suffix('.npz'),**archive)
    print(json.dumps({'passed':report['passed'],'checks':len(report['checks']),
                      'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
