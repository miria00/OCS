"""A bounded TV block can coexist with a free-row interior anchor."""
import numpy as np
import jax.numpy as jnp
import pytest
import ocs

@pytest.mark.parametrize('dc',[False,True])
def test_free_row_anchor_requires_positive_adjoint_and_keeps_tv_domain(dc):
    n=64;idx=np.array([0,3,7,11] if dc else [2,3,7,11]);m=len(idx)
    x=ocs.Variable(n);F=ocs.FFTMask(idx,n);G=ocs.Gradient3D(4,4,4)
    comp=ocs.minimize([ocs.nonneg(x),ocs.least_squares(F@x-ocs.Parameter('b')),
                      ocs.group_l2(G@x,lam=.02,group_size=3)]).compile(max_blocks=0)
    rng=np.random.default_rng(18);y=np.r_[rng.normal(size=2*m),rng.normal(size=3*n)].astype('f')
    b=(-10*y[:2*m]).astype('f')
    report=comp.objective_bound(jnp.ones(n),jnp.asarray(y),{'b':jnp.asarray(b)},return_dual=True)
    assert comp.meta['bound_anchor_setup_forward_actions']==1
    assert report['anchor_available']==dc
    if report['objective_lower_bound_available']:
        yy=np.asarray(report['certifying_dual']).astype('d');u=yy[:2*m];w=yy[2*m:].reshape(4,4,4,3)
        f=np.zeros(n,'complex128');f[idx]=u[:m]+1j*u[m:];a=np.fft.ifft(f,norm='ortho').real.reshape(4,4,4)
        for k in range(3):
            lo=[slice(None)]*3;hi=lo.copy();lo[k]=slice(0,-1);hi[k]=slice(1,None);lo=tuple(lo);hi=tuple(hi)
            a[lo]-=w[...,k][lo];a[hi]+=w[...,k][lo]
        if dc:
            assert report['source']=='feasible_anchor'
            assert a.min()>=0.
            assert np.linalg.norm(w,axis=-1).max()<=float(np.float32(.02))
        else:
            # Without DC the coupling must vanish exactly; the named
            # equality-projected route certifies the implicitly corrected
            # witness, whose ball margin is ||coupling||_2 / s_D.
            assert report['source']=='equality_projected'
            s_D=2*np.sin(np.pi/8)
            assert np.linalg.norm(w,axis=-1).max()+np.linalg.norm(a)/s_D<=float(np.float32(.02))*(1+1e-6)
            assert abs(a.sum())<=1e-12
