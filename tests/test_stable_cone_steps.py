"""Independent high-precision references for stable cone-construction steps."""
import decimal
import numpy as np
import pytest
from scipy.linalg import expm
import jax
import jax.numpy as jnp
from ocs.solver._cone_construct_steps import soc_step, psd_bounded_step
D = decimal.Decimal

def soc_ref(value, tangent, eta):
    # Direct exponentiation and subtraction in high precision: deliberately
    # does not use the production log-determinant reconstruction.
    with decimal.localcontext() as ctx:
        ctx.prec = 700
        v = [D.from_float(float(x)) for x in value]
        h = [D.from_float(float(x)) for x in tangent]
        e = D.from_float(float(eta))
        def spectral(x, fn):
            norm = sum(a*a for a in x[1:]).sqrt()
            a, b = fn(x[0]+norm), fn(x[0]-norm)
            direction = [t/norm if norm else D(0) for t in x[1:]]
            return [(a+b)/2] + [(a-b)*t/2 for t in direction]
        half = spectral(v, lambda t: t.sqrt())
        step = spectral(h, lambda t: (-e*t).exp())
        det = half[0]**2 - sum(t*t for t in half[1:])
        dot = sum(a*b for a,b in zip(half, step))
        trial = [2*a*dot - det*b*(1 if i==0 else -1)
                 for i,(a,b) in enumerate(zip(half,step))]
        lo, hi = D(-2).exp(), D(2).exp()
        clipped = spectral(trial, lambda t: min(hi,max(lo,t)))
        return np.array([float(x) for x in clipped])


@pytest.mark.parametrize("k", [3, 9, 33])
@pytest.mark.parametrize("scale", [.1, 10., 100.])
def test_soc_large_exponent_against_700_digit_direct_reference(k, scale):
    rng = np.random.default_rng(46191+k)
    direction = rng.normal(size=k-1); direction /= np.linalg.norm(direction)
    eig = np.exp(np.array([-1.9, 1.9]))
    value = np.r_[.5*sum(eig), .5*(eig[1]-eig[0])*direction].astype('f')
    h = (scale*rng.normal(size=k)).astype('f'); eta=np.float32(.3)
    expected=soc_ref(value,h,eta)
    out,valid=jax.jit(soc_step)(value[None],h[None],eta)
    assert bool(valid)
    assert np.linalg.norm(np.asarray(out[0])-expected)/np.linalg.norm(expected)<5e-5

@pytest.mark.parametrize("k", [2, 3, 8, 16])
def test_psd_noncommuting_large_step_and_64_copy_consistency(k):
    rng=np.random.default_rng(719+k)
    q,_=np.linalg.qr(rng.normal(size=(k,k)))
    value=((q*np.exp(np.linspace(-2,2,k)))@q.T).astype('f')
    h=rng.normal(size=(k,k)); h=(1e4*(h+h.T)).astype('f')
    f=jax.jit(psd_bounded_step)
    out,valid,eta=f(value[None],h[None],np.float32(.3))
    assert bool(valid) and 0<float(eta[0])<.3
    vals,vecs=np.linalg.eigh(value.astype(float)); half=(vecs*np.sqrt(vals))@vecs.T
    trial=half@expm(-float(eta[0])*h.astype(float))@half
    ev,eq=np.linalg.eigh(.5*(trial+trial.T))
    expected=(eq*np.clip(ev,np.exp(-2),np.exp(2)))@eq.T
    assert np.linalg.norm(np.asarray(out[0])-expected)/np.linalg.norm(expected)<5e-5
    stacked,ok,_=f(np.repeat(value[None],64,0),np.repeat(h[None],64,0),np.float32(.3))
    assert bool(ok)
    assert np.max(np.linalg.norm(np.asarray(stacked)-expected,axis=(-2,-1)))/np.linalg.norm(expected)<5e-5

@pytest.mark.parametrize("kind", ["soc", "psd"])
@pytest.mark.parametrize("eta", [np.nan,np.inf,-np.inf,-1.,0.,.3])
def test_invalid_step_refuses_and_preserves_previous_state(kind,eta):
    if kind=='soc':
        v=np.array([[1.,0.,0.]],'f'); h=np.array([[0.,1.,-.3]],'f'); fn=soc_step
    else:
        v=np.eye(3,dtype='f')[None];h=np.diag([1.,-.5,.2]).astype('f')[None];fn=psd_bounded_step
    out,valid,*_=jax.jit(fn)(v,h,np.float32(eta))
    if not np.isfinite(eta) or eta<0:
        assert not bool(valid);np.testing.assert_array_equal(out,v)
    else:
        assert bool(valid) and np.isfinite(out).all()
        if eta==0:np.testing.assert_array_equal(out,v)
