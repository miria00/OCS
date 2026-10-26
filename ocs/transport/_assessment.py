"""Same rounded-primal / quadratic-dual numerical gate, f64 GPU reductions.

Qualification compares every returned scalar to the blocked CPU implementation.
This is not interval arithmetic. No cached optimizer marginal is trusted.
OCS and QDROT use this identical checker from actual host return arrays.
"""
import numpy as np
import jax,jax.numpy as jnp

@jax.jit
def _evaluate(c,p,q,x,u,v,reg):
    c,p,q,x,u,v=(jnp.asarray(z,jnp.float64) for z in (c,p,q,x,u,v))
    reg=jnp.asarray(reg,jnp.float64)
    finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(z)) for z in (c,p,q,x,u,v,reg)]))
    admitted=finite&(reg>0)&(jnp.min(p)>=0)&(jnp.min(q)>=0)&(jnp.sum(p)==jnp.sum(q))
    r=jnp.sum(x,axis=1);cc=jnp.sum(x,axis=0);x2=jnp.sum(x*x)
    positive=jnp.maximum(x,0);positive_rows=jnp.sum(positive,axis=1)
    row_scale=jnp.minimum(1,jnp.where(positive_rows>0,p/jnp.where(positive_rows>0,positive_rows,1),1))
    scaled_cols=jnp.sum(positive*row_scale[:,None],axis=0)
    col_scale=jnp.minimum(1,jnp.where(scaled_cols>0,q/jnp.where(scaled_cols>0,scaled_cols,1),1))
    z=positive*row_scale[:,None]*col_scale[None,:]
    dp=p-jnp.sum(z,axis=1);dq=q-jnp.sum(z,axis=0)
    eps=jnp.finfo(jnp.float64).eps
    deficits_valid=(jnp.min(dp)>=-64*eps*jnp.maximum(jnp.max(p),jnp.max(q)))&(jnp.min(dq)>=-64*eps*jnp.maximum(jnp.max(p),jnp.max(q)))
    dp=jnp.maximum(dp,0);dq=jnp.maximum(dq,0);mass=jnp.sum(dp)
    z=z+jnp.where((mass>0)&(jnp.sum(dq)>0),dp[:,None]*dq[None,:]/jnp.where(mass>0,mass,1),0)
    target=jnp.sqrt(jnp.sum(p*p)+jnp.sum(q*q))
    marginal=jnp.sqrt(jnp.sum((r-p)**2)+jnp.sum((cc-q)**2))/target
    rounded=jnp.sqrt(jnp.sum((jnp.sum(z,axis=1)-p)**2)+jnp.sum((jnp.sum(z,axis=0)-q)**2))/target
    negative=jnp.where(x2>0,jnp.sqrt(jnp.sum(jnp.minimum(x,0)**2)/jnp.where(x2>0,x2,1)),0)
    raw=jnp.sum(c*x)+reg/2*jnp.sum(x*x)
    upper=jnp.sum(c*z)+reg/2*jnp.sum(z*z)
    lower=jnp.dot(p,u)+jnp.dot(q,v)-jnp.sum(jnp.maximum(u[:,None]+v[None,:]-c,0)**2)/(2*reg)
    width=upper-lower;scale=jnp.maximum(jnp.abs(upper),jnp.abs(lower))
    relative=jnp.where(scale>0,width/jnp.where(scale>0,scale,1),jnp.where(width==0,0,jnp.inf))
    arithmetic_valid=admitted&deficits_valid&(target>0)&(rounded<=1e-12)&(width>=-128*eps*scale)
    passed=arithmetic_valid&(marginal<=1e-3)&(negative<=1e-3)&(relative>=0)&(relative<=1e-3)
    return dict(gate_pass=passed,arithmetic_valid=arithmetic_valid,marginal_relative=marginal,
        nonneg_violation_relative=negative,minimum=jnp.min(x),raw_objective=raw,upper=upper,lower=lower,
        width=width,relative_width=relative,rounded_marginal_relative=rounded,
        rounding_displacement=jnp.sqrt(jnp.sum((z-x)**2)))

def assess(c,p,q,x,u,v,reg):
    # Scoped x64 only for assessment. All optimization arrays stay f32.
    with jax.enable_x64():
        out=jax.device_get(_evaluate(c,p,q,x,u,v,np.float64(reg)))
    result={k:bool(z) if k in ('gate_pass','arithmetic_valid') else float(z) for k,z in out.items()}
    result.update(numerical_float64=True,exact_arithmetic=False,backend='GPU f64 common assessor')
    if not result['arithmetic_valid']:raise ArithmeticError(str(result))
    return result
