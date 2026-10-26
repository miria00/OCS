"""OCS first-order lowering for positive-quadratic marginal OT.

Integrated from the qualified warm OT prototype without recurrence changes.
No optimizing initializer or native QDROT call.
All arrays are float32. Potentials here use the minimization saddle sign y;
the common maximization-dual assessor receives -y. Standard and accelerated
recurrences are explicit, separate policies. Checks never reset the state.
"""
import functools
import jax
import jax.numpy as jnp
from jax import lax

def reference_step(state,cost,p,q,mu,accelerated):
    x,u,v,xb,tau,sigma,k=state
    un=u+sigma*(jnp.sum(xb,axis=1)-p)
    vn=v+sigma*(jnp.sum(xb,axis=0)-q)
    xn=jnp.maximum(x-tau*(cost+un[:,None]+vn[None,:]),0)/(1+tau*mu)
    theta=lax.rsqrt(1+2*mu*tau) if accelerated else jnp.float32(1)
    return xn,un,vn,xn+theta*(xn-x),tau*theta,sigma/theta,k+1

def tiled_update(x,cost,u,v,tau,mu,*,interpret=False):
    """One plan pass emits new X and both sets of tile partial sums.

    Each tile reads only its own X entries; aliasing its output is safe.
    Masking covers arbitrary rectangular tails. No atomics or hidden dual solve.
    """
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plt
    m,n=x.shape;br,bc=32,128;nr=(m+br-1)//br;nc=(n+bc-1)//bc
    def kernel(xr,cr,ur,vr,tr,mur,xo,ro,co):
        ir=pl.program_id(0);ic=pl.program_id(1)
        rows=ir*br+jnp.arange(br);cols=ic*bc+jnp.arange(bc)
        mask=(rows[:,None]<m)&(cols[None,:]<n)
        old=plt.load(xr.at[rows[:,None],cols[None,:]],mask=mask,other=0)
        c=plt.load(cr.at[rows[:,None],cols[None,:]],mask=mask,other=0)
        uu=plt.load(ur.at[rows],mask=rows<m,other=0)
        vv=plt.load(vr.at[cols],mask=cols<n,other=0)
        t=tr[()];mu0=mur[()]
        new=jnp.where(mask,jnp.maximum(old-t*(c+uu[:,None]+vv[None,:]),0)/(1+t*mu0),0)
        plt.store(xo.at[rows[:,None],cols[None,:]],new,mask=mask)
        ro[ir,ic,jnp.arange(br)]=jnp.sum(new,axis=1)
        co[ir,ic,jnp.arange(bc)]=jnp.sum(new,axis=0)
    xn,rp,cp=pl.pallas_call(kernel,
        out_shape=(jax.ShapeDtypeStruct((m,n),jnp.float32),
            jax.ShapeDtypeStruct((nr,nc,br),jnp.float32),jax.ShapeDtypeStruct((nr,nc,bc),jnp.float32)),
        grid=(nr,nc),input_output_aliases={0:0},interpret=interpret,
        compiler_params=plt.CompilerParams(num_warps=4))(x,cost,u,v,tau,mu)
    r=jnp.sum(rp,axis=1).reshape(-1)[:m]
    c=jnp.sum(cp,axis=0).reshape(-1)[:n]
    return xn,r,c

def cached_step(state,cost,p,q,mu,accelerated,execution='xla'):
    x,u,v,r,c,rb,cb,tau,sigma,k=state
    un=u+sigma*(rb-p);vn=v+sigma*(cb-q)
    if execution in ('pallas','interpret'):
        xn,rn,cn=tiled_update(x,cost,un,vn,tau,mu,interpret=execution=='interpret')
    else:
        xn=jnp.maximum(x-tau*(cost+un[:,None]+vn[None,:]),0)/(1+tau*mu)
        rn=jnp.sum(xn,axis=1);cn=jnp.sum(xn,axis=0)
    theta=lax.rsqrt(1+2*mu*tau) if accelerated else jnp.float32(1)
    return xn,un,vn,rn,cn,rn+theta*(rn-r),cn+theta*(cn-c),tau*theta,sigma/theta,k+1

def initial_state(m,n,mu,*,reference=False):
    tau=jnp.float32(1)/jnp.float32(mu);sigma=jnp.float32(.99)*jnp.float32(mu)/jnp.float32(m+n)
    x=jnp.zeros((m,n),jnp.float32);u=jnp.zeros(m,jnp.float32);v=jnp.zeros(n,jnp.float32)
    if reference:return x,u,v,jnp.zeros((m,n),jnp.float32),tau,sigma,jnp.int32(0)
    return (x,u,v,jnp.zeros(m,jnp.float32),jnp.zeros(n,jnp.float32),
        jnp.zeros(m,jnp.float32),jnp.zeros(n,jnp.float32),tau,sigma,jnp.int32(0))

@jax.jit
def centered_initial_state(cost,p,q,mu):
    """Algebraic equality-constrained quadratic dual, before nonnegativity.

    One cost marginal action, no iteration/search/optimizer. The returned primal
    start remains zero. Potentials are derived from original inputs only.
    """
    m,n=cost.shape
    row=jnp.sum(cost,axis=1);col=jnp.sum(cost,axis=0)
    offset=(jnp.sum(row)+mu*jnp.sum(p))/(2*jnp.float32(m)*jnp.float32(n))
    up=(row+mu*p)/n-offset;vp=(col+mu*q)/m-offset
    state=initial_state(m,n,mu)
    return (state[0],-up,-vp,*state[3:])

def make_segment(accelerated=True,execution='xla',donate=True):
    """Dynamic cap: one executable reused without restart or reinitialization."""
    def segment(state,cost,p,q,mu,steps):
        def body(_,s):
            if execution=='reference':return reference_step(s,cost,p,q,mu,accelerated)
            return cached_step(s,cost,p,q,mu,accelerated,execution)
        return lax.fori_loop(0,steps,body,state)
    return jax.jit(segment,donate_argnums=(0,) if donate else ())

@jax.jit
def screening(state,cost,p,q,mu):
    """Cheap f32 observation only. Cannot award original accuracy."""
    x,u,v=state[:3]
    r=jnp.sum(x,axis=1)-p;c=jnp.sum(x,axis=0)-q
    marginal=jnp.sqrt((jnp.sum(r*r)+jnp.sum(c*c))/(jnp.sum(p*p)+jnp.sum(q*q)))
    primal=jnp.sum(cost*x)+mu/2*jnp.sum(x*x)
    vpos=jnp.maximum(-u[:,None]-v[None,:]-cost,0)
    dual=-jnp.dot(p,u)-jnp.dot(q,v)-jnp.sum(vpos*vpos)/(2*mu)
    rawgap=(primal-dual)/jnp.maximum(jnp.abs(primal),jnp.abs(dual))
    return marginal,rawgap,primal,dual
