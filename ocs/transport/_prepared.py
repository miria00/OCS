"""Owned prepared implementation of strongly convex Chambolle–Pock for OT."""
from dataclasses import dataclass
import threading
import time
from types import MappingProxyType
from typing import Mapping

import numpy as np


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class QuadraticOTResult:
    """Independent host arrays and a numerical (not interval) accuracy report.

    ``plan`` is the raw float32 solution. The reported upper objective uses
    its feasible rounding; the lower uses the returned maximizing potentials.
    Arrays are initially read-only and remain valid after session close.
    """
    plan: np.ndarray
    row_potential: np.ndarray
    column_potential: np.ndarray
    report: Mapping
    statistics: Mapping

    @property
    def solved(self):
        return bool(self.report["gate_pass"])


class OTConvergenceError(RuntimeError):
    """The finite iteration budget ended without complete acceptance."""
    def __init__(self, iterations, report):
        self.iterations = iterations
        self.report = _freeze(report)
        super().__init__(f"Quadratic OT did not meet its accuracy contract in {iterations} updates")


class PreparedQuadraticOT:
    """Prepare repeated square, dense quadratic OT solves on one NVIDIA GPU.

    Minimize ``sum(C*X) + regularization/2 * sum(X**2)`` subject to
    ``X >= 0, X.sum(1) = p, X.sum(0) = q``. Inputs must be C-contiguous
    float32 NumPy arrays, with finite nonnegative entries and positive,
    exactly equal represented marginal mass. The complete numerical float64
    check requires relative marginal, nonnegativity and objective-bracket
    errors <= 1e-3. Optimizer arithmetic stays float32.

    Construction pays compilation, workspace and 1..3 distinct pinned output
    slots. Each solve starts fresh and consumes one slot, including a failed
    attempt after admission. Outputs are never reused. A session is serial;
    callers must not mutate input arrays during solve. Dense O(n**2) storage
    is required. This is a specialized path, not automatic cone equilibration.

    Set ``batch_size=B`` to prepare vectorized batches and use
    ``solve_many_results({'cost': costs, 'p': rows, 'q': columns})``.
    Each output slot then holds B results. Calls on a session remain serial;
    the B independent problems execute together on the GPU. See that method
    for shared-endpoint stopping and per-lane iteration-limit results.
    """
    def __init__(self, n, *, regularization, return_capacity=3, max_iterations=20000,
                 batch_size=None):
        if type(n) is not int or n < 2 or n*n > 2**31-1025:
            raise ValueError("n must be an integer >= 2 within the supported indexing range")
        if type(return_capacity) is not int or not 1 <= return_capacity <= 3:
            raise ValueError("return_capacity must be an integer in 1..3")
        if type(max_iterations) is not int or not 1 <= max_iterations <= 2**31-101:
            raise ValueError("max_iterations must be a positive bounded integer")
        if batch_size is not None and (type(batch_size) is not int or batch_size < 1
                                      or batch_size*n*n > 2**31-1025):
            raise ValueError("batch_size must be a positive integer within the supported indexing range")
        if isinstance(regularization, (bool, np.bool_)) or not np.isscalar(regularization):
            raise ValueError("regularization must be a finite positive float32 scalar")
        try:
            reg = float(regularization)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid regularization") from exc
        if not np.isfinite(reg) or not float(np.finfo(np.float32).tiny) <= reg <= float(np.finfo(np.float32).max):
            raise ValueError("regularization must be positive and normal in float32")
        if not all(float(np.finfo(np.float32).tiny) <= x <= float(np.finfo(np.float32).max)
                   for x in (1/reg, .99*reg/(2*n))):
            raise ValueError("regularization produces unrepresentable initial steps")
        import jax
        import jax.numpy as jnp
        if jax.__version__ != "0.11.1":
            raise RuntimeError("PreparedQuadraticOT requires the qualified JAX 0.11.1 transport extra")
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError("Install ocsolver[transport] and a compatible CUDA GPU JAX environment") from exc
        if jax.devices()[0].platform != "gpu":
            raise RuntimeError("PreparedQuadraticOT requires an NVIDIA GPU")
        from ._pallas_compat import ensure_pallas_device
        ensure_pallas_device(jax.devices()[0])
        from ._first_order import centered_initial_state, make_segment, screening
        from ._assessment import assess
        self._jax, self._jnp, self._cp = jax, jnp, cp
        self._assess, self._screen = assess, screening
        self._n, self._regularization = n, np.float32(reg)
        self._batch_size = batch_size
        self._max_iterations, self._return_capacity = max_iterations, return_capacity
        self._closed, self._used = False, 0
        self._lock = threading.Lock()
        self._outputs = []
        self._state = None
        self._dc = self._dp = self._dq = None
        if batch_size is not None:
            from ._batch import prepare_batch
            prepare_batch(self)
            return
        def pinned(shape):
            memory = cp.cuda.alloc_pinned_memory(int(np.prod(shape))*4)
            return np.frombuffer(memory, dtype=np.float32, count=int(np.prod(shape))).reshape(shape)
        self._hc, self._hp, self._hq = pinned((n,n)), pinned((n,)), pinned((n,))
        self._hu, self._hv = pinned((n,)), pinned((n,))
        self._outputs = [(pinned((n,n)), pinned((n,)), pinned((n,))) for _ in range(return_capacity)]
        self._hc.fill(0); self._hp.fill(np.float32(1/n)); self._hq[:] = self._hp
        self._dc = cp.empty((n,n), np.float32); self._dc.set(self._hc)
        self._dp, self._dq = cp.asarray(self._hp), cp.asarray(self._hq)
        cp.cuda.get_current_stream().synchronize()
        self._segment = make_segment(True, "pallas", True)
        self._reset = jax.jit(lambda old,c,p,q,mu: centered_initial_state(c,p,q,mu),
                              donate_argnums=(0,), keep_unused=True)
        @jax.jit
        def validate(c,p,q):
            return (jnp.all(jnp.isfinite(c)) & (jnp.min(c)>=0) &
                    jnp.all(jnp.isfinite(p)) & jnp.all(jnp.isfinite(q)) &
                    (jnp.min(p)>=0) & (jnp.min(q)>=0) &
                    (jnp.sum(p.astype(jnp.float64))>0) &
                    (jnp.sum(p.astype(jnp.float64))==jnp.sum(q.astype(jnp.float64))))
        self._validate = validate
        c,p,q = (jax.dlpack.from_dlpack(z, copy=False) for z in (self._dc,self._dp,self._dq))
        with jax.enable_x64():
            assert bool(validate(c,p,q))
        self._state = centered_initial_state(c,p,q,self.regularization)
        self._state = self._segment(self._state,c,p,q,self.regularization,jnp.int32(100))
        jax.block_until_ready(self._state)
        jax.block_until_ready(screening(self._state,c,p,q,self.regularization))
        assess(c,p,q,self._state[0],-self._state[1],-self._state[2],self.regularization)
        self._state = self._reset(self._state,c,p,q,self.regularization)
        jax.block_until_ready(self._state)

    @property
    def n(self): return self._n

    @property
    def regularization(self): return self._regularization

    @property
    def max_iterations(self): return self._max_iterations

    @property
    def return_capacity(self): return self._return_capacity

    @property
    def batch_size(self): return self._batch_size

    def solve_many_results(self, data):
        """Return one :class:`QuadraticOTResult` per prepared batch lane.

        Prepare with ``batch_size=B``. ``data`` maps ``cost``, ``p``, ``q``
        to stacked arrays or sequences of B arrays satisfying ``solve``'s
        input contract. Vectorized lockstep updates stop when every current
        pair passes, or at the cap. At the cap, inspect each result's ``solved``
        property; a passing row never awards success to another. Results bind
        the final common endpoint, not earlier individually stopping points.
        One admitted batch consumes one distinct prepared output slot.
        """
        if self.batch_size is None:
            raise ValueError("prepare batch_size to use solve_many_results")
        from ._batch import solve_many_results
        return solve_many_results(self, data)

    def solve(self, cost, p, q):
        if self.batch_size is not None:
            raise ValueError("use solve_many_results for a prepared batch")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("session is busy")
        start = time.perf_counter()
        try:
            if self._closed:
                raise RuntimeError("session is closed")
            n = self.n
            for z,shape in ((cost,(n,n)),(p,(n,)),(q,(n,))):
                if not isinstance(z,np.ndarray) or z.shape!=shape or z.dtype!=np.float32 or not z.flags.c_contiguous:
                    raise ValueError("inputs must be C-contiguous float32 NumPy arrays of the prepared shape")
            if self._used >= self.return_capacity:
                raise RuntimeError("prepared output capacity exhausted")
            slot = self._used; self._used += 1
            x,u,v = self._outputs[slot]
            cp,jax,jnp = self._cp,self._jax,self._jnp
            t=time.perf_counter()
            np.copyto(self._hc,cost); np.copyto(self._hp,p); np.copyto(self._hq,q)
            record=dict(host_staging_s=time.perf_counter()-t, events=[])
            t=time.perf_counter()
            self._dc.set(self._hc); self._dp.set(self._hp); self._dq.set(self._hq)
            cp.cuda.get_current_stream().synchronize()
            cd,pd,qd=(jax.dlpack.from_dlpack(z,copy=False) for z in (self._dc,self._dp,self._dq))
            record["input_transfer_s"]=time.perf_counter()-t
            t=time.perf_counter()
            with jax.enable_x64():
                if not bool(self._validate(cd,pd,qd)):
                    raise ValueError("inputs must be finite, nonnegative, with positive exactly equal marginal mass")
            record["input_validation_s"]=time.perf_counter()-t
            t=time.perf_counter()
            self._state=self._reset(self._state,cd,pd,qd,self.regularization)
            jax.block_until_ready(self._state)
            record["fresh_initialize_s"]=time.perf_counter()-t
            target=0; gate={}
            while target < self.max_iterations:
                steps=min(100,self.max_iterations-target); target+=steps
                t=time.perf_counter()
                self._state=self._segment(self._state,cd,pd,qd,self.regularization,jnp.int32(steps))
                jax.block_until_ready(self._state)
                event=dict(iterations=target, update_s=time.perf_counter()-t)
                t=time.perf_counter(); screen=np.asarray(self._screen(self._state,cd,pd,qd,self.regularization))
                event.update(screen_s=time.perf_counter()-t,marginal=float(screen[0]),raw_gap=float(screen[1]))
                if not np.isfinite(screen).all():
                    raise ArithmeticError("nonfinite quadratic OT screen")
                if (screen[0]<=1e-3 and abs(screen[1])<=1e-3) or target==self.max_iterations:
                    t=time.perf_counter()
                    gate=self._assess(cd,pd,qd,self._state[0],-self._state[1],-self._state[2],self.regularization)
                    event.update(gate=gate,common_check_s=time.perf_counter()-t)
                record["events"].append(event)
                if event.get("gate",{}).get("gate_pass"):
                    break
            if not gate.get("gate_pass"):
                raise OTConvergenceError(target,gate)
            t=time.perf_counter()
            cp.from_dlpack(self._state[0]).get(out=x,blocking=True)
            cp.from_dlpack(self._state[1]).get(out=self._hu,blocking=True)
            cp.from_dlpack(self._state[2]).get(out=self._hv,blocking=True)
            record["device_return_s"]=time.perf_counter()-t
            t=time.perf_counter(); np.negative(self._hu,out=u); np.negative(self._hv,out=v)
            for value in (x,u,v): value.setflags(write=False)
            record.update(return_finalize_s=time.perf_counter()-t,output_slot=slot,
                          iterations=target,prepared_output_capacity=self.return_capacity,
                          prepared_output_payload_bytes=sum(z.nbytes for out in self._outputs for z in out),
                          independent_host_copy_s=0.0)
            report=_freeze(gate); statistics=_freeze(record)
            # Include construction of the immutable public result in the call timer.
            result=QuadraticOTResult(x,u,v,report,statistics)
            record["complete_warm_s"]=time.perf_counter()-start
            return QuadraticOTResult(result.plan,result.row_potential,result.column_potential,
                                     result.report,_freeze(record))
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("session is busy")
        try:
            if not self._closed:
                self._state=None; self._outputs=[]
                self._dc=self._dp=self._dq=None
                self._hc=self._hp=self._hq=self._hu=self._hv=None
                self._closed=True
        finally:
            self._lock.release()

    def __enter__(self):
        if self._closed: raise RuntimeError("session is closed")
        return self

    def __exit__(self, *exc):
        self.close()

    def __copy__(self):
        raise TypeError("PreparedQuadraticOT owns device state and cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("PreparedQuadraticOT owns device state and cannot be copied")

    def __reduce_ex__(self, protocol):
        raise TypeError("PreparedQuadraticOT cannot be serialized; prepare a new session")
