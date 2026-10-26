"""Experimental layers above the engine.  Everything here is optional and off
by default, and nothing here changes the solver: the engine, the operator
contracts, and the proximal library are untouched by this package.

Contents.

``plan``: the hardware-aware planner.  It reads the device the family will run
on and its memory, estimates the compiled family's footprint, probes the
candidate compile layouts with a two-block budget, and caches the choice.

    plan = ocs.experimental.plan(prob, data, mode="auto")   # or "cached"
    comp = prob.compile(**plan.compile_kwargs)
    comp = ocs.experimental.compile_with_plan(prob, data)   # the same in one call
    problem.solve(solver="OCS", plan="auto")                # the CVXPY entry

``prob.compile()`` without a plan keeps the static defaults.

Not in this package but in the same spirit: the restart-gated primal weight is
a runner outside the package, ``examples/experimental/gated_weight_runner.py``,
which drives the shipped solver at fixed weights through its public compile
keywords.  Its measurements are in ``profiling/gated_weight_20260910``.
"""
from .plan import Plan, PlanningError, device_info, estimate_bytes, family_signature, plan, plan_family


def compile_with_plan(family, data=None, *, mode="auto", **compile_kwargs):
    """Plan for this device, then compile: the one-call form of the planner.

    ``compile_kwargs`` are the keywords the caller has already decided
    (tolerance, method, weight); the plan fills the hardware-facing ones.
    """
    import functools
    import jax
    import numpy as np

    dev, _, _ = device_info(compile_kwargs.get("device"))
    chosen = plan_family(family, data, mode=mode, **compile_kwargs)
    with jax.default_device(dev):
        compiled = family.compile(**chosen.compile_kwargs)

    # Keep the public CompiledFamily type and signatures. Entry points pin both
    # subsequent tracing and caller-owned array placement to the planned device.
    # Passing compile_kwargs to family.compile manually chooses layouts only;
    # this wrapper is the explicit device-placement form of the API.
    def pin(method):
        @functools.wraps(method)
        def call(*args, **kwargs):
            def put(value):
                if isinstance(value, jax.core.Tracer):
                    return jax.lax.with_sharding_constraint(value, jax.sharding.SingleDeviceSharding(dev))
                return jax.device_put(value, dev) if isinstance(value, (jax.Array, np.ndarray)) else value
            with jax.default_device(dev):
                result = method(*jax.tree_util.tree_map(put, args),
                                **jax.tree_util.tree_map(put, kwargs))
                result = jax.tree_util.tree_map(
                    lambda v: jax.lax.with_sharding_constraint(v, jax.sharding.SingleDeviceSharding(dev))
                    if isinstance(v, jax.core.Tracer) else v, result)
            def arrays(value):
                if isinstance(value, jax.Array):
                    yield value
                elif hasattr(value, "x") and hasattr(value, "y"):
                    yield from arrays((value.x, value.y))
                elif isinstance(value, dict):
                    yield from arrays(tuple(value.values()))
                elif isinstance(value, (tuple, list)):
                    for item in value:
                        yield from arrays(item)
            for value in arrays(result):
                # Tracers have no concrete device set. device_put above still
                # carries placement into the traced computation; checking the
                # abstract result would break jit/vmap/grad at this boundary.
                if not isinstance(value, jax.core.Tracer) and value.devices() != {dev}:
                    raise ValueError("compiled plan returned an array on a different device")
            return result
        return call

    for name in dir(compiled):
        method = getattr(compiled, name)
        if not name.startswith("_") and callable(method):
            setattr(compiled, name, pin(method))
    compiled.plan = chosen
    return compiled


__all__ = ["Plan", "PlanningError", "compile_with_plan", "device_info", "estimate_bytes",
           "family_signature", "plan", "plan_family"]
