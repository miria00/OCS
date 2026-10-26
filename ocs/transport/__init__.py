"""Prepared quadratic transport on NVIDIA GPUs.

This opt-in solver has its own numerical feasibility/objective-bracket contract;
it does not change ``ocs.CompiledFamily`` or its stopping policy. CUDA imports
are deferred until construction. Install the ``transport`` extra first.
"""
from ._prepared import PreparedQuadraticOT, QuadraticOTResult, OTConvergenceError

__all__ = ["PreparedQuadraticOT", "QuadraticOTResult", "OTConvergenceError"]
