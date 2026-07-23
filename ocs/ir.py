"""The internal representation (IR) of an OCS problem family.

Everything downstream: equilibration, the PDHG engine, stopping, reads only
this IR, never the surface modeling syntax.  A problem family is

    minimize_x  g(x) + sum_i f_i(K_i x + l_i),

where each K_i is a LinearOperator (matvec/adjoint only), each f_i is a ProxBlock
from the prox library, l_i is a per-instance offset (a function of the instance
data), and g is an optional primal block (a term whose operator is the identity).

The matrix is never formed.  Per-instance data enters only through the offsets
l_i (and, later, through parametric operators).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .operators.base import LinearOperator
from .prox.base import ProxBlock


@dataclass
class Term:
    """One term f(K x + l).

    op        : the linear operator K (shape (m, n))
    block     : the prox block f
    offset_fn : maps instance data -> l  (an (m,) array); None means l = 0
    """
    op: LinearOperator
    block: ProxBlock
    offset_fn: Optional[Callable[[dict], "object"]] = None

    @property
    def out_dim(self) -> int:
        return self.op.shape[0]


@dataclass
class ProblemFamily:
    """A symbolic, static description of a family of problems.

    n        : number of variables (dimension of x)
    terms    : the list of Term objects
    cost_fn  : maps instance data -> c (an (n,) linear-cost vector); None => 0.
               This is what lets the cost c change across instances (the
               "same K, many b's and c's" regime), alongside the offsets l_i.
    """
    n: int
    terms: list = field(default_factory=list)
    cost_fn: object = None

    def compile(self, **kwargs):
        # imported here to avoid a circular import at module load
        from .model.compile import compile_family
        return compile_family(self, **kwargs)
