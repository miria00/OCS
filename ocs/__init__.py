"""OCS — an operator-native convex solver for large-scale optimization on GPUs.

Solve  min_x  sum_i f_i(K_i x + l_i)  where each K_i is given ONLY by its
forward/adjoint products and each f_i by a GPU-native proximal block.  The matrix
is never formed.  Compile a family once, estimate scaling once, solve many
instances with changing data on the GPU in float32.  A result is ``solved``
only when the returned point passes the acceptance test of
``ocs.model.compile.ACCEPTANCE_POLICY_ID`` at the requested tolerance.
"""
from .experimental.plan import plan, plan_family, Plan   # ocs.plan is the experimental planner
from .operators import LinearOperator, Identity, Diagonal, Dense, Weighted, Compose, OpSum, Scale, FFTMask, MarginalOperator, Gradient2D, LeftMatMul, BinPack, Gradient3D, Sylvester, Kronecker, MultiCommodityFlow, CommoditySum, DCTOperator, HaarWavelet, GaussianBlur, InverseLaplacian
from .prox import (
    ProxBlock, SumSquares, L1, L2, GroupL2, NonNeg, Box, Zero, Equality,
    Pinball, Logistic, Quadratic, PiecewiseLinearConvex,
    SecondOrderCone, PSDCone, ExpCone, PowerCone,
    SCALAR, DIAGONAL, MATRIX,
)
from .prox.block_metric import BlockMetric, BLOCK, restrict_condition
from .prox.cone_metric import ConeAutomorphism, nt_scaling_point, MAX_AUTOMORPHISM_COND
from .model import (
    Variable, Parameter, AffineExpr,
    minimize, least_squares, sum_squares, l1, l2, nonneg, box, equality, group_l2,
    pinball, quantile, logistic, pwl, soc, psd, exp_cone, pow_cone,
    CompiledFamily, Solution,
    OBJECTIVE_BOUND_STATUS, OBJECTIVE_BOUND_EPISTEMIC_LIMIT,
)

__version__ = "0.3.8"

__all__ = [
    # operators
    "LinearOperator", "Identity", "Diagonal", "Dense", "Weighted", "Compose", "OpSum", "Scale", "FFTMask", "MarginalOperator", "Gradient2D", "LeftMatMul", "BinPack", "Gradient3D", "Sylvester", "Kronecker", "MultiCommodityFlow", "CommoditySum", "DCTOperator", "HaarWavelet", "GaussianBlur", "InverseLaplacian",
    # prox
    "ProxBlock", "SumSquares", "L1", "L2", "GroupL2", "NonNeg", "Box", "Zero", "Equality",
    "Pinball", "Logistic", "Quadratic", "PiecewiseLinearConvex",
    "SecondOrderCone", "PSDCone", "ExpCone", "PowerCone",
    "SCALAR", "DIAGONAL", "MATRIX", "BlockMetric", "BLOCK", "restrict_condition",
    "ConeAutomorphism", "nt_scaling_point", "MAX_AUTOMORPHISM_COND",
    # modeling
    "Variable", "Parameter", "AffineExpr",
    "minimize", "least_squares", "sum_squares", "l1", "l2", "nonneg", "box",
    "equality", "group_l2", "pinball", "quantile", "logistic", "pwl",
    "soc", "psd", "exp_cone", "pow_cone",
    "CompiledFamily", "Solution",
    "OBJECTIVE_BOUND_STATUS", "OBJECTIVE_BOUND_EPISTEMIC_LIMIT",
]
