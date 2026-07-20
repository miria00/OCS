"""OCS — an operator-native convex solver for large-scale optimization on GPUs.

Solve  min_x  g(x) + sum_i f_i(K_i x + l_i)  where each K_i is given ONLY by its
forward/adjoint products and each f_i by a GPU-native proximal block.  The matrix
is never formed.  Compile a family once, estimate scaling once, solve many
instances with changing data on the GPU (float32, to a 1e-3 relative residual).
"""
from .operators import LinearOperator, Identity, Diagonal, Dense, Weighted, FFTMask, MarginalOperator, Gradient2D, LeftMatMul, BinPack, Gradient3D, Sylvester, Kronecker, MultiCommodityFlow, CommoditySum, DCTOperator, HaarWavelet, GaussianBlur, InverseLaplacian
from .prox import (
    ProxBlock, SumSquares, L1, L2, GroupL2, NonNeg, Box, Zero, Equality,
    Pinball, Logistic, Quadratic, PiecewiseLinearConvex,
    SecondOrderCone, PSDCone, ExpCone, PowerCone,
    SCALAR, DIAGONAL, MATRIX,
)
from .model import (
    Variable, Parameter, AffineExpr,
    minimize, least_squares, sum_squares, l1, l2, nonneg, box, equality, group_l2,
    pinball, quantile, logistic, pwl, soc, psd, exp_cone, pow_cone,
    CompiledFamily, Solution,
)

__version__ = "0.1.0"

__all__ = [
    # operators
    "LinearOperator", "Identity", "Diagonal", "Dense", "Weighted", "FFTMask", "MarginalOperator", "Gradient2D", "LeftMatMul", "BinPack", "Gradient3D", "Sylvester", "Kronecker", "MultiCommodityFlow", "CommoditySum", "DCTOperator", "HaarWavelet", "GaussianBlur", "InverseLaplacian",
    # prox
    "ProxBlock", "SumSquares", "L1", "L2", "GroupL2", "NonNeg", "Box", "Zero", "Equality",
    "Pinball", "Logistic", "Quadratic", "PiecewiseLinearConvex",
    "SecondOrderCone", "PSDCone", "ExpCone", "PowerCone",
    "SCALAR", "DIAGONAL", "MATRIX",
    # modeling
    "Variable", "Parameter", "AffineExpr",
    "minimize", "least_squares", "sum_squares", "l1", "l2", "nonneg", "box",
    "equality", "group_l2", "pinball", "quantile", "logistic", "pwl",
    "soc", "psd", "exp_cone", "pow_cone",
    "CompiledFamily", "Solution",
]
