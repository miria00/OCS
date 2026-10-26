"""The optional hardware-aware planner above compile (ocs.experimental.plan).

Runs on any backend: the planner only queries the device, estimates the
family's footprint, probes candidate compile layouts with a two-block budget,
and caches the choice.  Run: python -m pytest tests/test_plan.py
"""
import os
import numpy as np
import pytest
import jax.numpy as jnp
import ocs
from ocs.experimental.plan import estimate_bytes, family_signature, plan_family, Plan
from ocs.experimental import compile_with_plan

# Pinned to the RESIDUAL acceptance criterion: this file asserts about
# the setup planner, not an acceptance claim.
# OCS accepts on a certified objective gap by DEFAULT; see the
# `residual_criterion` fixture in tests/conftest.py for why that is not
# what these assertions are about. No assertion here is weakened.
pytestmark = pytest.mark.usefixtures("residual_criterion")


def _lp_family(n=40, m=25, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, n)).astype(np.float32)
    x0 = np.abs(rng.normal(size=n)).astype(np.float32)
    b = A @ x0
    x = ocs.Variable(n)
    fam = ocs.minimize([ocs.nonneg(x), ocs.equality(ocs.Dense(A) @ x - ocs.Parameter("b"))],
                       cost=ocs.Parameter("c"))
    data = {"b": jnp.asarray(b), "c": jnp.asarray(rng.uniform(0.5, 1.5, n).astype(np.float32))}
    return fam, data


def test_estimate_is_positive_and_monotone():
    fam, _ = _lp_family()
    e_small = estimate_bytes(fam)
    fam2, _ = _lp_family(n=400, m=250)
    assert 0 < e_small < estimate_bytes(fam2)
    assert family_signature(fam, {}) != family_signature(fam2, {})


def test_plan_probes_caches_and_compiles(tmp_path):
    fam, data = _lp_family()
    p = plan_family(fam, data, mode="auto", cache_dir=str(tmp_path), tol=1e-4, max_blocks=50)
    assert isinstance(p, Plan)
    assert p.compile_kwargs["data_layout"] in ("dense", "compact")
    assert p.compile_kwargs["tol"] == 1e-4                      # fixed keywords survive
    assert any(r.get("seconds_per_block") for r in p.candidates)
    assert os.path.exists(os.path.join(str(tmp_path), p.key + ".json"))
    # the plan compiles and solves the family
    comp = fam.compile(**p.compile_kwargs)
    sol = comp.solve_instance(data)
    assert np.isfinite(sol.residual)
    # a cached plan is returned without probing (candidates are the saved ones)
    p2 = plan_family(fam, data, mode="cached", cache_dir=str(tmp_path), tol=1e-4, max_blocks=50)
    assert p2.key == p.key and p2.compile_kwargs == p.compile_kwargs


def test_plan_without_data_uses_the_estimate_only(tmp_path):
    fam, _ = _lp_family()
    p = plan_family(fam, None, mode="auto", cache_dir=str(tmp_path))
    assert "estimate only" in p.note
    assert p.fits_estimate


def test_invalid_mode(tmp_path):
    fam, data = _lp_family()
    with pytest.raises(ValueError):
        plan_family(fam, data, mode="sometimes", cache_dir=str(tmp_path))


def test_compile_with_plan_and_top_level_alias(tmp_path):
    fam, data = _lp_family()
    comp = compile_with_plan(fam, data, mode="auto", cache_dir=str(tmp_path), tol=1e-4, max_blocks=50)
    sol = comp.solve_instance(data)
    assert np.isfinite(sol.residual)
    assert ocs.plan is ocs.experimental.plan                 # the short name is the same object
