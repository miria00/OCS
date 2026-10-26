"""Shared fixtures.

The only thing here is the pin that names which acceptance criterion a legacy
test file is making its assertions about.
"""
import pytest

import ocs.model.compile as _compile_module


@pytest.fixture
def residual_criterion(monkeypatch):
    """Compile with the RESIDUAL acceptance criterion for the marked file.

    OCS accepts on a CERTIFIED OBJECTIVE GAP by default: ``status == "solved"``
    means a strict lower bound on the optimal value was established at the
    returned point and the gap to the objective value there met ``tol``.  The
    relative prox-graph residual is still computed and still reported, but it
    no longer awards the status, because it is a property of the algorithm's
    splitting rather than of the point's objective accuracy.

    A file marked with this fixture is making assertions ABOUT THAT RESIDUAL,
    or about something measured through it:

      * the residual and relative-scaling machinery itself;
      * equivalence between code paths (dense against compact, batched against
        single, dual layouts, PDHG variants), where the claim is that two
        routes produce the SAME point and the same reported number;
      * the effect of an algorithm knob, such as the primal weight;
      * cone KKT acceptance, for families whose blocks declare no exact
        conjugate-domain contract and therefore cannot certify an objective
        bound at all.

    None of those are objective-accuracy claims, and none of their assertions
    are weakened by this fixture: it names the criterion they were written
    for.  The default criterion has its own tests, in
    ``tests/test_batch_result_contract.py``.

    Grep for ``residual_criterion`` to see exactly which files are pinned.
    """
    original = _compile_module.compile_family

    def pinned(family, **kwargs):
        kwargs.setdefault("accuracy", "residual")
        return original(family, **kwargs)

    monkeypatch.setattr(_compile_module, "compile_family", pinned)
    return pinned
