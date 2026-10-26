"""The import package and the distribution report one version.

PyPA's guidance: the distribution name (``ocsolver``) and the import name
(``ocs``) may differ, but their reported versions must agree.  This runs
against an INSTALLED package (a wheel, or an editable install from the
checkout); on a bare source tree without metadata it fails rather than skips,
because the release suite is run on installed bytes.
"""
from importlib.metadata import version

import ocs
from ocs.model.compile import ACCEPTANCE_POLICY_ID

RELEASE = "0.3.7"


def test_import_version_equals_distribution_version():
    assert ocs.__version__ == version("ocsolver") == RELEASE


def test_release_candidate_carries_the_named_acceptance_policy():
    assert ACCEPTANCE_POLICY_ID == "returned-point-three-condition-scaleinv-v2"
