"""The pinned optional backend recognizes real 5090 metadata without guessing."""
from types import SimpleNamespace
import subprocess
import sys

import pytest

from ocs.transport._pallas_compat import ensure_pallas_device


KIND = "NVIDIA GeForce RTX 5090"


def device(**changes):
    values = dict(device_kind=KIND, platform="gpu", compute_capability="12.0")
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture
def registry(monkeypatch):
    from jax._src.pallas.triton import gpu_info
    monkeypatch.setattr(gpu_info, "registry", {})
    return gpu_info


def test_missing_5090_metadata_and_idempotence(registry):
    assert ensure_pallas_device(device())
    callback = registry.registry[KIND]
    value = callback()
    assert value.gpu_version is None
    assert value.arch_name == "12.0" and value.compute_capability == 120
    assert not ensure_pallas_device(device())
    assert registry.registry[KIND] is callback


def test_preserves_existing_registration(registry):
    sentinel = lambda: None
    registry.registry[KIND] = sentinel
    assert not ensure_pallas_device(device())
    assert registry.registry[KIND] is sentinel


def test_prefers_builtin_recognition(registry, monkeypatch):
    monkeypatch.setattr(registry, "gpu_version_from_device_kind", lambda kind: object())
    assert not ensure_pallas_device(device())
    assert registry.registry == {}


@pytest.mark.parametrize("capability", [None, "8.9", "12.1", 120])
def test_refuses_mismatched_architecture(registry, capability):
    with pytest.raises(RuntimeError, match="compute capability 12.0"):
        ensure_pallas_device(device(compute_capability=capability))
    assert registry.registry == {}


def test_does_not_infer_unknown_device(registry):
    assert not ensure_pallas_device(device(device_kind="unqualified GPU"))
    assert registry.registry == {}


def test_refuses_other_jax_version(registry, monkeypatch):
    import jax
    monkeypatch.setattr(jax, "__version__", "different")
    with pytest.raises(RuntimeError, match="qualified JAX 0.11.1"):
        ensure_pallas_device(device())
    assert registry.registry == {}


def test_transport_import_keeps_device_compatibility_lazy():
    subprocess.run([sys.executable, "-I", "-c", """
import jax
def refuse():
    raise AssertionError('import queried a device')
jax.devices = refuse
import sys
import ocs.transport
assert 'cupy' not in sys.modules
assert 'ocs.transport._pallas_compat' not in sys.modules
"""], check=True)
