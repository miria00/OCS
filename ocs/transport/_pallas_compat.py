"""Lazy device metadata for the pinned JAX 0.11.1 transport backend.

JAX recognizes SM 12.0 but omits the RTX 5090 marketing name from its
Pallas/Triton table. Use its unknown-device registry only for this observed
name/architecture pair. This changes no kernel, optimizer or acceptance rule.
"""


def ensure_pallas_device(device):
    """Register the known missing device, preserving upstream/user entries.

    Called only during GPU transport construction, before compilation or
    workspace allocation. Other device kinds retain JAX's own behavior.
    """
    kind = device.device_kind
    if kind != "NVIDIA GeForce RTX 5090":
        return False
    if device.platform != "gpu" or getattr(device, "compute_capability", None) != "12.0":
        raise RuntimeError("RTX 5090 transport requires reported CUDA compute capability 12.0")

    import jax
    if jax.__version__ != "0.11.1":
        raise RuntimeError("RTX 5090 transport compatibility requires qualified JAX 0.11.1")
    # Private metadata interface, deliberately bounded to the pinned version.
    from jax._src.pallas.triton import gpu_info
    if gpu_info.gpu_version_from_device_kind(kind) is not None or kind in gpu_info.registry:
        return False

    def metadata():
        return gpu_info.GpuInfo(gpu_version=None, arch_name="12.0", compute_capability=120)

    # Preserve an entry installed by another constructor between the checks.
    return gpu_info.registry.setdefault(kind, metadata) is metadata
