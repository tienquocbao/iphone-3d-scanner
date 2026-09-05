"""Backend-neutral canonical object observations for classical surface methods.

The implementation remains in :mod:`nksr_input` for now because it already
contains the tested, RGB-D based extraction path.  Despite that historical
module name it imports neither NKSR nor PyTorch.  This facade gives native
surface backends a truthful dependency boundary while preserving the C2 API.
"""

from nksr_input import (  # noqa: F401 - deliberate compatibility facade
    NKSRInput as SurfaceInput,
    NKSRInputConfig as SurfaceInputConfig,
    joint_voxel_aggregate,
    prepare_nksr_input as prepare_surface_input,
)

__all__ = [
    "SurfaceInput",
    "SurfaceInputConfig",
    "joint_voxel_aggregate",
    "prepare_surface_input",
]
