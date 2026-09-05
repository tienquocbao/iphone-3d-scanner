# Optional NKSR backend

Gate C2 keeps TSDF as the deterministic reference and invokes NKSR only through
a subprocess. The FastAPI receiver and the `iphone3d` environment do not import
NKSR, PyTorch, or its CUDA extension. Set `IPHONE3D_NKSR_PYTHON` to the Python
executable of a separately managed NKSR environment, then restart the receiver.
Do not put that machine-specific path in source control.

## Upstream version and API

The adapter targets official [nv-tlabs/NKSR](https://github.com/nv-tlabs/NKSR)
commit `e40336845e67761343a756788e5a98b827d4a143`, package version `1.0.3`.
It follows [NKSR-USAGE.md](https://github.com/nv-tlabs/NKSR/blob/public/NKSR-USAGE.md):
aligned `xyz` and per-point `sensor` arrays are passed to
`Reconstructor.reconstruct`. Full mode passes `voxel_size` instead of
`detail_level`; chunk mode passes neither and rescales metric coordinates by
`0.1 / voxel_size` as required by the upstream documentation.

The default `ks` (kitchen-sink) checkpoint is loaded by `Reconstructor` and uses
the upstream/PyTorch cache. Keep that cache persistent; model weights are not
stored in this repository and are not downloaded by the receiver itself.

## Native Windows investigation

Native Windows NKSR is not currently available on this workstation. The
official environment specifies Python 3.10 and CUDA 12.8/NVCC. The package
builds a custom CUDA extension and its setup reports x86-64 Linux support. This
host currently has only the NVIDIA driver (RTX 2050, 4 GiB); Python 3.10, NVCC,
MSVC `cl`, Ninja, CMake, and an executable Conda frontend were not discoverable.
A non-installing `pip --dry-run --no-build-isolation` against the pinned commit
also stopped at metadata generation because PyTorch is intentionally absent
from `iphone3d`.

Do not install or downgrade packages in `iphone3d` to work around this. If a
Linux/WSL NKSR environment is prepared later, the existing subprocess contract
can be bridged to it without changing scan/session data. WSL integration is not
implemented in Gate C2.

## Runtime policy

`auto` selects CPU when CUDA is unavailable. On CUDA, GPUs at or below the
configured low-memory threshold (default 6 GiB) use chunk mode. Larger GPUs use
full mode for bounded inputs, with one chunk retry only for a recognized CUDA
out-of-memory error. Optional CPU fallback after CUDA OOM is disabled by
default because it may be very slow.

The RTX 2050 has not been benchmarked because no compatible NKSR runtime is
installed. These defaults are conservative safety policy, not performance
claims, and remain configurable.

## Licensing

NKSR source is under the NVIDIA Source Code License for NKSR, including the
non-commercial research/evaluation use limitation in section 3.3. The README
states that the kitchen-sink model is released under CC-BY-SA 4.0. Review both
upstream licenses before redistribution or commercial use. No upstream source,
license text, or model weights are copied into this project.
