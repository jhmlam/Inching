"""Import smoke test.

The top-level package and the modules below do not touch a GPU at import time,
so this runs on a CPU-only machine (e.g. CI). Solver modules under
InchingLiteInteger.Burn seed CuPy's RNG at import and therefore need a CUDA
device; they are tested only when one is present.
"""
import importlib

import pytest

CPU_SAFE = [
    "InchingLiteInteger",
    "InchingLiteInteger.util",
    "InchingLiteInteger.Fuel.Coordinate.T1",
    "InchingLiteInteger.Fuel.Coordinate.T2",
    "InchingLiteInteger.Burn.Coordinate.T3",
    "InchingLiteInteger.Burn.Visualisation.T1",
    "InchingLiteInteger.Burn.PolynomialFilters.T0",
]

# Modules the notebooks import; each one seeds cupy.random, allocates a cupy
# array, or evaluates torch.device(0) in a default argument at import time.
GPU_ONLY = [
    "InchingLiteInteger.Fuel.T1",
    "InchingLiteInteger.Burn.Coordinate.T1",
    "InchingLiteInteger.Burn.Coordinate.T3",
    "InchingLiteInteger.Burn.Visualisation.T1",
    "InchingLiteInteger.Burn.Visualisation.T2",
    "InchingLiteInteger.Burn.HermitianLanczos.T2",
    "InchingLiteInteger.Burn.PolynomialFilters.T2",
    "InchingLiteInteger.Burn.JacobiDavidsonHotellingDeflation.T1",
    "InchingLiteInteger.Burn.ThickRestartLanczosHotellingDeflation.T1",
    "InchingLiteInteger.Burn.ChebyshevDavidsonSubspaceIteration.T1",
    "InchingLiteInteger.Burn.ImplicitlyRestartedLanczosHotellingDeflation.T1",
]


def _has_cuda_device():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.parametrize("name", CPU_SAFE)
def test_import_cpu_safe(name):
    importlib.import_module(name)


@pytest.mark.skipif(not _has_cuda_device(), reason="needs a CUDA device")
@pytest.mark.parametrize("name", GPU_ONLY)
def test_import_gpu(name):
    importlib.import_module(name)
