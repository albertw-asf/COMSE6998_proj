#!/usr/bin/env python
import os
import sys
import logging
import subprocess

# ———————————————
# 1) Put your project root on sys.path
# ———————————————
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ———————————————
# 2) Set up logging
# ———————————————
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ———————————————
# 3) Try the CUDA‐SSM kernels via mamba_ssm
# ———————————————
use_cuda_kernels = False
try:
    from mamba_ssm.ops import selective_scan_interface
    log.info("✅  Loaded CUDA SSM kernels from mamba_ssm.ops.selective_scan_interface")
    use_cuda_kernels = True
except ImportError as e:
    log.warning("⚠️  CUDA SSM kernels not found; falling back to PyKeOps")

    # 3a) Attempt to install pykeops if it isn't already present
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "pykeops"])
    except Exception:
        pass

    # 3b) Try importing pykeops
    try:
        import pykeops  # noqa: F401
        log.info("✅  PyKeOps available for SSM speedups")
    except ImportError as e2:
        log.warning(f"❌  PyKeOps not available ({e2}); using pure-Python fallback")

# ———————————————
# 4) Try Triton‐accelerated LayerNorm/RMSNorm
# ———————————————
use_triton_ln = False
try:
    import triton  # noqa: F401
    # if you have a local Triton LN implement, import it here:
    # from hiss.extensions.triton_layernorm import TritonLayerNorm
    log.info("✅  Triton LayerNorm/RMSNorm available")
    use_triton_ln = True
except ImportError:
    log.warning("⚠️  Triton LayerNorm/RMSNorm not available; using PyTorch defaults")

# ———————————————
# 5) Summary print
# ———————————————
log.info("")
log.info("=== Backend summary ===")
log.info(f"CUDA kernels:       {'enabled' if use_cuda_kernels else 'disabled'}")
log.info(f"PyKeOps fallback:   {'enabled' if (not use_cuda_kernels) else 'n/a'}")
log.info(f"Triton LayerNorm:   {'enabled' if use_triton_ln else 'disabled'}")

# ———————————————
# 6) Exit
# ———————————————
if __name__ == "__main__":
    log.info("Test fallback complete!")