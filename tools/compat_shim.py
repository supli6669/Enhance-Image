"""
Universal Compatibility Shim for Custom AI Enhancer.
Fixes modern PyTorch / torchvision deprecations (e.g. functional_tensor removal in torchvision >= 0.18)
and guarantees backwards compatibility across BasicSR and Real-ESRGAN dependencies.
"""
import sys

def apply_shims():
    # 1. Shim torchvision.transforms.functional_tensor -> torchvision.transforms.functional
    try:
        import torchvision
        import torchvision.transforms.functional as F
        if 'torchvision.transforms.functional_tensor' not in sys.modules:
            sys.modules['torchvision.transforms.functional_tensor'] = F
    except ImportError:
        pass


# Scheduler resume handling lives in the vendored BasicSR scheduler.
# Importing BasicSR here would cache the wrong package before pipeline bootstrap.
apply_shims()
