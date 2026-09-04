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

    # 3. Patch BasicSR CosineAnnealingRestartLR out-of-bounds bug when resuming past original cycle
    try:
        import math
        import basicsr.models.lr_scheduler as b_sched

        def safe_get_position_from_periods(iteration, cumulative_period):
            for i, period in enumerate(cumulative_period):
                if iteration <= period:
                    return i
            return len(cumulative_period) - 1

        b_sched.get_position_from_periods = safe_get_position_from_periods

        def safe_get_lr(self):
            idx = safe_get_position_from_periods(self.last_epoch, self.cumulative_period)
            if idx is None or idx >= len(self.restart_weights):
                idx = len(self.restart_weights) - 1
            current_weight = self.restart_weights[idx]
            nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
            current_period = self.periods[idx] if idx < len(self.periods) else self.periods[-1]

            if self.last_epoch >= self.cumulative_period[-1]:
                return [self.eta_min for _ in self.base_lrs]

            return [
                self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
                for base_lr in self.base_lrs
            ]

        b_sched.CosineAnnealingRestartLR.get_lr = safe_get_lr
    except Exception:
        pass

apply_shims()
