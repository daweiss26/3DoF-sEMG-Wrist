import numpy as np

class AdaptiveScaler:
    """Adjusts normalization dynamically using Exponential Moving Average"""

    def __init__(self, initial_scale, alpha=0.001, min_scale=1.00):
        self.scale = initial_scale
        self.alpha = alpha # How fast we adapt (lower = slower/stabler)
        self.min_scale = min_scale # Prevent division by zero or noise amplification

    def update_and_normalize(self, window):
        current_max = np.percentile(np.abs(window), 99)
        
        # We need to be careful here that we don't adapt to rest
        # Otherwise, it will jump when we give a new signal
        if current_max > (0.1 * self.scale):
            self.scale = (self.alpha * current_max) + ((1 - self.alpha) * self.scale)

        self.scale = max(self.scale, self.min_scale)
        return window / self.scale
