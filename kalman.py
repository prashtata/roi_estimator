import numpy as np

class FaceKalman:
    def __init__(self, n_points=468, alpha=0.85):
        self.alpha = alpha
        self.state = np.zeros((n_points, 3), dtype=np.float32)
        self.initialized = False

    def update(self, z):
        if not self.initialized:
            self.state[:] = z
            self.initialized = True
        else:
            self.state = self.alpha * self.state + (1 - self.alpha) * z
        return self.state
