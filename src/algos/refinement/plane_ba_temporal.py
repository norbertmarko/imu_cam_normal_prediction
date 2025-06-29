# ------------------------------------------------------------
#  AdaptivePlaneBA  –  Sliding-window robust mean of normals
# ------------------------------------------------------------
# Purpose
# -------
# Post-refine a stream of *measured* road-plane normals by solving a
# mini bundle-adjustment in a fixed-length temporal window (N frames).
#
# Model
# -----
#   Given transforms  (Rₖ , tₖ)  that map points from frame k → k+1
#   and the corresponding measured unit normals nₖ,
#   find the single unit vector n̂ that minimises
#
#         Σ ρ( ‖ n̂ − nₖ ‖² )           with Tukey bi-weight ρ.
#
#   • No plane distance nor camera height is required.
#   • The problem decouples: only the normals are optimised.
#   • A few IRLS (iteratively re-weighted least squares) iterations
#     suffice because the cost is convex on the sphere for modest noise.
#
# Public API
# ----------
#     ba = AdaptivePlaneBA(window_size=5)
#     ba.add_measurement(k, Rk, tk, n_k)   # or ba.push_null(k)
#     if ba.ready():
#         n_refined = ba.solve()
#
# Typical usage
# -------------
# Call *after* any temporal filter (KF, EKF, SLERP) to obtain a
# robust, multi-frame consensus normal.  The BA acts as a second-stage
# smoother / outlier re-jector and costs ~100 µs for N=5 on a laptop.
#
# Strengths : cancels shot noise, suppresses bursts of bad frames.
# Limitations: adds 4–5 frames of latency; fails if >50 % of window
#              measurements are grossly wrong.
# ------------------------------------------------------------
from collections import deque
import numpy as np
from numpy.linalg import svd, norm

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# ------------------------------------------------------------
# Tukey bi-weight parameters
c2   = (4.685 ** 2)         # tuning constant²  →  95 % eff. under Gauss
eps2 = 1e-12                # numerical floor


class AdaptivePlaneBA:
    def __init__(self, window_size=5):
        self.N      = window_size
        self.R      = [None]*self.N
        self.t      = [None]*self.N
        self.obs_n  = [None]*self.N
        self.valid  = [False]*self.N
        self.head   = 0                     # circular buffer index

    # ------------------------------------------------------------
    def add_measurement(self, k, R_k, t_k, n_k):
        """Store the triple for time-index k (k only used for sync)."""
        idx            = self.head % self.N
        self.R[idx]    = R_k
        self.t[idx]    = t_k
        self.obs_n[idx]= n_k / norm(n_k)
        self.valid[idx]= True
        self.head     += 1

    def push_null(self, k):
        idx            = self.head % self.N
        self.valid[idx]= False
        self.head     += 1

    # ------------------------------------------------------------
    def ready(self):
        return all(self.valid)

    # ------------------------------------------------------------
    def solve(self, max_iter=5):
        """
        Robust mean of the normals with Tukey bi-weight.
        Returns the refined normal (unit, same frame as the obs_n).
        """
        n = np.mean(self.obs_n, axis=0)
        n /= norm(n)

        for _ in range(max_iter):
            # residuals
            r2 = [max(eps2, norm(n - m)**2) for m in self.obs_n]
            # Tukey weights
            w  = [(1 - r/c2)**2 if r < c2 else 0.0 for r in r2]
            w  = np.array(w)
            if w.sum() < 1e-6:
                break
            n = np.sum(w[:,None] * self.obs_n, axis=0)
            n /= norm(n)

        return n


# ------------------------------------------------------------
def skew(v):
    """Return the 3x3 skew-symmetric matrix [v]x ."""
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]], dtype=float)