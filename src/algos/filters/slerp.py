# ------------------------------------------------------------
#  RoadNormalSlerpFilter   –  Spherical EMA (“steady-state KF”)
# ------------------------------------------------------------
# Drift-free low-pass filter that keeps the estimate *on the
# unit sphere* by blending along the great-circle:
#
#     nₖ = SLERP(nₖ₋₁, n_meas, α(Rₖ))
#
# α = α₀ · σ₀ / (σ² + σ₀)  →  larger noise ⇒ stronger smoothing.
# A simple outlier gate (max_angle_deg) ignores grossly wrong inputs.
#
# Public API (drop-in):
#     filt.predict()        # NO-OP, kept for compatibility
#     filt.update(n_meas, Rk) → (φ, θ)
#     filt.get_pitch_roll()
#     filt.get_angle_cov()  # returns a huge dummy covariance
#
# Pros : geometry-exact, numerically cheap, only one tuning knob.
# Cons : gives no meaningful covariance; gain schedule is heuristic.
# ------------------------------------------------------------
import numpy as np
import numpy.linalg as npl


class RoadNormalSlerpFilter:
    def __init__(self,
                 alpha0=0.15,                # nominal gain  (0…1)
                 sigma0=np.deg2rad(3)**2,    # 1-σ² at which alpha≈alpha0
                 max_angle_deg=25.0):        # outlier gate
        self.n_hat = None
        self.alpha0 = alpha0
        self.sigma0 = sigma0
        self.max_ang = np.deg2rad(max_angle_deg)

    # -----------------------------------------------------------------
    def predict(self):
        """Nothing to predict – kept for drop-in compatibility."""
        pass

    # -----------------------------------------------------------------
    def update(self, n_meas, Rk=None):
        """
        n_meas – unit 3-vector (virtual frame)
        Rk     – 3×3 measurement covariance  (we only look at its trace)
        """
        n_meas = _norm(n_meas)

        if self.n_hat is None:                 # first frame
            self.n_hat = n_meas
            return self.get_pitch_roll()

        # ---- outlier gate ------------------------------------------------
        ang = np.arccos(np.clip(np.dot(self.n_hat, n_meas), -1, 1))
        if ang > self.max_ang:                 # skip this update
            return self.get_pitch_roll()

        # ---- adaptive gain ----------------------------------------------
        if Rk is None:
            alpha = self.alpha0
        else:
            sigma2 = np.trace(Rk) / 3.0             # rad²   (all diag equal)

            # NEW  — noise-to-gain mapping:
            #  σ = 0.10°  → α ≈ 0.70   (follow quickly)
            #  σ = 0.50°  → α ≈ 0.35
            #  σ = 3.00°  → α ≈ 0.15   (old behaviour)
            alpha = 0.7 * np.exp(-sigma2 / np.deg2rad(0.5)**2) + 0.15
            alpha = np.clip(alpha, 0.05, 0.8)       # safety clamp

        # debug
        print(
            f"[DBG] σ={np.sqrt(sigma2)*180/np.pi:4.2f}°  α={alpha:4.2f}  "
            f"measθ={(np.arcsin(-n_meas[0])*180/np.pi):5.2f}°"
        )

        # ---- spherical EMA ----------------------------------------------
        self.n_hat = slerp(self.n_hat, n_meas, alpha)
        return self.get_pitch_roll()

    # -----------------------------------------------------------------
    def get_pitch_roll(self):
        return normal_to_pitch_roll(self.n_hat)

    def get_angle_cov(self):
        # This simple filter does not keep a covariance – return “huge”
        return np.eye(2) * 1e6
    
    @property
    def state(self):     # for legacy checks
        return self.n_hat
    

# -------------------------- helpers ---------------------------------
def slerp(v1, v2, t):
    """Shortest great-circle interpolation between two unit vectors."""
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    if dot > 0.9995:                     # very small angle → linear is fine
        return _norm((1-t)*v1 + t*v2)
    theta   = np.arccos(dot)
    sin_t   = np.sin(theta)
    return (_norm(np.sin((1-t)*theta)/sin_t * v1 + np.sin(t*theta)/sin_t * v2))

def _norm(v):
    return v / npl.norm(v)

def normal_to_pitch_roll(n):
    """EKF-compat – returns roll φ, pitch θ  (radians) in your virtual frame."""
    n = _norm(n)
    phi   =  n[1]          # small-angle approx φ ≈ n_y
    theta = -n[0]          # θ ≈ −n_x
    return phi, theta
