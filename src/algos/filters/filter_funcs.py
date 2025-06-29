import numpy as np
import cv2


def fast_decompose(hg_mat, intrinsics):
	"""Return *all* physically valid solutions from a homography."""
	# normalize by camera intrinsic matrix
	hg_mat_intrinsics_norm = np.linalg.inv(intrinsics) @ hg_mat @ intrinsics
	# scale the homography so that H[2, 2] = 1
	hg_mat_final_norm = hg_mat_intrinsics_norm / hg_mat_intrinsics_norm[2, 2]
	_, Rs, Ts, Ns = cv2.decomposeHomographyMat(hg_mat_final_norm, np.eye(3))
	return [{"rotation": Rs[i],
			 "translation": Ts[i].flatten(),
			 "normal": Ns[i].flatten()}          # not unit yet
			for i in range(len(Rs))]

# ------------------------------------------------------------------#
#  CONSTANT ROTATION:  camera(+z-fwd) → virtual(+z-up)              #
# ------------------------------------------------------------------#
_RX_NEG90 = np.array([[1, 0, 0],
					  [0, 0, 1],     # cos=-, sin=-1  →  see note*
					  [0,-1, 0]], dtype=float)   #  *sin(-90°) = -1
_RX_POS90 = _RX_NEG90.T             # inverse = transpose (pure rotation)

# Helper: bring cam-frame vector ↔ EKF (virtual) frame
to_virtual   = lambda v: _RX_NEG90 @ v
to_camera    = lambda v: _RX_POS90 @ v
# ------------------------------------------------------------------#

# ---------- EKF ⇄ vector (unchanged math, now in virtual frame) ---#
def normal_from_angles(phi, theta):
	"""roll φ, pitch θ (rad)  →  unit normal,  EKF frame (+z up)."""
	n = np.array([-theta,  # n_x ≈ −θ
				   phi,    # n_y ≈  φ
				   1.0])
	return n / np.linalg.norm(n)

def angles_from_normal(n):
	n = n / np.linalg.norm(n)
	return n[1], -n[0]     # φ, θ
# ------------------------------------------------------------------#


def compute_measurement_variance(H, src_pts, dst_pts, inlier_mask,
                                 focal_px,
                                 conf=None, k_conf=0.5,
                                 min_inliers=10,
                                 fallback_sigma_rad=np.deg2rad(5)):
    """
    Robust σ² for R_k = σ² · I₃   [rad²]
    --------------------------------------------------------------
    - base term      : reprojection error / focal
    - confidence     : down-weights when LoFTR (or similar) is high
    - geo-inflation  : blows up σ when horizontal motion is weak
    - candidate-spread: blows up σ when the 4 homography roots disagree
    """

    # ---------- sanity checks -------------------------------------------------
    if (src_pts is None) or (dst_pts is None) or (inlier_mask is None):
        return fallback_sigma_rad ** 2

    ok = inlier_mask.ravel() == 1
    if ok.sum() < min_inliers:
        return fallback_sigma_rad ** 2

    # ---------- 1. base term  --------------------------------------------------
    proj   = cv2.perspectiveTransform(src_pts[ok][:, None, :], H)[:, 0, :]
    err_px = np.linalg.norm(proj - dst_pts[ok], axis=1)          # reproj-err
    sigma_rad = np.median(err_px) / focal_px                     # rad

    # ---------- 2. confidence scaling  ----------------------------------------
    if (conf is not None) and (len(conf) == len(src_pts)):
        mean_conf  = np.mean(conf[ok])            # 0…1
        sigma_rad *= (1.0 + k_conf * (1.0 - mean_conf))

    # ---------- 3. geo-inflation  ---------------------------------------------
    # horizontal vs. vertical flow
    dx = np.median(np.abs(dst_pts[ok, 0] - src_pts[ok, 0]))
    dy = np.median(np.abs(dst_pts[ok, 1] - src_pts[ok, 1]))

    # very little horizontal flow  → inflate strongly   (25 px = “good”)
    geo_gain = max(1.0, 25.0 / (dx + 1.0))
    # extremely vertical-dominant frames  → inflate again
    anisotropy = max(1.0, (dy + 1e-3) / (dx + 1e-3))
    sigma_rad *= geo_gain * anisotropy

    # ---------- 4. candidate-spread inflation  --------------------------------
    try:
        all_roots = fast_decompose(H, np.eye(3))             # no intrinsics
        if len(all_roots) == 4:
            # angular spread of the 4 normals  (EKF frame)
            n = np.stack([r["normal"] / np.linalg.norm(r["normal"])
                          for r in all_roots])
            ang = np.arccos(np.clip(n @ n.mean(axis=0), -1, 1))
            spread_gain = 1.0 + np.rad2deg(np.max(ang)) / 5.0     # 5° → ×2
            sigma_rad *= spread_gain
    except Exception:
        pass  # decomposition sometimes fails – ignore and keep current σ

    # ---------- 5. very small technical floor ---------------------------------
    sigma_rad = max(sigma_rad, np.deg2rad(0.05))   # 0.05° (≈ 1 px at 1 m)

    return sigma_rad ** 2

# -------------------- decompose + candidate picking ----------------#
_REF_CAM = np.array([0.0, -1.0,  0.0])               # “up” in cam

def _flip_to_ref(n_cam):
	"""flip so the angle to reference `[0,-1,0]` ≤ 90°."""
	if np.dot(n_cam, _REF_CAM) < 0:
		n_cam = -n_cam
	return n_cam / np.linalg.norm(n_cam)


def select_normal_candidate(
		candidates,
		ekf,
		ref_cam=np.array([0.0, -1.0, 0.0]),   # “up” in camera frame
		x_max=0.15,                           # |roll|  ≲   8.6°
		y_max=-0.7,                           # n_y ≤ -0.7   (≥ 45° from ground)
		z_range=(-0.35, 0.35),                # modest slope (≈ ±20°)
		max_dev_deg=45.0):                    # keep solutions ≤45° from ref
	"""
	Pick the most plausible ground-plane normal among the homography
	decomposition solutions.

	All tests are carried out **in the camera frame**; the chosen normal is
	then rotated to the EKF frame before it is returned.

	Parameters
	----------
	candidates : list of dicts from `fast_decompose`
	ekf        : your EKF / Slerp filter (may still be un-initialised)
	"""
	if len(candidates) == 0:
		raise ValueError("select_normal_candidate: empty candidate list")

	# ------------------------------------------------------------------
	# A. EKF prediction — expressed in CAMERA frame for the distance test
	# ------------------------------------------------------------------
	if ekf.state is None:
		φp, θp = 0.0, 0.0
		P_pred = np.eye(2) * 1e+2            # very large cov.
	else:
		φp, θp = ekf.get_pitch_roll()
		P_pred = ekf.get_angle_cov()

	S_inv = np.linalg.inv(P_pred)
	n_pred_cam = to_camera(normal_from_angles(φp, θp))

	# helper -------------------------
	def mahalanobis2(n_cam):
		"""d² between candidate and EKF prediction in (φ,θ) space."""
		n_virt = to_virtual(n_cam)
		φc, θc = angles_from_normal(n_virt)
		v = np.array([φc - φp, θc - θp])
		return v.T @ S_inv @ v
	# --------------------------------

	# ------------------------------------------------------------------
	# B. Physically plausible & properly oriented solutions
	# ------------------------------------------------------------------
	max_dev_rad = np.deg2rad(max_dev_deg)
	valid = []

	for sol in candidates:
		n   = sol["normal"].astype(float)
		t   = sol["translation"].astype(float)

		# 1) correct cheirality – the plane must lie in front of the camera
		if t[2] <= 0.0:
			continue

		# 2) orient the normal so that it points **away from the ground**
		#    (smallest angle w.r.t. ref_cam = [0,-1,0]).
		if np.dot(n, ref_cam) < 0.0:
			n = -n
			t = -t                                    # keep H = R + t nᵀ / d

		# 3) crude geometric bounds in CAMERA frame
		if abs(n[0]) > x_max:                         # roll too large
			continue
		if n[1] > y_max:                              # must point up (-y)
			continue
		if not (z_range[0] <= n[2] <= z_range[1]):    # pitch too large
			continue

		# 4) angular deviation from straight-up limited to max_dev_deg
		dev = np.arccos(np.clip(np.dot(n / np.linalg.norm(n),
									   ref_cam / np.linalg.norm(ref_cam)),
								-1.0, 1.0))
		if dev > max_dev_rad:
			continue

		valid.append((sol, n, t))

	# ------------------------------------------------------------------
	# C. Choose the statistically closest normal, fall back gracefully
	# ------------------------------------------------------------------
	if len(valid) == 0:                              # nothing passed the tests
		# –> take the one whose CAM-normal is closest to ref_cam
		best_sol = max(candidates,
					   key=lambda s: np.dot(ref_cam,
											(_flip_to_ref(s["normal"]))))
		n_cam = _flip_to_ref(best_sol["normal"])
	else:
		best_sol, n_cam, t = min(valid, key=lambda p: mahalanobis2(p[1]))

	# ------------------------------------------------------------------
	# D. Pack result in the format the rest of the pipeline expects
	# ------------------------------------------------------------------
	n_virt = to_virtual(n_cam) / np.linalg.norm(n_cam)
	best_sol = best_sol.copy()          # do not mutate input list
	best_sol["normal_cam"] = n_cam      # (optional, handy for debugging)
	best_sol["normal"]     = n_virt     # *** what EKF / downstream uses ***
	return best_sol