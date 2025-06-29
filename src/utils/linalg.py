import numpy as np


def align_normal_ref_vec(normal_calc, normal_ref):
	"""
	Aligns the calculated normal to point towards the reference normal.

	:param normal_calc: Calculated normal vector (numpy array).
	:param normal_ref: Ground truth normal vector (numpy array).
	:return: Aligned Calculated normal vector (numpy array).
	"""
	# Input Validation
	normal_calc = np.asarray(normal_calc, dtype=float)
	normal_ref  = np.asarray(normal_ref , dtype=float)

	if normal_calc.shape != (3,) or normal_ref.shape != (3,):
		raise ValueError("Both normal_calc and normal_ref must be 3-element vectors.")

	if np.linalg.norm(normal_ref) == 0:
		raise ValueError("Reference normal must be non-zero.")

	# Alignment
	if np.dot(normal_calc, normal_ref) < 0:
		normal_calc = -normal_calc

	return normal_calc


def align_normal_ref_pt(normal_calc, ref_pt=None, plane_ctr=None):
	"""
	Aligns the calculated normal vector to point towards a reference direction.
	
	:param normal_calc: Calculated normal vector (numpy array).
	:param ref_pt: A reference point for alignment (e.g., camera position).
	:param plane_ctr: Center of the plane (used for consistency).
	:return: Aligned Calculated normal vector (numpy array).

	Example:
	--------
	>>> normal_calc = np.array([0, 0, -1])
	>>> ref_pt = np.array([0, 0, 1])    # camera position in right-handed +y-forward frame
	>>> plane_ctr = np.array([0, 0, 0])  # center of the plane in the same frame
	>>> align_normal_ref_pt(normal_calc, ref_pt)
	res: array([ 0.,  0.,  1.])
	"""
	# Input Validation
	normal_calc = np.asarray(normal_calc, dtype=float)
	ref_pt      = np.asarray(ref_pt     , dtype=float)
	plane_ctr   = np.asarray(plane_ctr  , dtype=float)

	# All three must be length-3
	for name, vec in (("normal_calc", normal_calc),
					  ("ref_pt",      ref_pt),
					  ("plane_ctr",   plane_ctr)):
		if vec.shape != (3,):
			raise ValueError(f"{name} must be a 3-element vector.")

	# The reference direction is from the plane center toward ref_pt:
	normal_ref = ref_pt - plane_ctr
	if np.linalg.norm(normal_ref) == 0:
		raise ValueError("ref_pt and plane_ctr must not coincide; need a non-zero reference direction.")

	# Alignment
	if np.dot(normal_calc, normal_ref) < 0:
		normal_calc = -normal_calc

	return normal_calc


def rotate_vector(vec, R):
	"""Rotate a vector using a rotation matrix R, then normalize it."""
	vec_tf = R @ vec
	vec_tf /= np.linalg.norm(vec_tf)
	return vec_tf
