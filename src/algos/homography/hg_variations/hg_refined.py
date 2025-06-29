import cv2
import numpy as np
import warnings
from typing import Tuple, Union

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.algos.homography.hg_funcs import calc_homography


def calc_homography_if_loftr(
	img1: np.ndarray,
	img2: np.ndarray,
	pts1: np.ndarray,
	pts2: np.ndarray,
	method: int = cv2.USAC_MAGSAC,
	threshold: float = 1.5,
	max_iter: int = 2000,
	conf_thresh: float = 0.995,
	name: str = "LoFTR",
):
	"""
	Calculate homography using matches from LoFTR.
	"""
	# check minimum points to compute a homography (4)
	if pts1.shape[0] < 4 or pts2.shape[0] < 4:
		warnings.warn(f"Not enough points for homography calculation: {pts1.shape[0]}")
		return None, None

	H, mask = calc_homography(
		pts1, pts2, method=method, threshold=threshold,
		max_iters=max_iter, conf_hg_thresh=conf_thresh
	)
	if H is None or mask is None:
		warnings.warn(f"Failed to find a homography ({name}).")
		return None, None
	return H, mask
