import os
import warnings
from collections import deque
from pathlib import Path

from datetime import datetime
import cv2
import numpy as np
import hydra
from omegaconf import DictConfig

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import src.data_funcs.panda_funcs as panda
import src.algos.homography.hg_funcs as hg_funcs
import src.algos.homography.hg_decomp as hg_decomp
import src.algos.homography.hg_vis as hg_vis
import src.utils.data_utils as data_utils
import src.utils.linalg as linalg
import src.eval.eval_norm as eval_norm
import src.data_funcs.img_funcs as img_funcs

# transformer networks
from src.algos.homography.hg_variations.hg_refined import calc_homography_if_loftr
import src.algos.transformer.loftr_utils as loftr_utils
import src.algos.transformer.efficient_loftr as eloftr

# temporal filter
from src.algos.filters.slerp import RoadNormalSlerpFilter

# filter misc funcs
from src.algos.filters.filter_funcs import (
	compute_measurement_variance, normal_from_angles, angles_from_normal,
	select_normal_candidate, fast_decompose, to_camera, to_virtual, _flip_to_ref
)
# plane BA
from src.algos.refinement.plane_ba_temporal import AdaptivePlaneBA


def calc_homography_interframe(
		img1, img2, roi_pts, invert_direction=False,
		num_points=100, method=cv2.RANSAC, threshold=2.0
):
	"""
	Calculate homography between two consecutive frames.

	Parameters:
		img1 (numpy.ndarray): first image (t-1).
		img2 (numpy.ndarray): second image (t).
		roi_pts (list of tuples): List of four (x, y) tuples defining the ROI polygon.
		invert_direction (bool): Whether to invert the direction of the homography calculation.
		num_points (int): Number of points to use for homography calculation.
		method (cv2 method): Method to use for homography calculation.
		threshold (float): Threshold value for RANSAC method.

	Returns:
		homography_matrix (numpy.ndarray): The calculated homography matrix.
		mask (numpy.ndarray): Mask of inliers used in homography calculation.
		keypoints1, keypoints2 (list of cv2.KeyPoint): Keypoints from both images.
		matches (list of cv2.DMatch): Matched keypoints.
		src_pts, dst_pts (numpy.ndarray): Source and destination points used for homography.
	"""
	mask1 = hg_funcs.generate_roi_mask(img1, roi_pts)
	mask2 = hg_funcs.generate_roi_mask(img2, roi_pts)

	detector = 'SIFT'
	keypoints1, descriptors1 = hg_funcs.detect_keypoints(img1, detector, mask=mask1)
	keypoints2, descriptors2 = hg_funcs.detect_keypoints(img2, detector, mask=mask2)

	# check keypoints' and descriptors' validity
	if keypoints1 is None or keypoints2 is None:
		warnings.warn("Keypoints not found for one of the images.", UserWarning)
		return None, None, None, None, None, None, None
	if descriptors1 is None or descriptors2 is None:
		warnings.warn("Descriptors not found for one of the images.", UserWarning)
		return None, None, keypoints1, keypoints2, None, None, None

	# match keypoints between images
	matches = hg_funcs.match_features(descriptors1, descriptors2, detector)

	# filter and select keypoints for homography
	pts_set_1, pts_set_2 = hg_funcs.filter_matches(
		matches, keypoints1, keypoints2, num_points
	)
	# check minimum points to compute a homography (4)
	if pts_set_1 is None or pts_set_2 is None or len(pts_set_1) < 4 or len(pts_set_2) < 4:
		warnings.warn("Not enough points for homography calculation.", UserWarning)
		return None, None, keypoints1, keypoints2, matches, pts_set_1, pts_set_2

	if invert_direction:
		src_pts, dst_pts = pts_set_2, pts_set_1
	else:
		src_pts, dst_pts = pts_set_1, pts_set_2
	# calculate homography matrix
	hg_mat, mask = hg_funcs.calc_homography(src_pts, dst_pts, method, threshold)

	return hg_mat, mask, keypoints1, keypoints2, matches, src_pts, dst_pts


def process_data(data_root, seq_num):
	"""
	Load and process data for a single sequence.
	"""
	dataset, seq_list = panda.read_pandaset(data_root)
	cam_obj = panda.load_panda_seq_cam(dataset, seq_list, seq_num)
	gps_obj = panda.load_panda_seq_gps(dataset, seq_list, seq_num)
	_, abs_rot_mats, pitch_xyz, vels = panda.parse_panda_seq_poses(cam_obj, gps_obj)
	camera_K = panda.get_panda_seq_intrinsics(cam_obj)
	images = panda.get_panda_seq_images(cam_obj, pil=False)
	return images, camera_K, pitch_xyz


def decompose(hg_mat, intrinsics, last_n_norms, handle_outliers=False, debug=False):
	"""
	Normalize, decompose and extract normal vector from homography matrix
	along with the euler angles (ego-motion).
	"""
	hg_decomp.normalize_homography(hg_mat, intrinsics)
	solution, all_sols = hg_decomp.decompose_homography(
		hg_mat, intrinsics, last_n_norms, handle_outliers, debug=debug,
	)
	yaw, pitch, roll = hg_decomp.tf_rot_mat_to_euler(solution["rotation"])
	return solution, all_sols, yaw, pitch, roll


@hydra.main(
		version_base="1.3", config_path="./configs",
		config_name="exp_hg_panda_ts.yaml"
)
def main(cfg: DictConfig):
	"""
	Main entry point for homography-based experiment pipeline.
	"""
	output_dir = os.path.join(cfg.output_dir)
	os.makedirs(output_dir, exist_ok=True)
	eval_output_path = os.path.join(
		output_dir, cfg.seq_num, f"eval_data_{cfg.seq_num}.pkl"
	)
	os.makedirs(os.path.join(output_dir, cfg.seq_num), exist_ok=True)
 
	vis = cfg.vis
	save_vis = cfg.save_vis

	# load data
	images, camera_K, _ = process_data(cfg.data_root, cfg.seq_num)
	if len(images) < 2:
		print("Not enough images for homography.")
		return

	# handle ROI
	roi_idx = 0
	if not cfg.roi:
		default_roi = [(550, 430), (730, 430), (950, 720), (265, 720)]
		print("No ROI provided. Using default ROI:", default_roi)
		roi = default_roi
	else:
		print(cfg.roi)
		roi = list(map(tuple, cfg.roi['panda'][roi_idx].points))
		print("Using ROI:", roi)
		
	# rescale ROI (always rescale; if curr. img size is the same, no change)
	roi_img_src_size = cfg.roi_img_src_size.panda
	roi_img_dst_size = (images[0].shape[1], images[0].shape[0])
	roi = hg_funcs.calculate_scaled_roi(roi, roi_img_src_size, roi_img_dst_size)

	# load ground truth
	gt_dir = os.path.join(cfg.gt_dir, f"{cfg.seq_num}", "gt")
	gt_path = os.path.join(gt_dir, f"gt_data_{cfg.seq_num}.pkl")
	gt_data = data_utils.load_pickle_data(gt_path)

	# init data stores
	MAX_NORM_HISTORY = 5
	last_n_norms = {
		"normals": deque(maxlen=MAX_NORM_HISTORY),
		"rotations": deque(maxlen=MAX_NORM_HISTORY),
		"translations": deque(maxlen=MAX_NORM_HISTORY)
	}
	pitch_vals, pitch_vals_gt = [], []
	data_store, eval_data_store = {}, {}
	last_valid_solution = None
	plane_change_metrics_log = []


	# temporal filter initialization

	temp_filter_type = cfg.temporal_filter     # "KF", "EKF", "MEKF", "SLERP", or None
	use_plane_ba = cfg.use_plane_ba       	   # True / False
	plane_ba_window = cfg.ba_window_size  	   # e.g. 5
	fps = cfg.fps 						  	   # frame rate of the sequence

	# filtered parameters init
	phi = 0.0
	theta = 0.0

	focal_px = 0.5 * (camera_K[0,0] + camera_K[1,1])

	if temp_filter_type not in ["SLERP"]:
		warnings.warn(
			f"Unknown filter '{temp_filter_type}'. Using no filter.",
			UserWarning
		)
		filt = None
	elif temp_filter_type == "SLERP":
		filt = RoadNormalSlerpFilter(alpha0=0.15, sigma0=np.deg2rad(3)**2)

	# init plane BA filter (if required)
	if use_plane_ba:
		ba_filter = AdaptivePlaneBA(
			window_size=plane_ba_window
		)
	else:
		ba_filter = None

	# -------------------------------------------------
	plane_BA = AdaptivePlaneBA(window_size=5)   # <-- keep =5 unless you have <10 Hz
	# -------------------------------------------------


	# iterate over consecutive image pairs
	for i in range(len(images) - 1):
		img1 = images[i]
		img2 = images[i+1]
		
		print(f"Processing: image {i} and {i+1}, ROI {roi}")

		# process images (e.g. resize, normalize etc.)
		img1 = hg_funcs.proc_img(img1, roi_img_dst_size)
		img2 = hg_funcs.proc_img(img2, roi_img_dst_size)

		# --- TODO: GOES INTO PROCESSING IN hg_funcs --- #

		# prep images for homography
		diameter=7
		sigma_color=25
		sigma_space=25

		img1 = img_funcs.hist_equalization(img1, adaptive=True)
		# img1 = img_funcs.LaplacianFilter(always_apply=True)(image=img1)["image"]
		# img1 = img_funcs.bilateral_filter(img1, diameter=diameter, sigma_color=sigma_color, sigma_space=sigma_space)

		img2 = img_funcs.hist_equalization(img2, adaptive=True)
		# img2 = img_funcs.LaplacianFilter(always_apply=True)(image=img2)["image"]
		# img2 = img_funcs.bilateral_filter(img2, diameter=diameter, sigma_color=sigma_color, sigma_space=sigma_space)

			# matcher

		WEIGHTS_ROOT = Path(cfg.repo_root) / "src" / "_experiments" / "weights"

		# get raw matches through LOFTR then restrict to ROI
		src, dst, conf = eloftr.match_eloftr(
			img1,
			img2,
			weights=str((WEIGHTS_ROOT / "eloftr" / "eloftr_outdoor.ckpt").resolve()),
			conf_thresh=0.3
		)
		src_pts_roi, dst_pts_roi, conf_pts_roi = eloftr.filter_matches_to_roi(
			src, dst, conf, roi, img1.shape[:2]
		)

		# calculate homography

		hg_mat, mask = calc_homography_if_loftr(
			img1,
			img2,
			src_pts_roi,
			dst_pts_roi,
			method=cv2.USAC_MAGSAC,
			threshold=0.7,
			max_iter=10000,
			conf_thresh=0.995
		)
		# propagate the correct matches
		src_pts = src_pts_roi
		dst_pts = dst_pts_roi
		
	
		# if homography is valid: decompose + temporal filter update (optional)
		if hg_mat is not None:
			
			candidates = fast_decompose(hg_mat, camera_K)
			
			if cfg.debug:
				# print all candidate normals
				for cand in candidates:
					n_cam = _flip_to_ref(cand["normal"])
					print("[DBG] candidate:", angles_from_normal( to_virtual(n_cam) ))

			# choose candidate (returns: virtual normal measurement)
			sol = select_normal_candidate(candidates, filt)
			n_meas_virt = sol["normal"]  # EKF frame (right-handed +x-forward)

			# calculate adaptive measurement noise
			sigma2 = compute_measurement_variance(
				hg_mat, src_pts, dst_pts, mask,
				focal_px, conf_pts_roi,
				fallback_sigma_rad=np.deg2rad(3)
			)
			Rk = np.eye(3) * sigma2

		else:
			# if homography failed: keep prediction only
			if filt is not None:
				filt.predict()
			hg_mat = np.eye(3, dtype=np.float32)


		# getting the appropriate "normal_refined"
		if filt is not None:
			# filter update
			phi, theta = filt.update(n_meas_virt, Rk)

			# output (make it compatible with downstream code)
			n_refined_virt = normal_from_angles(phi, theta)
			normal_refined = to_camera(n_refined_virt)
		else:
			normal_refined = n_meas_virt


		# Plane BA (optional)
		if ba_filter is not None:
			# no R_k and t_k
			ba_filter.add_measurement(i, np.eye(3), np.zeros(3), normal_refined)
			if ba_filter.ready():
				normal_refined = ba_filter.solve()


		# access GT data
		gt = data_utils.extract_gt_data(gt_data, i, roi_idx)
		if gt is None:
			print(f"[WARN] No GT for frame {i}, skipping.")
			continue
		proj_pts_2d_gt = gt["proj_pts_2d"]
		normal_gt = gt["normal"]
		cam_pts_3d_gt = gt["cam_pts_3d"]
		plane_coeffs_gt = gt["plane_coeffs"]
		centroid_gt = gt["centroid"]
		ref_frame = gt["ref_frame"]
		R_cam_to_lidar = gt["R_cam_to_lidar"]


		# rotate and align chosen normal vector
		normal_refined = linalg.rotate_vector(normal_refined, R_cam_to_lidar)
		normal_refined = linalg.align_normal_ref_vec(normal_refined, normal_gt)
		
		# calculate pitch values
		pitch_gt = hg_funcs.calculate_pitch_from_normal(normal_gt, frame="y-backward")
		pitch_refined = hg_funcs.calculate_pitch_from_normal(normal_refined, frame="y-backward")
		

		# --- EVALUATION --- #

		# evaluation for a single example
		signed_norm_angle_deg, signed_pitch_error_deg, \
			pitch_error_deg, norm_angle_deg, reproj_err_px = eval_norm.eval_example(
			normal_refined, normal_gt, pitch_refined, pitch_gt,
			hg_mat, src_pts, dst_pts, mask,
			i, print_results=True
		)

		# store results
		pitch_vals.append(pitch_refined)
		pitch_vals_gt.append(pitch_gt)

		data_store[(i, i+1)] = {
			"homography_matrix": hg_mat,
			"normal": normal_refined,
			"normal_gt": normal_gt,
			"pitch": pitch_refined,
			"pitch_gt": pitch_gt,
			"norm_angle_deg": norm_angle_deg,
			"signed_norm_angle_deg": signed_norm_angle_deg,
			"pitch_error_deg": pitch_error_deg,
			"signed_pitch_error_deg": signed_pitch_error_deg,
			"mean_reproj_err_px": reproj_err_px
		}

		# store evaluation data
		eval_data_store[i] = {
			"normal": normal_refined,
			"normal_gt": normal_gt,
			"pitch": pitch_refined,
			"pitch_gt": pitch_gt,
			"norm_angle_deg": norm_angle_deg,
			"signed_norm_angle_deg": signed_norm_angle_deg,
			"pitch_error_deg": pitch_error_deg,
			"signed_pitch_error_deg": signed_pitch_error_deg,
			"mean_reproj_err_px": reproj_err_px
		}

		# visualizations

		# hg_vis.vis_ransac(
		# 	img1, img2, keypoints1, keypoints2, matches, mask,
		# 	output_dir, cfg.seq_num, i, vis, save_vis
		# )

		# matches_img = loftr_utils.vis_matches_loftr(
		# 	img1, img2, src_pts, dst_pts, conf=conf_pts_roi,
		# 	roi_pts=roi, root=output_dir, seq_num=cfg.seq_num,
		# 	idx=i, show=vis, save=save_vis
		# )

		blended_img = loftr_utils.vis_blend_loftr(
			img1, img2, hg_mat, src_pts, dst_pts, conf=conf_pts_roi,
			roi_pts=roi, invert_direction=False, alpha=0.6,
			root=output_dir, seq_num=cfg.seq_num, idx=i, 
			show=vis, save=save_vis
		)

		# baseline plots
		hg_vis.plot_complex_normal(
			blended_img, hg_mat, normal_refined,
			cam_pts_3d_gt, plane_coeffs_gt, centroid_gt, normal_gt,
			signed_norm_angle_deg,
			output_dir, cfg.seq_num, i, vis, save_vis,
			normal_candidates=None
		)

		hg_vis.plot_complex_pitch(
			blended_img, hg_mat, pitch_vals,
			pitch_vals_gt, pitch_error_deg,
			output_dir, cfg.seq_num, i, vis, save_vis
		)			

		if cfg.single_frame:		
			break


	# pitch sensitivity analysis
	hg_vis.plot_hg_param_sensitivity(
		pitch_vals, pitch_vals_gt,
		[hg_data["homography_matrix"] for key, hg_data in data_store.items()],
		output_dir, cfg.seq_num, vis, save_vis
	)
	timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
	eval_norm.eval_sequence(data_store, output_dir, cfg.seq_num, timestamp, print_results=True)

	# save data for evaluation (used for statistics and performance evaluation also)
	data_utils.save_pickle_data(eval_data_store, eval_output_path)


if __name__ == '__main__':
	main()
