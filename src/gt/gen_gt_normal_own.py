from typing import List, Optional
import os
from pathlib import Path
from collections import deque

from omegaconf import DictConfig
import hydra
import matplotlib.pyplot as plt
from matplotlib.path import Path as plt_path
import numpy as np
from sklearn.neighbors import LocalOutlierFactor  # RANSAC outlier removal 
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import cv2
import pandas as pd

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data_funcs import own_funcs
from src.gt import vis_gt_normal
import src.utils.data_utils as data
import src.utils.linalg as linalg
import src.algos.homography.hg_funcs as hg_funcs

# Video generation script:
# ./src/_experiments/scripts/ ???
# ./src/_experiments/scripts/ ???


def process_data(data_root, seq_num):
	"""
	Load and process data for a single sequence.
	"""
	calib_data = own_funcs.load_calib_from_yaml(
		Path(data_root) / seq_num / 'calib.yaml'
	)
	camera_K = calib_data['intrinsics']['matrix']
	Tr_lidar_to_cam = calib_data['transforms']['Tr_lidar_to_cam']
	Tr_lidar_to_lidar_center = calib_data['transforms']['Tr_lidar_to_lidar_center']
	seq_len = own_funcs.get_seq_len(data_root, seq_num)
	return camera_K, Tr_lidar_to_cam, Tr_lidar_to_lidar_center, seq_len


def filter_points_in_roi(points2d, roi):
	"""
	Filters 2D and corresponding 3D points based on whether they are inside the ROI.

	Args:
		points2d (numpy.ndarray): Nx2 array of 2D projected points.
		roi (list of list or tuple): List of (x, y) tuples defining the ROI polygon.

	Returns:
		filtered_points2d (numpy.ndarray): Mx2 array of 2D points inside the ROI.
		mask (numpy.ndarray): Boolean array of shape (N,) indicating points inside the ROI.
	"""
	# convert ROI to a Path object
	roi_path = plt_path(roi)
	# check which points are inside the ROI
	mask = roi_path.contains_points(points2d)
	# apply mask to filter points
	filtered_points2d = points2d[mask]
	
	return filtered_points2d, mask


def calc_ground_plane_ransac(points3d, threshold=0.01, max_iterations=100):
	"""
	Estimates a ground plane using RANSAC.

	:param points3d: Nx3 numpy array of 3D points.
	:param threshold: Distance threshold to consider a point an inlier.
	:param max_iterations: Maximum number of RANSAC iterations.
	:return: Plane coefficients (a, b, c, d) for the equation ax + by + cz + d = 0.
	"""
	best_plane = None
	max_inliers = 0

	for _ in range(max_iterations):
		# Randomly sample 3 points
		sample_indices = np.random.choice(points3d.shape[0], 3, replace=False)
		sample_points = points3d[sample_indices]

		# Calculate the plane coefficients (a, b, c, d)
		v1 = sample_points[1] - sample_points[0]
		v2 = sample_points[2] - sample_points[0]
		normal = np.cross(v1, v2)  # Normal vector to the plane

		if np.linalg.norm(normal) == 0:
			continue

		normal /= np.linalg.norm(normal)
		a, b, c = normal
		d = -np.dot(normal, sample_points[0])

		# calculate distances to the plane
		distances = np.abs(np.dot(points3d, normal) + d) / np.linalg.norm(normal)
		inliers = distances < threshold
		num_inliers = np.sum(inliers)

		# update the best plane if this one has more inliers
		if num_inliers > max_inliers:
			max_inliers = num_inliers
			best_plane = (a, b, c, d)

	return best_plane


def calc_ground_plane_normal(plane_coeffs):
	"""
	Extracts the normal vector from plane coefficients.

	:param plane_coeffs: Coefficients (a, b, c, d) of the plane equation.
	:return: Normalized normal vector (a, b, c).
	"""
	a, b, c, d = plane_coeffs
	normal = np.array([a, b, c])
	normal /= np.linalg.norm(normal)
	return normal


def calc_ground_plane_center(filtered_3d_inliers):
	centroid = np.mean(filtered_3d_inliers, axis=0)
	return centroid


def tf_pts_to_first_pose(pts, curr_pose_mat, velo_to_cam_mat, first_pose, target_frame="cam"):
	"""
	Transforms LiDAR points from the current frame into the first frame.
	
	The LiDAR points are first transformed to the first camera (odometry) frame.
	If target_frame is "velo", the points are then re-converted to the first LiDAR frame.
	
	Args:
		pts (np.ndarray): Array of shape (N, 3) containing LiDAR points.
		curr_pose_mat (np.ndarray): Current absolute pose (4x4) of the camera (T_n).
		velo_to_cam_mat (np.ndarray): Rigid-body transform (4x4) from LiDAR (velodyne) to camera frame.
		first_pose (np.ndarray): The first (reference) camera pose (4x4). Typically poses_abs[0].
		target_frame (str): Either "cam" (default) or "velo". If "cam", returns points in the first camera frame.
							If "velo", returns points in the first LiDAR (velodyne) frame.
							
	Returns:
		np.ndarray: Transformed points in the desired frame, shape (N, 3).
	"""
	# Compute the relative transformation from the first pose to the current pose.
	# T_relative = inv(first_pose) @ curr_pose_mat
	T_relative = np.linalg.inv(first_pose) @ curr_pose_mat

	# Compose with the LiDAR-to-camera transform:
	# T_total = T_relative @ velo_to_cam_mat
	T_total = T_relative @ velo_to_cam_mat

	# Convert the LiDAR points to homogeneous coordinates (N x 4)
	ones = np.ones((pts.shape[0], 1))
	pts_hom = np.hstack([pts, ones])  # shape: (N, 4)

	# Apply the total transformation:
	pts_transformed_hom = (T_total @ pts_hom.T).T  # shape: (N, 4)
	pts_first_cam = pts_transformed_hom[:, :3]      # points in the first camera (odometry) frame

	if target_frame == "cam":
		return pts_first_cam
	elif target_frame == "velo":
		# Convert points from the first camera frame back to the first LiDAR (velodyne) frame.
		# This is done by applying the inverse of the velo_to_cam_mat.
		Tr_cam_to_velo = np.linalg.inv(velo_to_cam_mat)
		ones_cam = np.ones((pts_first_cam.shape[0], 1))
		pts_first_cam_hom = np.hstack([pts_first_cam, ones_cam])  # shape: (N, 4)
		pts_transformed_velo_hom = (Tr_cam_to_velo @ pts_first_cam_hom.T).T
		pts_first_velo = pts_transformed_velo_hom[:, :3]
		return pts_first_velo
	else:
		raise ValueError("target_frame must be either 'cam' or 'velo'.")


def get_cam_tf_mat_own(cam_pose_curr,          # poses_abs_cam[i]   (4×4)
					   cam_pose_first,         # poses_abs_cam[0]   (4×4)
					   Tr_lidar_to_cam):       # 4×4, LiDAR → Cam
	"""
	Rotation that maps a vector expressed in the CURRENT camera frame
	into the LiDAR frame of the FIRST pose.

	Returns
	-------
	R_lidar0_cam_i : (3, 3) ndarray
		Use it exactly like ``rot_cam_to_lidar`` in the rest
		of your pipeline.
	"""

	# -----------------------------------------------------------
	# 1)  rotation   cam_i  -->  cam_0      (current → first cam)
	#     R_cam0_cam_i  =  R_cam0_world.T · R_cam_i_world
	# -----------------------------------------------------------
	R_cam_i_world = cam_pose_curr[:3, :3]      # from current pose matrix
	R_cam0_world  = cam_pose_first[:3, :3]     # from first pose matrix
	R_cam0_cam_i  = R_cam0_world.T @ R_cam_i_world   # ≡ inv(R0) · Ri

	# -----------------------------------------------------------
	# 2)  rotation   cam_0  -->  lidar_0     (fixed extrinsic)
	# -----------------------------------------------------------
	R_lidar0_cam0 = np.linalg.inv(Tr_lidar_to_cam)[:3, :3]

	# -----------------------------------------------------------
	# 3)  rotation   cam_i  -->  lidar_0     (what we need)
	# -----------------------------------------------------------
	R_lidar0_cam_i = R_lidar0_cam0 @ R_cam0_cam_i
	return R_lidar0_cam_i


def calc_gt_normal(
	data_root, seq_num, roi, cam_ref_pt, plane_ctr_pt, 
	results_root,
	tf_id=0, lidar_sensor=None, vis=False, 
	save_vis=False, single_frame=False
):
	
	# parameters for the spike-suppression filter
	WIN_SIZE   = 5        # length of running window (odd number, ≥5 is fine)
	X_JUMP_THRESH = 0.06  # normal x-component deviation threshold (0.05 is around 3–4 deg in pitch)

	# ring buffer (keep the last WIN_SIZE − 1 normals in it)
	normal_buffer: deque[np.ndarray] = deque(maxlen=WIN_SIZE - 1)

	ground_truth = {}

	camera_K, Tr_lidar_to_cam, Tr_lidar_to_lidar_center, seq_len = process_data(data_root, seq_num)

	poses_abs_cam = own_funcs.load_and_process_pose_data(
		data_root,
		seq_num,
		Tr_lidar_to_lidar_center,
		Tr_lidar_to_cam,
		seq_len
	)

	# iterate over images
	for i in range(seq_len):
		img = own_funcs.read_img_file(data_root, seq_num, i, invert_channels=False)
		pc, _ = own_funcs.read_pcd_file(data_root, seq_num, i, homogeneous=True)

		# skip any bad frames (very rare instances)
		if own_funcs.frame_is_invalid(img, pc, i):
			continue

		# visualize (debug)
		# o3d_pc = o3d.geometry.PointCloud()
		# o3d_pc.points = o3d.utility.Vector3dVector(pc[:, :3])
		# o3d.visualization.draw_geometries([o3d_pc])

		projected_pts, valid_pc, og_idxs = own_funcs.projection(
			img, pc, camera_K, Tr_lidar_to_cam
		)

		# visualize (debug)
		# vis_img = img.copy()
		# for pt in projected_pts.astype(int):
		# 	cv2.circle(vis_img, (pt[0], pt[1]), 2, (0, 255, 0), -1)
		# cv2.imshow("Projection", vis_img)
		# cv2.waitKey(0)

		print(f"Processing: image {i}, ROI {roi}")

		filtered_2d, roi_mask = filter_points_in_roi(projected_pts, roi)
		# own_funcs.visualize_filtered_projection(img, projected_pts, filtered_2d, roi)

		# map the ROI mask back to the original point cloud indices
		roi_idxs = og_idxs[roi_mask]
		# select the corresponding original points
		pc_og_filtered = pc[roi_idxs]

		# visualize (debug)
		# o3d_pc = o3d.geometry.PointCloud()
		# o3d_pc.points = o3d.utility.Vector3dVector(pc_og_filtered[:, :3])
		# o3d.visualization.draw_geometries([o3d_pc])
		
		frame = "first"
		pc_homogeneous = pc_og_filtered  # already in homogeneous coordinates (N, 4)

		# transform points in the first camera (odometry) frame
		pc_og_transformed = tf_pts_to_first_pose(
			pc_homogeneous[:, :3],   # LiDAR points (Nx3)
			poses_abs_cam[i],        # current camera pose (4x4)
			Tr_lidar_to_cam,         # LiDAR-to-camera transform (4x4)
			poses_abs_cam[0],        # reference camera pose (4x4)
			target_frame="velo"
		)

		# RANSAC ground plane estimation (with outlier removal)
		lof = LocalOutlierFactor(n_neighbors=50, contamination=0.01, metric='euclidean')
		try:
			inliers = lof.fit_predict(pc_og_transformed) > 0
		except ValueError as e:
			print(f"[WARN] Frame {i}: LOF failed ({e}), skipping.")
			continue
		filtered_3d_inliers = pc_og_transformed[inliers]

		ground_plane_coeffs = calc_ground_plane_ransac(
			filtered_3d_inliers,
			threshold=0.01,
			max_iterations=1000
		)

		# after you get the new normal, save its x for next iter
		ground_normal = calc_ground_plane_normal(ground_plane_coeffs)

		if ground_plane_coeffs is None:
			print("[WARN] Could not compute ground plane!")
			continue

		ground_normal = calc_ground_plane_normal(ground_plane_coeffs)
		ground_normal = linalg.align_normal_ref_pt(
			ground_normal,
			ref_pt=np.array(cam_ref_pt),
			plane_ctr=np.array(plane_ctr_pt)
		)

		# update buffer for the next frame
		normal_buffer.append(ground_normal)


		plane_centroid = calc_ground_plane_center(filtered_3d_inliers)

		# get camera to LiDAR transformation matrix (rotation only)
		# rot_cam_to_lidar = np.linalg.inv(Tr_lidar_to_cam)[:3, :3]

		rot_cam_to_lidar = get_cam_tf_mat_own(
			poses_abs_cam[i],        # current camera pose
			poses_abs_cam[0],        # first-frame camera pose
			Tr_lidar_to_cam)         # LiDAR→Cam from calib

		# visualize the results
		# vis_gt_normal.plot_complex_visualization(
		# 	img, filtered_2d, pc_og_transformed, ground_plane_coeffs,
		# 	ground_normal, plane_centroid, results_root, seq_num, i,
		# 	vis, save_vis, invert_img_channels=True
		# )			

		# save gt for current ROI
		ground_truth.setdefault(i, {})[0] = {
			"proj_pts_2d": filtered_2d.tolist(),
			"cam_pts_3d": pc_og_transformed.tolist(),
			"plane_coefficients": list(ground_plane_coeffs),
			"normal": ground_normal.tolist(),
			"centroid": plane_centroid.tolist(),
			"ref_frame": frame,
			"rot_cam_to_lidar": rot_cam_to_lidar.tolist()
		}

		if single_frame:
			break

	# save to disk
	output_path = os.path.join(results_root, seq_num, "gt", f"gt_data_{seq_num}.pkl")
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	data.save_pickle_data(ground_truth, output_path)		


@hydra.main(version_base="1.3", config_path="../configs/gt", config_name="gt_normal_own.yaml")
def main(cfg: DictConfig) -> None:
	"""
	Main entry point for ground truth generation.
	"""
	gt_root = os.path.join(cfg.repo_root, "results", "gt_own")
	os.makedirs(gt_root, exist_ok=True)

	# sequences too big, load files on the fly (load data would come here)

	# handle ROIs
	roi_idx = 0
	if not cfg.roi:
		default_roi = [[999, 916], [1293, 916], [1494, 1227], [780, 1228]]
		print("No ROI provided. Using default ROI:", default_roi)
		roi = default_roi
	else:
		roi = list(map(tuple, cfg.roi['own'][roi_idx].points))


	# rescale ROI (always rescale, if curr. img size is the same, no change)
	roi_img_src_size = cfg.roi_img_src_size.own

	roi_img_dst_size = own_funcs.read_img_file(
		cfg.processed_data_root, cfg.seq_num, frame_num=0
	).shape[:2]
	roi_img_dst_size = (roi_img_dst_size[1], roi_img_dst_size[0])
	roi = hg_funcs.calculate_scaled_roi(roi, roi_img_src_size, roi_img_dst_size)

	calc_gt_normal(
		cfg.processed_data_root, cfg.seq_num, roi, cfg.cam_ref_pt, cfg.plane_ctr_pt,
		gt_root, 
  		tf_id=cfg.tf_id, lidar_sensor=cfg.lidar_sensor,
	 	vis=cfg.vis, save_vis=cfg.save_vis, single_frame=cfg.single_frame
	)
	print(f"Ground truth generation complete for sequence {cfg.seq_num}!")


if __name__ == "__main__":
	main()
