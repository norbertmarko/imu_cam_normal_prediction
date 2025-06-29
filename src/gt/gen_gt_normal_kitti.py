import os
import yaml

from omegaconf import DictConfig
import hydra
import matplotlib.pyplot as plt
from matplotlib.path import Path as plt_path
import numpy as np
from sklearn.neighbors import LocalOutlierFactor  # RANSAC outlier removal 
import open3d as o3d

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data_funcs import kitti_funcs
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
	seq_len = kitti_funcs.get_kitti_cam_seq_len(data_root, seq_num)
	# images = kitti_funcs.get_kitti_seq_imgs(data_root, seq_num, seq_len)
	camera_K = kitti_funcs.read_kitti_calib(os.path.join(data_root, seq_num, "calib.txt"))
	return camera_K, seq_len


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


def get_cam_tf_mat_kitti(cam_pose_curr,        # poses_abs[i]   (4×4)
                         cam_pose_first,       # poses_abs[0]   (4×4)
                         Tr_velo_to_cam):      # 4×4 extrinsic  (LiDAR → Cam)
    """
    Return R (3×3) such that:
        v_lidar0 = R · v_cam_i

    where
        v_cam_i   – vector in the *current* camera frame  (Cam-i)
        v_lidar0  – the same vector in the first LiDAR frame (LiDAR-0)
    """

    # ---------------------------------------------------------
    # 1) current-camera  → first-camera
    #    R_cam0_cam_i = (R_world_cam0).T · R_world_cam_i
    # ---------------------------------------------------------
    R_world_cam_i = cam_pose_curr[:3, :3]
    R_world_cam0  = cam_pose_first[:3, :3]
    R_cam0_cam_i  = R_world_cam0.T @ R_world_cam_i

    # ---------------------------------------------------------
    # 2) first-camera  → first-LiDAR   (fixed extrinsic)
    # ---------------------------------------------------------
    R_lidar0_cam0 = np.linalg.inv(Tr_velo_to_cam)[:3, :3]

    # ---------------------------------------------------------
    # 3) current-camera → first-LiDAR  (final answer)
    # ---------------------------------------------------------
    R_lidar0_cam_i = R_lidar0_cam0 @ R_cam0_cam_i
    return R_lidar0_cam_i


def calc_gt_normal(
	data_root, seq_num, roi, cam_ref_pt, plane_ctr_pt, 
	results_root,
	tf_id=0, lidar_sensor=None, vis=False, 
	save_vis=False, single_frame=False
):
	ground_truth = {}

	kitti_pose_root = os.path.join(os.path.dirname(data_root), "poses")
	# check if poses are pre-processed (img ID added to poses)
	kitti_funcs.process_pose_files(kitti_pose_root)

	# sequences too big, load files on the fly
	camera_K, seq_len = process_data(data_root, seq_num)

	_, poses_abs = kitti_funcs.read_kitti_pose_abs(
		os.path.join(kitti_pose_root, f"{seq_num}_with_ids.txt")
	)
	_, Tr_velo_to_cam, _ = kitti_funcs.load_calib(data_root, seq_num)

	# iterate over images
	for i in range(seq_len):
		img = kitti_funcs.read_kitti_img_file(data_root, seq_num, i, invert_channels=True)
		pc = kitti_funcs.read_kitti_pc_file(data_root, seq_num, i)
		
		projected_pts, valid_pc, og_idxs = kitti_funcs.projection(img, pc, data_root, seq_num)
		# kitti_funcs.visualize_projection(img, projected_pts)


		print(f"Processing: image {i}, ROI {roi}")

		filtered_2d, roi_mask = filter_points_in_roi(projected_pts, roi)
		# kitti_funcs.visualize_filtered_projection(img, projected_pts, filtered_2d, roi)

		# map the ROI mask back to the original point cloud indices
		roi_idxs = og_idxs[roi_mask]
		# select the corresponding original points
		pc_og_filtered = pc[roi_idxs]
		

		# ??? T = inv(T_0) x T_n x T_velo2imu
		frame = "first"

		# convert LiDAR points to homogeneous coordinates (add a row of 1s)
		# ones = np.ones((1, pc_og_filtered.shape[0]))
		# pc_homogeneous = np.vstack((pc_og_filtered[:, :3].T, ones))  # Shape becomes (4, N)

		# transform LiDAR points to IMU frame
		# pc_og_imu_frame = Tr_velo_to_imu @ pc_homogeneous  # Now in IMU coordinates			

		# transform points to the first odometry frame
		# pc_og_transformed = tf_pts_to_first_pose(
		# 	pc_og_imu_frame[:3, :].T, poses_abs[i], poses_abs[0], use_first_pose=True
		# )

		# pc_og_transformed = pc_og_filtered[:, :3] # keep only xyz

		# transformed points in the first camera (odometry) frame
		pc_og_transformed = tf_pts_to_first_pose(
			pc_og_filtered[:, :3],   # the LiDAR points (Nx3)
			poses_abs[i],            # current pose (4x4)
			Tr_velo_to_cam,          # velodyne-to-camera transform (4x4)
			poses_abs[0],            # the reference pose (4x4)
			target_frame="velo"      # return points in the chosen frame ("cam" or "velo")
		)

		# RANSAC ground plane estimation (with outlier removal)
		lof = LocalOutlierFactor(n_neighbors=50, contamination=0.01, metric='euclidean')
		inliers = lof.fit_predict(pc_og_transformed) > 0
		filtered_3d_inliers = pc_og_transformed[inliers]

		ground_plane_coeffs = calc_ground_plane_ransac(
			filtered_3d_inliers,
			threshold=0.01,
			max_iterations=1000
		)
		if ground_plane_coeffs is None:
			print("[WARN] Could not compute ground plane!")
			continue

		ground_normal = calc_ground_plane_normal(ground_plane_coeffs)
		ground_normal = linalg.align_normal_ref_pt(
			ground_normal,
			ref_pt=np.array(cam_ref_pt),
			plane_ctr=np.array(plane_ctr_pt)
		)			
		plane_centroid = calc_ground_plane_center(filtered_3d_inliers)


		# get camera to LiDAR transformation matrix (rotation only)
		# rot_cam_to_lidar = np.linalg.inv(Tr_velo_to_cam)[:3, :3]

		rot_cam_to_lidar = get_cam_tf_mat_kitti(
				poses_abs[i],          # current camera pose (Cam-i)
				poses_abs[0],          # first camera pose   (Cam-0)
				Tr_velo_to_cam)        # LiDAR → Cam extrinsic

		# visualize the results
		vis_gt_normal.plot_complex_visualization(
			img, filtered_2d, pc_og_transformed, ground_plane_coeffs,
			ground_normal, plane_centroid, results_root, seq_num, i,
			vis, save_vis
		)			

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


@hydra.main(version_base="1.3", config_path="../configs/gt", config_name="gt_normal_kitti.yaml")
def main(cfg: DictConfig) -> None:
	"""
	Main entry point for ground truth generation.
	"""
	gt_root = os.path.join(cfg.repo_root, "results", "gt_kitti")
	os.makedirs(gt_root, exist_ok=True)    

	# sequences too big, load files on the fly (load data would come here)

	# handle ROIs
	roi_idx = 0
	if not cfg.roi:
		default_roi = [(550, 430), (730, 430), (950, 720), (265, 720)]
		print("No ROI provided. Using default ROI:", default_roi)
		roi = default_roi
	else:
		roi = list(map(tuple, cfg.roi['kitti'][roi_idx].points))


	# resize ROI for current image size
	roi_img_src_size = cfg.roi_img_src_size.kitti
	roi_img_dst_size = kitti_funcs.read_kitti_img_file(cfg.data_root, cfg.seq_num, 0).shape[:2]
	roi_img_dst_size = (roi_img_dst_size[1], roi_img_dst_size[0])
	roi = hg_funcs.calculate_scaled_roi(roi, roi_img_src_size, roi_img_dst_size)

	calc_gt_normal(
		cfg.data_root, cfg.seq_num, roi, cfg.cam_ref_pt, cfg.plane_ctr_pt,
		gt_root, 
  		tf_id=cfg.tf_id, lidar_sensor=cfg.lidar_sensor,
	 	vis=cfg.vis, save_vis=cfg.save_vis, single_frame=cfg.single_frame
	)
	print(f"Ground truth generation complete for sequence {cfg.seq_num}!")


if __name__ == "__main__":
	main()
