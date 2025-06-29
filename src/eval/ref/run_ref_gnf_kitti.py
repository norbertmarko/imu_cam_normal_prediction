import argparse
import os
import shutil
from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2
from omegaconf import DictConfig
import hydra

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pandaset import geometry
import src.algos.homography.hg_vis as hg_vis
import src.utils.data_utils as data_utils
import src.utils.linalg as linalg
import src.algos.homography.hg_funcs as hg_funcs
from scipy.spatial.transform import Rotation as R

from src.data_funcs import kitti_funcs

from src._ref.ground_normal_filter.src.filter import GroundNormalFilterIEKF
from src._ref.ground_normal_filter.src.visualizer import Visualization
import src.eval.eval_norm as eval_norm


# TODO: import from linalg.py later
def rotate_vector(vec, R):
	"""Rotate a vector using a rotation matrix R, then normalize it."""
	vec_tf = R @ vec
	vec_tf /= np.linalg.norm(vec_tf)
	return vec_tf


@hydra.main(version_base="1.3", config_path="../../configs/ref", config_name="ref_gnf_kitti.yaml")
def eval(cfg: DictConfig):

	data_root = os.path.dirname(cfg.data_root)

	kitti_pose_file = os.path.join(data_root, "poses", f"{cfg.seq_num}.txt")
	kitti_calib_path = os.path.join(data_root, cfg.seq_num, "calib.txt")
	
	# params
	kitti_d = None  # distortion coefficients (not required)
	kitti_wh = (1241, 376)
	
	# create output directories
	os.makedirs(cfg.output_dir, exist_ok=True)
	results_dir = os.path.join(cfg.output_dir, cfg.seq_num)
	os.makedirs(results_dir, exist_ok=True)
	og_vis_dir = os.path.join(results_dir, "og_vis")
	os.makedirs(og_vis_dir, exist_ok=True)
	eval_output_path = os.path.join(results_dir, f"eval_data_{cfg.seq_num}.pkl")


	cleaned_pose = os.path.join(data_root, "poses", f"{cfg.seq_num}_with_ids.txt")
	image_ids, relative_transforms = kitti_funcs.read_kitti_pose(cleaned_pose)

	camera_K = kitti_funcs.read_kitti_calib(os.path.join(data_root, "sequences", cfg.seq_num, "calib.txt"))
	seq_len = kitti_funcs.get_kitti_cam_seq_len(os.path.join(data_root, "sequences"), cfg.seq_num)

	print(
		f"Camera intrinsic matrix for PandaSet sequence {cfg.seq_num}:\n"
		f"{np.array2string(camera_K, formatter={'float_kind':lambda x: f'{x:.4f}'})}"
	)

	# import ground truth normal vectors
	gt_dir = os.path.join(os.path.dirname(cfg.output_dir), "gt_kitti", f"{cfg.seq_num}", "gt")
	gt_path = os.path.join(gt_dir, f"gt_data_{cfg.seq_num}.pkl")
	gt_data = data_utils.load_pickle_data(gt_path)


	# initialize objects
	pitch_vals = []
	pitch_vals_gt = []
	data_store = {}
	eval_data_store = {}


	gnf = GroundNormalFilterIEKF()
	vis = Visualization(K=camera_K, d=kitti_d, input_wh=kitti_wh)
	
	# main evaluation loop
	loop_length = len(relative_transforms)
	with tqdm(total=loop_length, desc="Processing Images", unit="img") as pbar:
		
		for idx in range(loop_length):

			# relative_so3 = rel_rot_mats[idx]
			relative_so3 = relative_transforms[idx][:3, :3]
			# print(relative_so3)
			# break
			compensation_se3 = gnf.update(relative_so3)
			compensation_so3 = compensation_se3[:3, :3]


			np.set_printoptions(precision=3, suppress=True)


			# return (extract) ground normal vector
			ground_normal_vec = compensation_se3[:3, 1]


			# access gt data (current frame)
			proj_pts_2d_gt = np.array(gt_data[idx][0]["proj_pts_2d"])
			cam_pts_3d_gt = np.array(gt_data[idx][0]["cam_pts_3d"])
			plane_coeffs_gt = np.array(gt_data[idx][0]["plane_coefficients"])
			normal_gt = np.array(gt_data[idx][0]["normal"])
			centroid_gt = np.array(gt_data[idx][0]["centroid"])
			ref_frame = gt_data[idx][0]["ref_frame"]
			R_cam_to_lidar = np.array(gt_data[idx][0]["rot_cam_to_lidar"])

			normal = rotate_vector(ground_normal_vec, R_cam_to_lidar)
			# postprocess normal vector
			normal = linalg.align_normal_ref_pt(
	   			normal,
				np.array([0, 0, 1]),
				np.array([0, 0, 0])
		  	)
			pitch_from_norm = hg_funcs.calculate_pitch_from_normal(normal)
			pitch_from_norm_gt = hg_funcs.calculate_pitch_from_normal(normal_gt)
   
			# store results
			pitch_vals.append(pitch_from_norm)
			pitch_vals_gt.append(pitch_from_norm_gt)


			# dummy data (calc or not needed)
			dummy_hg_mat = np.eye(3)

			# evaluation for a single example
			signed_norm_angle_deg, signed_pitch_error_deg, \
				pitch_error_deg, norm_angle_deg, reproj_err_px = eval_norm.eval_example(
				normal, normal_gt,
				pitch_from_norm, pitch_from_norm_gt,
				dummy_hg_mat, None, None, None, 
				idx, print_results=True
			)

			data_store[idx] = {
				"homography_matrix": dummy_hg_mat,
				"normal": normal,
				"normal_gt": normal_gt,
				"pitch": pitch_from_norm,
				"pitch_gt": pitch_from_norm_gt,
				"signed_norm_angle_deg": signed_norm_angle_deg,
				"pitch_error_deg": pitch_error_deg,
				"signed_pitch_error_deg": signed_pitch_error_deg,
				"mean_reproj_err_px": reproj_err_px
			}


			# store evaluation results
			eval_data_store[idx] = {
				"normal": normal,
				"normal_gt": normal_gt,
				"pitch": pitch_from_norm,
				"pitch_gt": pitch_from_norm_gt,
				"norm_angle_deg": norm_angle_deg,
				"signed_norm_angle_deg": signed_norm_angle_deg,
				"pitch_error_deg": pitch_error_deg,
				"signed_pitch_error_deg": signed_pitch_error_deg,
				"mean_reproj_err_px": reproj_err_px
			}

			image = kitti_funcs.read_kitti_img_file(os.path.join(data_root, "sequences"), cfg.seq_num, idx)
			combined_image = vis.get_frame(image, compensation_so3)
			output_path = os.path.join(og_vis_dir, f"{idx:06d}_{cfg.seq_num}.jpg")
			cv2.imwrite(output_path, combined_image)


			# comparison plotting
			hg_vis.plot_complex_normal(
				image, dummy_hg_mat, normal,
				cam_pts_3d_gt, plane_coeffs_gt, centroid_gt, normal_gt,
				signed_norm_angle_deg,
				cfg.output_dir, cfg.seq_num, idx, False, cfg.save_vis,
				normal_candidates=None
			)

			hg_vis.plot_complex_pitch(
				image, dummy_hg_mat, pitch_vals,
				pitch_vals_gt, signed_norm_angle_deg,
				cfg.output_dir, cfg.seq_num, idx, False, cfg.save_vis
			)

			pbar.update(1)

			# if idx == 149:
			# 	break

	timestamp = datetime.now().strftime("%y-%m-%d-%H-%M-%S")
	eval_norm.eval_sequence(data_store, results_dir, cfg.seq_num, timestamp, print_results=True)

	data_utils.save_pickle_data(eval_data_store, eval_output_path)


if __name__ == '__main__':
	eval()
