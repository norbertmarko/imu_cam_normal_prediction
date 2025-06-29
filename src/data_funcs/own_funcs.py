from pathlib import Path
import yaml
from typing import Optional, List

import numpy as np
import cv2
import open3d as o3d
import pandas as pd
from scipy.spatial.transform import Rotation as R


def load_calib_from_yaml(yaml_file_path):
    """
    Load calibration matrices from a YAML file.
    
    Args:
        yaml_file_path (str): Path to the YAML calibration file.
        
    Returns:
        dict: Dictionary containing calibration parameters including:
            - transforms: Dictionary of transformation matrices (e.g., Tr_lidar_to_cam)
            - intrinsics: Dictionary with matrix, focal_length, principal_point, image_size
    
    Example usage:
        ```python
        calib_data = load_calib_from_yaml('calib.yaml')
        Tr_lidar_to_cam = calib_data['transforms']['Tr_lidar_to_cam']
        Tr_lidar_to_lidar_center = calib_data['transforms']['Tr_lidar_to_lidar_center']
        intrinsic_matrix = calib_data['intrinsics']['matrix']
        ```
    """
    with open(yaml_file_path, 'r') as file:
        calib_data = yaml.safe_load(file)
    
    # Extract transforms
    transforms = {}
    if 'transforms' in calib_data:
        for transform_name, tr_data in calib_data['transforms'].items():
            if isinstance(tr_data, list):
                transform_matrix = np.array(tr_data).reshape(4, 4)
            else:
                transform_matrix = np.array([float(x) for x in tr_data.split()]).reshape(4, 4)
            transforms[transform_name] = transform_matrix
    else:
        raise KeyError("'transforms' section not found in YAML file")
    
    # extract intrinsic matrix
    if 'intrinsic_matrix' in calib_data:
        intrinsics = {}
        intr_data = calib_data['intrinsic_matrix']
        if isinstance(intr_data, list):
            intr_matrix = np.array(intr_data).reshape(4, 4)
        else:
            intr_matrix = np.array([float(x) for x in intr_data.split()]).reshape(4, 4)
        
        intrinsics['matrix'] = intr_matrix
        intrinsics['focal_length'] = [intr_matrix[0, 0], intr_matrix[1, 1]]
        intrinsics['principal_point'] = [intr_matrix[0, 2], intr_matrix[1, 2]]
    else:
        raise KeyError("'intrinsic_matrix' not found in YAML file")
    
    # extract image size
    if 'image_size' in calib_data:
        intrinsics['image_size'] = calib_data['image_size']
    else:
        raise KeyError("'image_size' not found in YAML file")
    
    return {'transforms': transforms, 'intrinsics': intrinsics}


def read_img_file(data_root, seq_num, frame_num, invert_channels=False):
    """Read a single image file and return the image data."""
    img_path = Path(data_root) / seq_num / "camera" / f"{frame_num:04d}.png"
    img_data = cv2.imread(str(img_path))
    if invert_channels:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    return img_data


def read_pcd_file(data_root, seq_num, frame_num, homogeneous=False):
    """
    Read a single point cloud file and return the point cloud data.

    Args:
        data_root (str or Path): Base directory for the dataset.
        seq_num (str or int): Sequence number.
        frame_num (int): Frame number.
        homogeneous (bool): If True, returns points in (N x 4) homogeneous format.

    Returns:
        If homogeneous == False:
            pcd_data: (N x 3) array of 3D points.
        If homogeneous == True:
            pcd_data: (N x 4) array of homogeneous 3D points.
            indices: (N,) array of original point indices.
    """
    pcd_path = Path(data_root) / seq_num / "lidar" / f"{frame_num:04d}.pcd"

    # read PCD file using Open3D
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    if len(pcd.points) == 0:
        print(f"[WARN] Empty point cloud loaded from {pcd_path}")
        if homogeneous:
            return np.empty((0, 4)), np.empty((0,), dtype=int)
        else:
            return np.empty((0, 3)), None

    points = np.asarray(pcd.points)
    if homogeneous:
        points_hom, indices = prepare_pc_points(points)
        points = points_hom
    else:
        indices = None
    return points, indices


def prepare_pc_points(pc):
    """
    Converts a point cloud (NumPy array of shape (N, 3)) into an (N x 4)
    homogeneous coordinate array. Also returns original point indices.
    
    Returns:
        pts3d_hom: A (N x 4) array of homogeneous points.
        valid_indices: An array of indices (np.arange(N))
    """
    pts = np.asarray(pc)
    N = pts.shape[0]
    pts3d_hom = np.hstack((pts, np.ones((N, 1))))  # Shape: (N, 4)
    valid_indices = np.arange(N)
    return pts3d_hom, valid_indices


def get_seq_len(data_root, seq_num):
    """Get the length of a sequence."""
    seq_path = Path(data_root) / seq_num / "camera"
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Could not calculate sequence length. Path {seq_path} does not exist."
        )
    img_files = list(seq_path.glob("*.png"))
    return len(img_files)


def projection(img, pc, camera_K, Tr_lidar_to_cam):
    """
    Projects 3D LiDAR point cloud onto the image plane.

    Args:
        img (np.ndarray): The image frame.
        pc (np.ndarray): Point cloud in either (N, 3) or (N, 4) format.
        camera_K (np.ndarray): Intrinsic matrix (3x3 or 3x4).
        Tr_lidar_to_cam (np.ndarray): 4x4 transformation matrix.

    Returns:
        projected_points (N x 2): 2D image coordinates.
        valid_pc (N x 3): Original 3D points (not transformed).
        original_indices (N,): Indices of valid points.
    """
    img_h, img_w = img.shape[:2]

    # Handle raw vs homogeneous point cloud
    if pc.shape[1] == 4:
        pc_hom = pc
        raw_pc = pc[:, :3]
    else:
        raw_pc = pc
        ones = np.ones((pc.shape[0], 1))
        pc_hom = np.hstack((pc, ones))

    # Transform to camera frame
    pc_cam_hom = (Tr_lidar_to_cam @ pc_hom.T).T  # (N x 4)
    pc_cam = pc_cam_hom[:, :3]  # (N x 3)

    # Filter points behind the camera
    Xc, Yc, Zc = pc_cam[:, 0], pc_cam[:, 1], pc_cam[:, 2]
    valid = Zc > 0
    Xc, Yc, Zc = Xc[valid], Yc[valid], Zc[valid]
    valid_pc = raw_pc[valid]

    # Intrinsics
    if camera_K.shape[1] == 4:
        camera_K = camera_K[:, :3]
    fx = camera_K[0, 0]
    fy = camera_K[1, 1]
    cx = camera_K[0, 2]
    cy = camera_K[1, 2]

    # Project to image plane
    x_proj = fx * Xc / Zc + cx
    y_proj = fy * Yc / Zc + cy

    # Round and keep within image bounds
    x_proj = np.round(x_proj).astype(int)
    y_proj = np.round(y_proj).astype(int)
    in_img = (x_proj >= 0) & (x_proj < img_w) & (y_proj >= 0) & (y_proj < img_h)

    # Final outputs
    projected_points = np.stack((x_proj[in_img], y_proj[in_img]), axis=-1)
    valid_pc = valid_pc[in_img]
    original_indices = np.where(valid)[0][in_img]

    return projected_points, valid_pc, original_indices


def visualize_filtered_projection(img, projected_pts, filtered_pts, roi):
    """Visualize the projected points on the image."""

    vis_img = img.copy()
    # draw ROI polygon on the image
    roi_np = np.array(roi, dtype=np.int32)
    print(roi_np)
    cv2.polylines(vis_img, [roi_np], isClosed=True, color=(255, 0, 0), thickness=2)

    # for pt in projected_pts:
    #     cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
    for pt in filtered_pts:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
    cv2.imshow("Filtered ROI points", vis_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_and_process_pose_data(
	data_root: str,
	seq_num: str,
	Tr_lidar_right_to_lidar_center: np.ndarray,
	Tr_camera_to_lidar_right: np.ndarray,       
	seq_len: int
) -> Optional[List[np.ndarray]]:
	"""
	Loads LIO-SAM pose (from center LiDAR), transforms it to the right LiDAR frame,
	and then to the absolute camera pose frame.

	Args:
		data_root: Root directory of the dataset.
		seq_num: Sequence number identifier.
		Tr_lidar_right_to_lidar_center: 4x4 extrinsic matrix T_lidar_right <- lidar_center.
		Tr_camera_to_lidar_right: 4x4 extrinsic matrix T_camera <- lidar_right.
		seq_len: Expected number of poses for validation.

	Returns:
		A list of 4x4 absolute camera pose matrices (T_world <- camera),
		or None if loading/processing fails.
	"""
	pose_file = Path(data_root) / seq_num / "pose" / 'liosam_upsampled.csv'

	if not pose_file.is_file():
		print(f"[ERROR] Pose file not found: {pose_file}")
		return None

	try:
		pose_data = pd.read_csv(pose_file)
	except Exception as e:
		print(f"[ERROR] Failed to load pose file {pose_file}: {e}")
		return None

	if len(pose_data) != seq_len:
		print(f"[WARNING] Mismatch: Expected {seq_len} poses, found {len(pose_data)} in {pose_file}.")
		return None # Enforce consistency

	# pre-calculate needed inverse transforms
	try:
		# T_lidar_center <- lidar_right
		Tr_lidar_center_to_right = np.linalg.inv(Tr_lidar_right_to_lidar_center)
		# T_lidar_right <- camera
		Tr_lidar_right_to_camera = np.linalg.inv(Tr_camera_to_lidar_right)
	except np.linalg.LinAlgError as e:
		print(f"[ERROR] Failed to invert calibration matrices: {e}")
		return None

	poses_abs_cam = []
	required_cols = ['pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z', 'ori_w']
	if not all(col in pose_data.columns for col in required_cols):
		print(f"[ERROR] Pose file {pose_file} missing required columns: {required_cols}")
		return None

	for index, row in pose_data.iterrows():
		# 1. Get T_world <- lidar_center from CSV row
		translation = np.array([row['pos_x'], row['pos_y'], row['pos_z']])
		quat = np.array([row['ori_x'], row['ori_y'], row['ori_z'], row['ori_w']]) # xyzw

		try:
			rotation_matrix = R.from_quat(quat).as_matrix()
		except ValueError as e:
			print(f"[ERROR] Invalid quaternion at index {index}: {quat}. Error: {e}")
			rotation_matrix = np.identity(3) # Fallback

		T_world_lidar_center = np.identity(4)
		T_world_lidar_center[:3, :3] = rotation_matrix
		T_world_lidar_center[:3, 3] = translation

		# 2. Calculate T_world <- camera using the combined formula
		# T_world_cam = (T_world_lidar_center @ T_lidar_center_to_right) @ T_lidar_right_to_camera
		T_world_camera = (T_world_lidar_center @ Tr_lidar_center_to_right) @ Tr_lidar_right_to_camera
		poses_abs_cam.append(T_world_camera)

	print(f"Successfully loaded and processed {len(poses_abs_cam)} poses via multi-sensor transforms.")
	return poses_abs_cam


def calculate_relative_poses(poses_abs_cam: List[np.ndarray]) -> List[np.ndarray]:
    """
    Calculate relative camera poses between consecutive frames given a list of absolute poses.
    
    The absolute poses should be 4x4 matrices (T_world_cam) such that the relative transform
    from frame i to frame i+1 can be computed by:
    
        T_rel = inv(T_abs[i]) @ T_abs[i+1]
    
    This transform maps coordinates from frame i to frame i+1.
    
    Args:
        poses_abs_cam (List[np.ndarray]): A list of 4x4 absolute camera pose matrices.
        
    Returns:
        List[np.ndarray]: A list of relative transform matrices (4x4 each), with one fewer element 
        than the input list.
    """
    relative_transforms = []
    
    for i in range(len(poses_abs_cam) - 1):
        # Compute the inverse of the current absolute pose
        T_inv = np.linalg.inv(poses_abs_cam[i])
        # Compute the relative transform mapping frame i to frame i+1
        T_relative = T_inv @ poses_abs_cam[i + 1]
        relative_transforms.append(T_relative)
    
    return relative_transforms


## UTILITY FUNCTIONS

def frame_is_invalid(img: np.ndarray, pc: np.ndarray, frame_idx: int) -> bool:
    """
    Returns True if the loaded image or point-cloud is missing/empty.
    """
    # 1) Image check
    if img is None or img.size == 0:
        print(f"[WARN] Frame {frame_idx}: image is None or empty")
        return True

    # 2) Point-cloud check
    #    pc should be an (N×4) array when homogeneous=True
    if pc is None or pc.size == 0:
        print(f"[WARN] Frame {frame_idx}: point-cloud is None or empty")
        return True
    return False
