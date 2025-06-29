import os
from os.path import join, basename, splitext
import yaml

import numpy as np
import cv2


def read_kitti_calib(calib_path):
    """Read kitti calibration file and get camera intrinsic matrix"""
    with open(calib_path, "r") as f:
        lines = f.readlines()
        p2 = lines[2].split()[1:]
        p2 = np.array(p2, dtype=np.float32).reshape(3, 4)
    return p2[:, :3]


def add_image_ids(input_file, output_file):
    """Add image IDs (line numbers) at the start of each line of a pose file"""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for idx, line in enumerate(infile, start=1):
            outfile.write(f"{idx+1} {line}")


def clean_kitti_pose_file(pose_file, kitti_pose_root):
    """Check if pose file has 12 values per line. If not, add image IDs"""
    with open(pose_file, 'r') as f:
        first_line = f.readline().strip().split()
    if len(first_line) == 12:
        print("Pose file has 12 values per line. Adding image IDs...")
        # Create a new filename by appending "_with_ids" to the original file name
        corrected_pose_file = join(
            kitti_pose_root, f"{splitext(basename(pose_file))[0]}_with_ids.txt"
        )
        add_image_ids(pose_file, corrected_pose_file)
        pose_file = corrected_pose_file  # Use the corrected file
    return pose_file


def invT(transform):
    """Inverse a transform matrix without using np.linalg.inv"""
    R_Transposed = transform[:3, :3].T
    result = np.eye(4)
    result[:3, :3] = R_Transposed
    result[:3, 3] = -R_Transposed @ transform[:3, 3]
    return result


def read_kitti_pose(pose_path):
    """Read kitti pose file and get relative transform
    Args:
        pose_path (string): path to pose file
    Returns:
        ndarray: relative transforms with shape=(N,4,4)
    """
    input_pose = np.loadtxt(pose_path)
    assert (input_pose.shape[1] == 13)
    image_ids = input_pose[:, 0].astype(np.int32)
    input_pose = input_pose[:, 1:]
    length = input_pose.shape[0]
    input_pose = input_pose.reshape(-1, 3, 4)
    bottom = np.zeros((length, 1, 4))
    bottom[:, :, -1] = 1
    absolute_transform = np.concatenate((input_pose, bottom), axis=1)
    relative_transforms = []
    for idx in range(length - 1):
        relative_transform = invT(
            absolute_transform[idx + 1]) @ absolute_transform[idx]
        relative_transforms.append(relative_transform)
    return image_ids, relative_transforms


def read_kitti_pose_abs(pose_path):
    """Read kitti pose file and get absolute transforms
    Args:
        pose_path (string): path to pose file
    Returns:
        tuple: image_ids and absolute transforms with shape=(N,4,4)
    """
    input_pose = np.loadtxt(pose_path)
    assert (input_pose.shape[1] == 13)
    image_ids = input_pose[:, 0].astype(np.int32)
    input_pose = input_pose[:, 1:]
    length = input_pose.shape[0]
    input_pose = input_pose.reshape(-1, 3, 4)
    bottom = np.zeros((length, 1, 4))
    bottom[:, :, -1] = 1
    absolute_transforms = np.concatenate((input_pose, bottom), axis=1)
    return image_ids, absolute_transforms


### --- GT generation --- ###


def get_kitti_cam_seq_len(kitti_root, seq_num, cam_num=2):
    img_dir = join(kitti_root, seq_num, f"image_{cam_num}")
    img_file_paths = sorted(os.listdir(img_dir))
    return len(img_file_paths)


def read_kitti_pc_file(kitti_root, seq_num, frame_num):
    """Read a single KITTI point cloud file and return the point cloud data (x, y, z)."""
    pc_path = join(kitti_root, seq_num, "velodyne", f"{frame_num:06d}.bin")
    pc_data = np.fromfile(pc_path, '<f4')   # little-endian float32
    pc_data = np.reshape(pc_data, (-1, 4))  # x, y, z, r
    # pc_data = pc_data[:, :3]  # x, y, z (exclude reflectance / luminance)
    return pc_data


def read_kitti_img_file(kitti_root, seq_num, frame_num, invert_channels=False):
    """Read a single KITTI image file and return the image data."""
    img_path = join(kitti_root, seq_num, "image_2", f"{frame_num:06d}.png")
    img_data = cv2.imread(img_path)
    if invert_channels:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    return img_data


def get_kitti_seq_imgs(kitti_root, seq_num, seq_len):
    imgs = []
    for i in range(seq_len):
        img = read_kitti_img_file(kitti_root, seq_num, i)
        imgs.append(img)
    return imgs


def prepare_pc_points(pc_raw):
    '''
    Prepares the point cloud by replacing the reflectance value with 1
    and transposing the array for matrix multiplication so it can be 
    directly multiplied by the camera projection matrix.
    
    Args:
        pc_raw (numpy.ndarray): Raw point cloud data of shape (N, 4).
    
    Returns:
        pts3d (numpy.ndarray): Transposed point cloud data of shape (4, M).
        valid_indices (numpy.ndarray): Array of original indices of valid points.
    '''
    # Identify valid points (reflectance > 0)
    valid_mask = pc_raw[:, 3] > 0
    valid_indices = np.where(valid_mask)[0]
    
    # Keep only valid XYZ points
    pts3d = pc_raw[valid_mask, :3]
    
    # Add homogeneous coordinate
    pts3d = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    
    return pts3d.T, valid_indices


def load_calib(kitti_root, seq_num, cam_num=2):
    """
    Load KITTI calibration matrices required for LiDAR to image projection.

    Args:
        kitti_root (str): Path to the KITTI dataset root.
        seq_num (str): Sequence number (e.g., '0001').
        cam_num (int): Camera number (0-3).

    Returns:
        P (numpy.ndarray): 3x4 projection matrix for the specified camera.
        Tr_velo_to_cam (numpy.ndarray): 4x4 transformation from Velodyne to Camera.
        R_rect (numpy.ndarray): 4x4 rectification matrix (identity if not provided).
    """
    calib_file = os.path.join(kitti_root, seq_num, "calib.txt")
    
    with open(calib_file, "r") as f:
        lines = f.readlines()
    
    # parse projection matrices P0 to P3
    P = {}
    for i in range(4):
        key = f"P{i}"
        values = np.array(lines[i].split()[1:], dtype=np.float32)
        P[key] = values.reshape(3, 4)
    
    # parse the transformation matrix Tr_velo_to_cam
    # assuming 'Tr' is the 5th line (index 4)
    # ensure that 'Tr' is correctly identified
    Tr_line = None
    for line in lines:
        if line.startswith("Tr:"):
            Tr_line = line
            break
    
    if Tr_line is None:
        raise ValueError("Transformation matrix 'Tr' not found in calibration file.")
    
    Tr_values = np.array(Tr_line.split()[1:], dtype=np.float32).reshape(3, 4)
    Tr_velo_to_cam = np.eye(4, dtype=np.float32)
    Tr_velo_to_cam[:3, :4] = Tr_values
    
    # since R0_rect is not provided, assume it to be identity (can be changed if needed)
    R_rect = np.eye(4, dtype=np.float32)
    
    return P[f"P{cam_num}"], Tr_velo_to_cam, R_rect


def projection(img, pc, kitti_root, seq_num, cam_num=2):
    """
    Aligns the point cloud with the image by projecting 3D points onto the 2D image plane.
    
    Args:
        img (numpy.ndarray): The image frame.
        pc (numpy.ndarray): The raw point cloud data.
        kitti_root (str): Path to the KITTI dataset root.
        seq_num (str): Sequence number.
        cam_num (int): Camera number (default is 2).
    
    Returns:
        projected_points (numpy.ndarray): 2D projected points on the image plane.
        valid_pc (numpy.ndarray): Corresponding 3D points in the point cloud.
        original_indices (numpy.ndarray): Indices in the original point cloud.
    """
    P, Tr_velo_to_cam, R_rect = load_calib(kitti_root, seq_num, cam_num)
    
    # Prepare point cloud: replace reflectance with 1, keep valid points
    pts3d_hom, valid_indices = prepare_pc_points(pc)
    
    # Full transformation chain: 3D LiDAR -> Camera
    pts_cam = R_rect @ Tr_velo_to_cam @ pts3d_hom  # Result is 4xM
    
    np.set_printoptions(precision=3, suppress=True)
    # print(f"Velo to cam transform:\n{Tr_velo_to_cam}")
    
    # Discard homogeneous coordinate
    pts_cam = pts_cam[:3, :]  # 3xM
    
    # Filter out points behind the camera
    valid_depth = pts_cam[2, :] > 0
    pts_cam = pts_cam[:, valid_depth]
    valid_pc = pc[valid_indices][valid_depth]
    original_indices = valid_indices[valid_depth]
    
    # Project to 2D image plane using camera projection matrix P (3x4)
    pts_cam_hom = np.vstack((pts_cam, np.ones((1, pts_cam.shape[1]))))  # 4xM
    pts_2d = P @ pts_cam_hom  # 3xM
    
    # Normalize homogeneous coordinates
    pts_2d /= pts_2d[2, :]
    
    # Get image dimensions
    img_h, img_w = img.shape[:2]
    
    # Remove points that are outside the image frame
    valid_x = (pts_2d[0, :] >= 0) & (pts_2d[0, :] < img_w)
    valid_y = (pts_2d[1, :] >= 0) & (pts_2d[1, :] < img_h)
    valid_idx = valid_x & valid_y
    projected_points = pts_2d[:2, valid_idx].T  # Shape: (K, 2)
    
    # Filter the point cloud accordingly
    valid_pc = valid_pc[valid_idx]
    original_indices = original_indices[valid_idx]
    
    return projected_points, valid_pc, original_indices


def read_kitti_calib_from_yaml(yaml_file, matrix_name):
	"""
	Reads a KITTI calibration matrix from a YAML file and parses it into a 4x4 NumPy array.

	Args:
		yaml_file (str): Path to the YAML calibration file.
		matrix_name (str): The name of the transformation matrix to retrieve (e.g., 'Tr_velo_to_imu').

	Returns:
		np.ndarray: A 4x4 transformation matrix.
	"""
	with open(yaml_file, "r") as f:
		calib_data = yaml.safe_load(f)

	if matrix_name not in calib_data:
		raise ValueError(f"Matrix {matrix_name} not found in the YAML file.")

	values = np.array(calib_data[matrix_name], dtype=np.float32)
	return values.reshape(4, 4)


### --- Process Pose Files --- ###


def add_image_ids(input_file, output_file):
    """Add image IDs (line numbers) at the start of each line of a pose file."""
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for idx, line in enumerate(infile, start=1):
            outfile.write(f"{idx} {line}")

def clean_kitti_pose_file(pose_file, seq_num):
    """Check if pose file has 12 values per line. If not, add image IDs."""
    with open(pose_file, 'r') as f:
        first_line = f.readline().strip().split()
    
    if len(first_line) == 12:
        print(f"Processing {pose_file}: Adding image IDs...")
        corrected_pose_file = os.path.join(os.path.dirname(pose_file), f"{seq_num:02d}_with_ids.txt")
        add_image_ids(pose_file, corrected_pose_file)
    else:
        print(f"Skipping {pose_file}: Incorrect format.")

def process_pose_files(root_path):
    """Cycle through pose files (00 to 10), check for {seq_num}_with_ids.txt, and create if missing."""
    for i in range(11):  # subfolders named 00 to 21
        file_name = f"{i:02d}.txt"
        file_path = os.path.join(root_path, file_name)
        
        if not os.path.isfile(file_path):
            print(f"Skipping {file_path}: Not a valid pose file.")
            continue
        
        output_file = os.path.join(root_path, f"{i:02d}_with_ids.txt")
        
        if os.path.exists(output_file):
            # print(f"Skipping {folder_name}: poses_with_ids.txt already exists.")
            continue
        
        if os.path.exists(file_path):
            clean_kitti_pose_file(file_path, i)
        else:
            print(f"Skipping {file_name}: pose file not found.")


### --- GT Visualization Checks --- ###


def visualize_projection(img, projected_pts):
    """Visualize the projected points on the image."""
    for pt in projected_pts:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
    cv2.imshow("Projected Points", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_filtered_projection(img, projected_pts, filtered_pts, roi):
    """Visualize the projected points on the image."""

    vis_img = img.copy()
    # Draw ROI polygon on the image
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


### --- Extra eval functions --- ###

def clean_kitti_pose_file_updated(pose_file):
    """Check if pose file has 12 values per line. If not, add image IDs"""
    if not os.path.exists(pose_file):
        raise FileNotFoundError(f"Pose file not found: {pose_file}")

    with open(pose_file, 'r') as f:
        first_line = f.readline().strip().split()
    
    if len(first_line) == 12:
        print(f"Processing {pose_file}: Adding image IDs...")
        corrected_pose_file = os.path.join(os.path.dirname(pose_file), "poses_with_ids.txt")
        add_image_ids(pose_file, corrected_pose_file)
        return corrected_pose_file
    else:
        print(f"Skipping {pose_file}: Incorrect format.")
        return pose_file  # Return the original file if already formatted correctly
