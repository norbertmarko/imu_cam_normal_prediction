import pickle
import os
import yaml
import warnings

import numpy as np

def save_pickle_data(data, save_path):
    """
    Saves data to disk in pickle format.

    :param data: Dictionary containing data.
    :param save_path: Path to save the pickle file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Data saved to {save_path}")


def load_pickle_data(file_path):
    """
    Loads data from a pickle file.

    :param file_path: Path to the pickle file.
    :return: Dictionary containing data.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Data loaded from {file_path}")
    return data


def read_calibration_from_yaml(file_path):
    """
    Reads calibration data from a YAML file.
    :param file_path: Path to the YAML file.
    :return: Tuple of rotation (dict) and translation (dict).
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    rotation = data['camera_to_lidar']['rotation']
    translation = data['camera_to_lidar']['translation']
    return rotation, translation


def extract_gt_data(gt_data, frame_idx, roi_idx=0):
    """
    Extract ground truth data for a given frame and ROI index.
    This function acts as a standard, every dataset ground truth data
    should be saved in this format.

    Parameters:
        gt_data (list or dict): Ground truth data loaded from file.
        frame_idx (int): Index of the frame.
        roi_idx (int): Index of the ROI (default is 0).

    Returns:
        dict or None: Dictionary containing the extracted ground truth fields,
                      or None if data is missing.
    """
    # safely check for frame existence
    if frame_idx not in gt_data:
        warnings.warn(f"[extract_gt_data] No GT data found for frame {frame_idx}", UserWarning)
        return None
    frame_gt = gt_data[frame_idx]

    # safely check for ROI index existence
    if roi_idx not in frame_gt:
        warnings.warn(f"[extract_gt_data] No GT data for ROI {roi_idx} in frame {frame_idx}", UserWarning)
        return None

    frame_data = frame_gt[roi_idx]
    return {
        "proj_pts_2d": np.array(frame_data["proj_pts_2d"]),
        "cam_pts_3d": np.array(frame_data["cam_pts_3d"]),
        "plane_coeffs": np.array(frame_data["plane_coefficients"]),
        "normal": np.array(frame_data["normal"]),
        "centroid": np.array(frame_data["centroid"]),
        "ref_frame": frame_data["ref_frame"],
        "R_cam_to_lidar": np.array(frame_data["rot_cam_to_lidar"])
    }