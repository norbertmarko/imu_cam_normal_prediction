from pathlib import Path
import json

import pandas as pd
import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

# data loading

# load images in numpy format
def get_sztaki_seq_images(data_root, bag_dir):
	images = []
	
	data_dir = Path(data_root) / bag_dir / "extracted_data" / "camera"
	df = pd.read_csv(data_dir / "cam_data.csv")

	# images
	image_paths = list(data_dir.glob("*.png"))
	image_paths.sort()
	for image_path in image_paths:
		image = cv2.imread(str(image_path))
		images.append(image)
	# timestamps
	images_timestamps = df["cam_timestamps_rel"].values

	return images, images_timestamps


def get_sztaki_seq_intrinsics(data_root, bag_dir):
	data_dir = Path(data_root) / bag_dir / "extracted_data" / "camera"
	intrinsics_file = data_dir / "intrinsics.json"
	
	with open(intrinsics_file, 'r') as f:
		camera_K = json.load(f)
	
	fx = camera_K["fx"]
	fy = camera_K["fy"]
	cx = camera_K["cx"]
	cy = camera_K["cy"]

	camera_K = np.array([
		[fx, 0, cx],
		[0, fy, cy],
		[0,  0,  1]
	])

	return camera_K


def get_sztaki_seq_dist_coeffs(data_root, bag_dir):
	data_dir = Path(data_root) / bag_dir / "extracted_data" / "camera"
	intrinsics_file = data_dir / "dist_coeffs.json"
	
	with open(intrinsics_file, 'r') as f:
		camera_D = json.load(f)
	
	dist_coeffs = camera_D["dist_coeffs"]
	dist_coeffs = np.array(dist_coeffs)

	return dist_coeffs


def get_sztaki_seq_imu(data_root, bag_dir, imu_type):
	assert imu_type in ["ouster", "livox"], "Invalid IMU type."

	data_dir = Path(data_root) / bag_dir / "extracted_data"

	if imu_type == "livox":
		imu_path = data_dir / "livox_imu" / "livox_data.csv"
		df = pd.read_csv(imu_path)
		imu_values = df["livox_lin_acc_z"].values
		imu_timestamps = df["livox_timestamps_rel"].values

	elif imu_type == "ouster":
		imu_path = data_dir / "ouster_imu" / "ouster_data.csv"
		df = pd.read_csv(imu_path)
		imu_values = df["ouster_lin_acc_z"].values
		imu_timestamps = df["ouster_timestamps_rel"].values

	return imu_values, imu_timestamps


# pre-processing

def undistort_imgs(imgs, intrinsics, dist_coeffs):
	"""
	Undistorts and optionally crops a list of images.

	Args:
		imgs (list): List of distorted images.
		intrinsics (np.ndarray): Camera intrinsic matrix (3x3).
		dist_coeffs (np.ndarray): Distortion coefficients.

	Returns:
		tuple:
			- undistorted_images (list): List of undistorted (cropped) images.
			- refined_intrinsics (np.ndarray): New intrinsic matrix for cropped images.
			- roi (tuple): Region of interest (x, y, w, h).
	"""
	h, w = imgs[0].shape[:2]
	refined_intrinsics, roi = cv2.getOptimalNewCameraMatrix(
		intrinsics, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h))
	x, y, w, h = roi

	undist_imgs = []
	for img in imgs:
		undist_img = cv2.undistort(img, intrinsics, dist_coeffs, None, refined_intrinsics)
		cropped_img = undist_img[y:y+h, x:x+w]
		undist_imgs.append(cropped_img)

	return undist_imgs, refined_intrinsics, roi


def show_img_pair(img1, img2):
	combined = np.hstack((img1, img2))
	cv2.imshow("Image Pair", combined)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# augementation (part of the pre-processing pipeline)

def laplacian_filter(image, kernel_size=3, center_coeff=5, **kwargs):
	"""
	Applies a Laplacian filter for edge enhancement.
	
	Args:
		image (np.ndarray): Input image.
		kernel_size (int): Size of the Laplacian kernel. Must be 3 or 5.
		center_coeff (int): Center coefficient in the Laplacian kernel.
	
	Returns:
		np.ndarray: Laplacian-filtered image.
	"""
	if kernel_size not in [3, 5]:
		raise ValueError("Kernel size must be 3 or 5.")

	laplacian_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
	laplacian_kernel[kernel_size // 2, :] = -1
	laplacian_kernel[:, kernel_size // 2] = -1
	laplacian_kernel[kernel_size // 2, kernel_size // 2] = center_coeff

	laplacian_filtered_image = cv2.filter2D(image, -1, laplacian_kernel)
	return laplacian_filtered_image


class LaplacianFilter(ImageOnlyTransform):
	def __init__(self, always_apply=False, p=1.0):
		super(LaplacianFilter, self).__init__(always_apply, p)
	
	def apply(self, image, **params):
		# apply custom augmentation logic
		return laplacian_filter(image)


def hist_equalization(image, adaptive=False, clip_limit=2.0, tile_grid_size=(8, 8), **kwargs):
	"""
	Enhances the contrast of an image using histogram equalization.
	
	Args:
		image (np.ndarray): Input image.
		adaptive (bool): Whether to use adaptive histogram equalization (CLAHE).
		clip_limit (float): Clip limit for CLAHE (ignored if adaptive=False).
		tile_grid_size (tuple): Tile grid size for CLAHE (ignored if adaptive=False).
	
	Returns:
		np.ndarray: Contrast-enhanced image.
	"""
	if adaptive:
		# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
		clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
		ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
		y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
		y_channel_eq = clahe.apply(y_channel)
		equalized_ycrcb = cv2.merge((y_channel_eq, cr_channel, cb_channel))
		equalized_image = cv2.cvtColor(equalized_ycrcb, cv2.COLOR_YCrCb2BGR)
	else:
		# Standard histogram equalization
		equalized_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
		y_channel, cr_channel, cb_channel = cv2.split(equalized_image)
		y_channel_eq = cv2.equalizeHist(y_channel)
		equalized_ycrcb = cv2.merge((y_channel_eq, cr_channel, cb_channel))
		equalized_image = cv2.cvtColor(equalized_ycrcb, cv2.COLOR_YCrCb2BGR)

	return equalized_image


def gaussian_denoising(image, kernel_size=(5, 5), sigma=1.0):
	"""
	Reduces noise in the image using a Gaussian filter.
	
	Args:
		image (np.ndarray): Input image.
		kernel_size (tuple): Kernel size for the Gaussian filter.
		sigma (float): Standard deviation for the Gaussian kernel.
	
	Returns:
		np.ndarray: Denoised image.
	"""
	return cv2.GaussianBlur(image, kernel_size, sigma)


def bilateral_filter(image, diameter=9, sigma_color=75, sigma_space=75):
	"""
	Applies a bilateral filter to smooth the image while preserving edges.
	
	Args:
		image (np.ndarray): Input image.
		diameter (int): Diameter of each pixel neighborhood.
		sigma_color (float): Filter sigma in color space.
		sigma_space (float): Filter sigma in coordinate space.
	
	Returns:
		np.ndarray: Edge-preserving smoothed image.
	"""
	return cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)


def sharpening_filter(image, alpha=1.5, beta=-0.5, gamma=0):
    """
    Enhances image details using a sharpening kernel.

    Args:
        image (np.ndarray): Input image.
        alpha (float): Weight for the original image.
        beta (float): Weight for the Laplacian-filtered image.
        gamma (float): Scalar added to the output.

    Returns:
        np.ndarray: Sharpened image.
    """
    # Compute Laplacian of the image (64F to handle negatives)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Convert Laplacian to the same type as the input image
    laplacian = cv2.convertScaleAbs(laplacian)

    # Add weighted sum of the original image and Laplacian
    sharpened = cv2.addWeighted(image, alpha, laplacian, beta, gamma)

    return sharpened

# used in LoFTR 
def gamma_correction(image: np.ndarray, gamma: float = 1.2):
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def guided_filter(image: np.ndarray, radius: int = 8, eps: float = 1e-3):
    # needs OpenCV-contrib (> 3.4).  If not available just return the input.
    if not hasattr(cv2, "ximgproc"):
        return image
    return cv2.ximgproc.guidedFilter(image, image, radius, eps)


def unsharp_mask(image: np.ndarray, blur_sigma: float = 3, amount: float = 1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


# processing pipeline
def proc_for_loftr(img: np.ndarray) -> np.ndarray:
    """
    CLAHE (Y channel) → gamma stretch → guided-filter denoise → un-sharp mask
    """
    img = hist_equalization(img, adaptive=True)
    img = gamma_correction(img, gamma=1.2)
    img = guided_filter(img, radius=8, eps=1e-3)
    img = unsharp_mask(img, blur_sigma=3, amount=0.5)
    return img


# Example usage
if __name__ == "__main__":
	# Intrinsics and distortion coefficients
	K = np.array([[751.855595111958, 0, 651.34945122955276],
				  [0, 763.09679378498311, 496.31490467892837],
				  [0, 0, 1]])
	D = np.array([-0.30661727903873737, 0.077168182137361443, 
				  0.00021123986905573208, -0.0027659136961402531, 0])
	
	# Path to your image
	image_path = "/mnt/e/kitti/dataset/sequences/01/image_2/000000.png"
	roi = [(460, 515), (640, 515), (750, 690), (220, 690)]
	roi_polygon = np.array(roi, dtype=np.int32)

	# Load image
	imgs = []
	img = cv2.imread(image_path)
	imgs.append(img)

	# Show distorted vs undistorted
	# undist_imgs, refined_intrinsics, roi = undistort_imgs(imgs, K, D)
	# print("ROI:", roi)
	# imgs_resized = [cv2.resize(img, roi[2:]) for img in imgs]

	test_img = imgs[0]
	test_img = hist_equalization(test_img)
	test_img = LaplacianFilter(always_apply=True)(image=test_img)["image"]
	# test_img = gaussian_denoising(test_img)
	test_img = bilateral_filter(test_img, diameter=7, sigma_color=25, sigma_space=25)
	# test_img = sharpening_filter(test_img)

	cv2.polylines(test_img, [roi_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
	cv2.imshow("Undistorted Image", test_img)
	cv2.waitKey(0)

	# show_img_pair(imgs_resized[0], imgs[0])
