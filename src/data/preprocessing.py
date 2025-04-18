"""
Image preprocessing utilities for MobileNeRF Edge.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class ImagePreprocessor:
    """
    Handles preprocessing of images for MobileNeRF training and inference.
    """

    def __init__(self, 
                 target_size: Tuple[int, int] = (256, 256),
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target resolution (height, width) for preprocessing
            camera_matrix: Camera intrinsic matrix (3x3) if available
            dist_coeffs: Distortion coefficients if available
        """
        self.target_size = target_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Remove lens distortion from the image.
        
        Args:
            image: Input image
            
        Returns:
            Undistorted image
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        
        undistorted = cv2.undistort(
            image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        
        # Crop the image to the ROI
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]
            
        return undistorted
    
    def normalize_exposure(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image exposure.
        
        Args:
            image: Input image
            
        Returns:
            Exposure normalized image
        """
        # Convert to YUV color space
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # Apply histogram equalization to the Y channel
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        
        # Convert back to BGR color space
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            
        Returns:
            Resized image
        """
        return cv2.resize(image, (self.target_size[1], self.target_size[0]), 
                          interpolation=cv2.INTER_AREA)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all preprocessing steps to an image.
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        # Apply undistortion if camera parameters are available
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            image = self.undistort(image)
        
        # Normalize exposure
        image = self.normalize_exposure(image)
        
        # Resize to target size
        image = self.resize(image)
        
        return image
    
    def batch_process(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Process a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of processed images
        """
        return [self.process(img) for img in images]


def calibrate_camera(calibration_images: List[np.ndarray], 
                     pattern_size: Tuple[int, int],
                     square_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrate camera using chessboard pattern.
    
    Args:
        calibration_images: List of chessboard images
        pattern_size: Inner corners of the chessboard pattern (width, height)
        square_size: Size of chessboard square in your preferred unit
        
    Returns:
        Camera matrix and distortion coefficients
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    for img in calibration_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return camera_matrix, dist_coeffs


def extract_camera_poses(images: List[np.ndarray], 
                         camera_matrix: np.ndarray,
                         dist_coeffs: np.ndarray,
                         pattern_size: Tuple[int, int],
                         square_size: float = 1.0) -> List[Dict[str, np.ndarray]]:
    """
    Extract camera poses from calibration images.
    
    Args:
        images: List of images with visible calibration pattern
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        pattern_size: Inner corners of the pattern (width, height)
        square_size: Size of pattern square
        
    Returns:
        List of camera poses (rotation, translation) for each image
    """
    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    
    poses = []
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the pattern corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Find the rotation and translation vectors
            ret, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            poses.append({
                'rotation': rotation_matrix,
                'translation': tvec,
                'image_size': gray.shape[::-1]
            })
    
    return poses
