import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple


class VisualOdometry(object):
    
    def __init__(self, data_dir: str, feature_extractor=cv2.ORB_create(3000), matcher=None, max_limit=None):
        """
        Initializes the Visual Odometry class with camera calibration, images, ground truth poses, feature extractor, and matcher.

        Args:
            data_dir (str): Path to the dataset directory.
            feature_extractor: The feature extractor to be used (default is ORB).
            matcher: The feature matcher to be used (default is FLANN-based matcher).
        """

        # Load data from data directory
        self.max_limit = max_limit
        self.K = self._load_calibration(data_dir,)
        self.images = self._load_images(data_dir)
        self.gt_poses = self._load_gt_poses(data_dir)
        # Define what feature extractor to use
        self.feature_extractor = feature_extractor
        # Define what matcher to use
        if matcher is None:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        else:
            self.matcher = matcher
        
    def _load_images(self, data_dir: str) -> List[np.ndarray]:
        """
        Loads grayscale images from the 'image_0' directory.

        Args:
            data_dir (str): The path to the dataset directory.

        Returns:
            List[np.ndarray]: A list of images loaded as numpy arrays.
        """
        imgs = []
        image_folder = os.path.join(data_dir, 'image_0')
        image_files = os.listdir(image_folder)
        if self.max_limit:
            image_files = image_files[:self.max_limit]
        
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(img)
        return imgs

    def _load_calibration(self, data_dir: str) -> np.ndarray:
        """
        Loads the camera calibration matrix (P0) from the 'calib.txt' file.

        Args:
            data_dir (str): The path to the dataset directory.

        Returns:
            np.ndarray: The 3x3 intrinsic camera matrix.
        """
        filepath = os.path.join(data_dir, "calib.txt")
        with open(filepath, 'r') as f:
            calibration_data = f.readlines()
        P0 = np.array([float(x) for x in calibration_data[0].strip().split()[1:]]).reshape(3, 4)
        return P0[:, :3]

    def _load_gt_poses(self, data_dir: str) -> List[np.ndarray]:
        """
        Loads the ground truth poses from the 'poses.txt' file.

        Args:
            data_dir (str): The path to the dataset directory.

        Returns:
            List[np.ndarray]: A list of 3x4 pose matrices representing the ground truth poses.
        """
        filepath = os.path.join(data_dir, "poses.txt")
        gt_poses = []
        with open(filepath, 'r') as f:
            gt_data = f.readlines()
        if self.max_limit:
            gt_data = gt_data[:self.max_limit]
        for line in gt_data:
            values = [float(x) for x in line.strip().split()]
            C = np.array(values).reshape(3, 4) 
            gt_poses.append(C)

        return gt_poses


    def _get_matches(self, frame):
        pass

    def _get_pose(self):
        pass

    def _get_gt_poses(self):
        return self.gt_poses
    
    def plot_path(self, path: List[Tuple[float, float]]):
        """
        Plots a path based on the 2D coordinates provided.

        Args:
            path (List[Tuple[float, float]]): A list of tuples where each tuple contains the (x, y) or (x, z) coordinates.
        """
        # Extract x and y or x and z coordinates
        x_vals, y_vals = zip(*path)

        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label='Path', linewidth=1)
        plt.title('Path (x, y)')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def run(self):
        estimated_path = []
        gt_path = []
        pose_progress = tqdm(vo._get_gt_poses(), unit="pose", desc="Processing ground truth poses")
        for i, gt_pose in enumerate(pose_progress):
            # if i ==0:
            #     current_R = gt_pose[:, :3]
            #     current_t = gt_pose[:, 3]
            # else:
            #     q1, q2 = self._get_matches(i)
            #     R, t = self._get_pose(q1, q2)
            #     current_R = R @ current_R
            #     current_t += current_R @ t

            gt_path.append((gt_pose[0,3], gt_pose[2,3]))
        self.plot_path(gt_path)


if __name__ == "__main__":
    data_dir = "../data/dataset/sequences/00"
    
    vo = VisualOdometry(data_dir, max_limit=1000)
    print(f"Intrinsic camera parameters is K = \n{vo.K}")
    print(f"Number of images in {data_dir}: {len(vo.images)}")
    print(f"Number of poses in {data_dir}: {len(vo.gt_poses)}")
    vo.run()


