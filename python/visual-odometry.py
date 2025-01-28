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
        image_files = sorted(os.listdir(image_folder))
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


    def _get_matches(self, frame, plot=False):
        image1 = self.images[frame-1]
        image2 = self.images[frame]

        kp1, desc1 = self.feature_extractor.detectAndCompute(image1, None)
        kp2, desc2 = self.feature_extractor.detectAndCompute(image2, None)

        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        if not matches:
            print(f"No matches found between frame {frame-1} and frame {frame}")
            return None, None
        
        # Apply Lowe's ratio test to find good matches (Keep matches with distance ratio < distance ratio threshold)
        good = []
        for m,n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)


        filtered_kp1 = np.array([kp1[m.queryIdx].pt for m in good]) 
        filtered_kp2 = np.array([kp2[m.trainIdx].pt for m in good])
        if plot:
            match_img = cv2.drawMatches(image1, kp1, image2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv2.imshow(f"Good Matches (Frame {frame - 1} -> Frame {frame})", match_img)
            cv2.waitKey(0) 
            cv2.destroyAllWindows()

        return filtered_kp1, filtered_kp2

    def _get_pose(self, kp1: np.ndarray, kp2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the relative pose (R, t) between two frames using the essential matrix.

        Args:
            kp1 (np.ndarray): Keypoints from the first frame (shape: Nx2).
            kp2 (np.ndarray): Keypoints from the second frame (shape: Nx2).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Rotation matrix (R) and translation vector (t).
        """
        E, mask = cv2.findEssentialMat(kp1, kp2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or mask.sum() < 5: 
            raise ValueError("Essential matrix could not be computed or insufficient inliers.")
        _, R, t, _ = cv2.recoverPose(E, kp1, kp2, self.K, mask=mask)
        return R, t.squeeze()

    def _get_gt_poses(self):
        return self.gt_poses
    
    def plot_paths(self, paths: List[Tuple[str, List[Tuple[float, float]]]]):
        """
        Plots multiple paths on the same plot.

        Args:
            paths (List[Tuple[str, List[Tuple[float, float]]]]): A list of tuples where each tuple contains:
                - A string representing the name of the path (label).
                - A list of (x, z) or (x, y) coordinates representing the path.
        """
        plt.figure(figsize=(10, 6))

        for label, path in paths:
            x, z = zip(*path)
            plt.plot(x, z, marker='o', linestyle='-', label=label, linewidth=1)

        # Customize the plot
        plt.title('Paths Comparison')
        plt.xlabel('X (meters)')
        plt.ylabel('Z (meters)')
        plt.grid(True)
        plt.legend()
        plt.show()

    def _calculate_relative_scale(self, R, t, R2, t2, q1, q2):
        """
        Calculate the relative scale between two camera views based on triangulated 3D points.

        Parameters
        ----------
        R (ndarray): Rotation matrix between the two frames (3x3).
        t (ndarray): Translation vector between the two frames (3,).
        q1 (ndarray): Keypoints in the first frame (Nx2).
        q2 (ndarray): Keypoints in the second frame (Nx2).

        Returns
        -------
        relative_scale (float): The calculated relative scale between the two frames.
        """
        T = np.hstack((R, t.reshape(3, 1)))
        
        P1 = np.matmul(self.K, np.hstack((R, t.reshape(3, 1))))
        P2 = np.matmul(self.K, np.hstack((R2, t2.reshape(3, 1))))

        hom_Q1 = cv2.triangulatePoints(P1, P2, q1.T, q2.T)
        
        uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        uhom_Q2 = (T @ hom_Q1)[:3, :] / hom_Q1[3, :]
        valid_mask = ~np.isnan(uhom_Q1).any(axis=0) & ~np.isnan(uhom_Q2).any(axis=0)
        if valid_mask.sum() < 2:
            raise ValueError("Insufficient valid triangulated points for scale calculation.")

        relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1) / 
                                np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))

        return relative_scale
            
    def run(self):
        estimated_path = []
        gt_path = []
        pose_progress = tqdm(self._get_gt_poses(), unit="pose", desc="Processing ground truth poses")
        for i, gt_pose in enumerate(pose_progress):
            if i == 0:
                current_R = gt_pose[:, :3]
                current_t = gt_pose[:, 3]
            else:
                q1, q2 = self._get_matches(i, plot=False)
                R, t = self._get_pose(q1, q2)
                scale = self._calculate_relative_scale(current_R, current_t, R, t, q1, q2)
                # scale = 1
                t *= -scale
                R = R.T
                current_R = R @ current_R
                current_t += current_R @ t

            # DEBUG SECTION
            # U, _, Vt = np.linalg.svd(current_R)
            # current_R_normalized = U @ Vt

            # # Normalizing the translation vectors
            # current_t_normalized = current_t / np.linalg.norm(current_t)
            # gt_t_normalized = gt_pose[:3, 3] / np.linalg.norm(gt_pose[:3, 3])

            # # Rotation comparison (angle error)
            # rotation_error = np.arccos((np.trace(current_R_normalized.T @ gt_pose[:3, :3]) - 1) / 2)

            # # Translation comparison (Euclidean distance)
            # translation_error = np.linalg.norm(current_t_normalized - gt_t_normalized)

            # # Print normalized results and errors
            # print(f"Normalized Estimated R: \n{current_R_normalized}")
            # print(f"Normalized Estimated t: \n{current_t_normalized}")
            # print(f"Normalized Ground Truth R: \n{gt_pose[:3, :3]}")
            # print(f"Normalized Ground Truth t: \n{gt_t_normalized}")
            # print(f"Rotation Error (radians): {rotation_error}")
            # print(f"Translation Error (Euclidean distance): {translation_error}")
            # print(f"Current Position x: {current_t[0]}")
            # print(f"Current Position y: {current_t[2]}")


            estimated_path.append((current_t[0], current_t[2]))
            gt_path.append((gt_pose[0,3], gt_pose[2,3]))
        self.plot_paths([
            ("Ground Truth Path", gt_path),
            ("Estimated Path", estimated_path)]
            )


if __name__ == "__main__":
    data_dir = "../data/dataset/sequences/01"
    vo = VisualOdometry(
        data_dir,
        max_limit=50,
        feature_extractor=cv2.ORB_create(2000)
        )
    print(f"Intrinsic camera parameters is K = \n{vo.K}")
    print(f"Number of images in {data_dir}: {len(vo.images)}")
    print(f"Number of poses in {data_dir}: {len(vo.gt_poses)}")
    vo.run()


