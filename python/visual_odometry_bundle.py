import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple
from scipy.optimize import least_squares
from visual_odometry import VisualOdometry
from scipy.spatial.transform import Rotation

class VisualOdometryWithBA(VisualOdometry):
    
    def __init__(self, data_dir: str, feature_extractor=cv2.ORB_create(3000), matcher=None, max_limit=None, ba_window_size=5):
        super().__init__(data_dir, feature_extractor, matcher, max_limit)
        self.ba_window_size = ba_window_size
        self.points_3d = []
        self.keypoints = []
        self.poses = []
        self.visibility = []

    def get_shape(self, lst):
        if isinstance(lst, list):
            return (len(lst),) + self.get_shape(lst[0]) if lst else (0,)
        return ()
    def _run_bundle_adjustment(self, keypoints, poses, points_3d):
        """
        Perform local bundle adjustment to optimize the camera poses and 3D points.
        
        Args:
            keypoints: List of keypoints for each frame.
            poses: List of camera poses (rotation and translation).
            points_3d: List of 3D points corresponding to keypoints.
        
        Returns:
            optimized poses and points_3d.
        """
        points3d = np.concatenate([p.flatten() for p in points_3d], axis=0)
        pose_vec = []
        for R, t in poses:
            rotation = Rotation.from_matrix(R)
            q = rotation.as_quat()
            pose_vec.extend(q.tolist())
            pose_vec.extend(t.flatten().tolist())


        def reprojection_residuals(params):
            poses = self._extract_poses_from_vec(params[:len(pose_vec)])
            point_idx = 0
            residuals = []
            for frame_idx, kp in enumerate(keypoints):
                num_visible_points = self.visibility[frame_idx]
                points3d_visible = np.array(params[len(pose_vec) + point_idx: len(pose_vec) + point_idx + num_visible_points * 3]).reshape(-1, 3)
                point_idx += num_visible_points * 3
                q, t = poses[frame_idx]
                R_matrix = Rotation.from_quat(q).as_matrix()
                projected_points = self._project_points(R_matrix, t, points3d_visible)
                for point, keypoint in zip(projected_points, kp):
                    residuals.append(keypoint[0] - point[0])
                    residuals.append(keypoint[1] - point[1])

            return residuals

        params_init = np.concatenate([pose_vec, points3d])
        print(f"BA using {params_init.size} parameters")
        result = least_squares(reprojection_residuals, params_init, verbose=2, method='dogbox', max_nfev=2000)

        optimized_params = result.x
        optimized_poses = self._extract_poses_from_quaternion_poses(optimized_params[:len(pose_vec)])
        optimized_points_3d = np.array(optimized_params[len(pose_vec):]).reshape(-1, 3)

        return optimized_poses, optimized_points_3d
    

    def _extract_poses_from_vec(self, vec):
        poses = []
        for i in range(0, len(vec), 7):
            q_vec = vec[i:i+4] 
            t_vec = vec[i+4:i+7]
            
            q = np.array(q_vec)
            t = np.array(t_vec)  
            poses.append((q, t))
        
        return poses
    
    def _extract_poses_from_quaternion_poses(self, vec):
        poses = []
        for i in range(0, len(vec), 7):
            q_vec = vec[i:i+4] 
            t_vec = vec[i+4:i+7]
            rotation = Rotation.from_quat(q_vec)  
            R_matrix = rotation.as_matrix()
            t = np.array(t_vec)  
            poses.append((R_matrix, t))

        return poses



    def _project_points(self, R, t, points_3d):
        """
        Project 3D points onto 2D using camera intrinsic matrix and the given pose.
        """
        proj_points = []
        for point in points_3d:
            point_homog = np.array([point[0], point[1], point[2], 1])
            proj_point = self.K @ (R @ point_homog[:3] + t)
            proj_points.append([proj_point[0] / proj_point[2], proj_point[1] / proj_point[2]])
        return proj_points
    
    def triangulate_3d_points(self, kp1, kp2, R, t):
        """
        Triangulate 3D points using two views.
        """
        P1 = np.hstack((R, t.reshape(3, 1)))
        P2 = np.hstack((np.eye(3), np.zeros((3, 1))))

        points_3d = cv2.triangulatePoints(P1, P2, kp1.T, kp2.T)
        points_3d = points_3d[:3] / points_3d[3]
        return points_3d.T
    
    
    def run(self):
        estimated_path = []
        gt_path = []
        pose_progress = tqdm(self._get_gt_poses(), unit="pose", desc="Processing ground truth poses")
        for i, gt_pose in enumerate(pose_progress):
            if i == 0:
                current_R = gt_pose[:, :3]
                current_t = gt_pose[:, 3]
                self.poses.append((current_R, current_t))

            else:
                q1, q2 = self._get_matches(i, plot=False)
                R, t = self._get_pose(q1, q2)
                scale = self._calculate_relative_scale(current_R, current_t, R, t, q1, q2)
                t *= -scale
                R = R.T
                current_R = R @ current_R
                current_t += current_R @ t
                self.poses.append((current_R, current_t))
                
                self.keypoints.append(q1)
                self.points_3d.append(self.triangulate_3d_points(q1, q2, current_R, current_t))
                self.visibility.append(self.triangulate_3d_points(q1, q2, current_R, current_t).shape[0])
                print(f"Visible poitns: {self.triangulate_3d_points(q1, q2, current_R, current_t).shape[0]}")

                if len(self.poses) >= self.ba_window_size:
                    keypoints_window = self.keypoints[-self.ba_window_size:]
                    poses_window = self.poses[-self.ba_window_size:]
                    points_3d_window = self.points_3d[-self.ba_window_size:]
                    print(f"Running bundle adjustment for frame: {i}")
                    optimized_poses, optimized_points_3d = self._run_bundle_adjustment(keypoints_window, poses_window, points_3d_window)
                    print(f"Finished with BA")
                    self.poses[-self.ba_window_size:] = optimized_poses
                    self.points_3d[-self.ba_window_size:] = optimized_points_3d

            estimated_path.append((current_t[0], current_t[2]))
            gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))

        self.plot_paths([
            ("Ground Truth Path", gt_path),
            ("Estimated Path", estimated_path)]
        )


if __name__ == "__main__":
    data_dir = "../data/dataset/sequences/01"
    vo = VisualOdometryWithBA(
        data_dir,
        max_limit=50,
        feature_extractor=cv2.ORB_create(1000),
        )
    print(f"Intrinsic camera parameters is K = \n{vo.K}")
    print(f"Number of images in {data_dir}: {len(vo.images)}")
    print(f"Number of poses in {data_dir}: {len(vo.gt_poses)}")
    vo.run()
