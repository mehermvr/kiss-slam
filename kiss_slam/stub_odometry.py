from bisect import bisect_left
from pathlib import Path

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import get_preprocessor, get_voxel_hash_map
from kiss_icp.voxelization import voxel_down_sample
from scipy.spatial.transform import Rotation as R

from kiss_slam.kiss_slam_pybind import kiss_slam_pybind


def create_transformation(translation, quaternion_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quaternion_xyzw).as_matrix()
    T[:3, 3] = translation
    return T


def read_tum_trajectory(trajectory_fp: Path) -> dict[float, np.ndarray]:
    assert trajectory_fp.exists(), f"trajectory file {trajectory_fp} does not exist."
    trajectory = {}
    with open(trajectory_fp, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                timestamp = float(parts[0])
                translation = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
                quaternion_xyzw = np.array(
                    [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                )
                pose = create_transformation(translation, quaternion_xyzw)
                trajectory[timestamp] = pose
    return trajectory


class StubOdometry:
    """Mocking a Lidar Inertial Odometry which actually just reads poses from a file on disk"""

    def __init__(self, config: KISSConfig, trajectory_fp: Path):
        """
        Initialize the StubOdometry object with a configuration.
        """
        self.config = config
        self.preprocessor = get_preprocessor(self.config)
        self.last_pose = np.eye(4)  # Default to identity matrix
        self.local_map = get_voxel_hash_map(self.config)

        self._trajectory = read_tum_trajectory(trajectory_fp)
        self._sorted_trajectory_times = sorted(self._trajectory.keys())

    def _get_pose(self, stamp: float):
        if stamp in self._trajectory:
            pose = self._trajectory[stamp]
        elif stamp < self._sorted_trajectory_times[0]:
            pose = np.eye(4)
        else:
            # Find the lowest nearest neighbor in time
            upper_idx = bisect_left(self._sorted_trajectory_times, stamp)
            lower_idx = upper_idx - 1
            assert (
                lower_idx >= 0
            ), f"lower_idx is {lower_idx}, somethings wrong, stamp {stamp}"

            t1, t2 = (
                self._sorted_trajectory_times[lower_idx],
                self._sorted_trajectory_times[upper_idx],
            )
            assert (
                t1 < stamp and t2 > stamp
            ), f"{t1} < {stamp}, {t2} > {stamp}, not the case, so something is wrong"
            pose1, pose2 = self._trajectory[t1], self._trajectory[t2]
            # Interpolate
            t = (stamp - t1) / (t2 - t1)
            pose = kiss_slam_pybind._interpolate_se3d(t, pose1, pose2)
        return pose

    def register_frame(self, frame, timestamps: np.ndarray):
        # get the max timestamp from timestamps as the stamp of the lidar point cloud
        min_stamp = min(timestamps)
        max_stamp = max(timestamps)
        min_stamp_pose = self._get_pose(min_stamp)
        max_stamp_pose = self._get_pose(max_stamp)

        # here i am using optimized lio poses for motion delta, in actual it would be the imu initial guess for a true odometry result
        motion_delta = np.linalg.inv(min_stamp_pose) @ max_stamp_pose
        deskewed_frame = self.preprocessor.preprocess(frame, timestamps, motion_delta)
        assert self.config.mapping.voxel_size is not None, "i need it for type checking"
        frame_downsample = voxel_down_sample(
            deskewed_frame, self.config.mapping.voxel_size * 0.5
        )
        new_pose = self.last_pose @ motion_delta
        self.local_map.update(frame_downsample, new_pose)
        self.last_pose = new_pose
        return deskewed_frame, None
