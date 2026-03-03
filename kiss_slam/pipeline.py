import datetime
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from kiss_icp.tools.pipeline_results import PipelineResults
from pyquaternion import Quaternion
from tqdm import tqdm, trange

from kiss_slam.config.config import KissSLAMConfig, write_config
from kiss_slam.occupancy_mapper import OccupancyGridMapper
from kiss_slam.slam import KissSLAM
from kiss_slam.tools.visualizer import RegistrationVisualizer, StubVisualizer


class SlamPipeline:
    def __init__(
        self,
        dataset,
        config: KissSLAMConfig,
        visualize: bool = False,
        refuse_scans: bool = False,
    ):
        # dataset / scan range
        self._dataset = dataset
        self._n_scans = len(self._dataset)
        self._first = 0
        self._last = self._n_scans

        # config and output dir
        self.config = config
        self.results_dir = None

        # pipelines
        self.kiss_slam = KissSLAM(self.config)

        # results / buffers
        self.results = PipelineResults()
        self.poses = np.zeros((self._n_scans, 4, 4))
        self.dataset_name = self._dataset.__class__.__name__

        # visualizer
        self.visualize = visualize
        self.visualizer = (
            RegistrationVisualizer() if self.visualize else StubVisualizer()
        )
        self._vis_infos = {
            "max_range": self.config.rko_lio.lio.max_range,
            "min_range": self.config.rko_lio.lio.min_range,
        }

        self.refuse_scans = refuse_scans

    def run(self):
        self._run_pipeline()
        self.dump_results_to_disk()
        if self.refuse_scans:
            self._global_mapping()
        return self.results

    def _run_pipeline(self):
        for data in tqdm(self._dataset, total=self._n_scans, unit="data"):
            self.kiss_slam.process_data(data)
            self.visualizer.update(self.kiss_slam)

        self.kiss_slam.generate_new_node()
        self.kiss_slam.local_map_graph.erase_last_local_map()
        self.poses, self.pose_graph = self.kiss_slam.fine_grained_optimization()
        self.poses = np.array(self.poses)

    def _global_mapping(self):
        from kiss_icp.preprocess import get_preprocessor

        if hasattr(self._dataset, "reset"):
            self._dataset.reset()
        ref_ground_alignment = (
            self.kiss_slam.closer.detector.get_ground_alignment_from_id(0)
        )
        deskewing_deltas = np.vstack(
            (
                np.eye(4)[None],
                np.eye(4)[None],
                np.linalg.inv(self.poses[:-2]) @ self.poses[1:-1],
            )
        )
        preprocessor = get_preprocessor(self.config)
        occupancy_grid_mapper = OccupancyGridMapper(self.config.occupancy_mapper)
        print("KissSLAM| Computing Occupancy Grid")
        for idx in trange(self._first, self._last, unit=" frames", dynamic_ncols=True):
            scan, timestamps = self._next(idx)
            deskewed_scan = preprocessor.preprocess(
                scan, timestamps, deskewing_deltas[idx]
            )
            occupancy_grid_mapper.integrate_frame(
                deskewed_scan, ref_ground_alignment @ self.poses[idx - self._first]
            )
        occupancy_grid_mapper.compute_3d_occupancy_information()
        occupancy_grid_mapper.compute_2d_occupancy_information()
        occupancy_dir = os.path.join(self.output_dir, "occupancy_grid")
        os.makedirs(occupancy_dir, exist_ok=True)
        occupancy_grid_mapper.write_3d_occupancy_grid(occupancy_dir)
        occupancy_2d_map_dir = os.path.join(occupancy_dir, "map2d")
        os.makedirs(occupancy_2d_map_dir, exist_ok=True)
        occupancy_grid_mapper.write_2d_occupancy_grid(occupancy_2d_map_dir)

    @property
    def output_dir(self) -> Path:
        if hasattr(self, "_output_dir") and self._output_dir is not None:
            return self._output_dir

        out_path = Path(self.config.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        index = 0
        run_name = self.config.run_name or "kiss_slam"
        while True:
            output_dir = out_path / f"{run_name}_{index:02d}"
            if not output_dir.exists():
                break
            index += 1
        output_dir.mkdir()
        self._output_dir = output_dir
        return output_dir

    def dump_results_to_disk(self):
        output_dir = self.output_dir

        # Poses
        poses = self.poses
        timestamps = self.kiss_slam.scan_end_times
        traj_file = output_dir / f"{output_dir.name}_tum.txt"

        tum_data = np.zeros((len(poses), 8))
        for idx in range(len(poses)):
            tx, ty, tz = poses[idx, :3, -1]
            qw, qx, qy, qz = Quaternion(matrix=poses[idx]).elements
            tum_data[idx] = [timestamps[idx], tx, ty, tz, qx, qy, qz, qw]

        np.savetxt(traj_file, tum_data, fmt="%.6f")

        # Config
        write_config(self.config, (output_dir / "config.yml").as_posix())

        # Metrics log
        self.results.append(
            desc="Number of closures found",
            units="closures",
            value=len(self.kiss_slam.closures),
        )
        self.results.log_to_file(
            str(output_dir / "result_metrics.log"),
            f"Kiss-SLAM Results {self.dataset_name}",
        )

        # Pose graph
        self.pose_graph.write_graph((output_dir / "trajectory.g2o").as_posix())

        # Closures plot
        import matplotlib.pyplot as plt

        locations = [pose[:3, -1] for pose in self.poses]
        loc_x = [loc[0] for loc in locations]
        loc_y = [loc[1] for loc in locations]
        plt.figure(figsize=(10, 8))
        plt.scatter(loc_x, loc_y, s=0.1, color="black")
        key_poses = self.kiss_slam.get_keyposes()
        for closure in self.kiss_slam.closures:
            i, j = closure
            plt.plot(
                [key_poses[i][0, -1], key_poses[j][0, -1]],
                [key_poses[i][1, -1], key_poses[j][1, -1]],
                color="red",
                linewidth=1,
            )
        plt.axis("equal")
        plt.tight_layout()
        plt.savefig(
            output_dir / "trajectory.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

        # Local maps
        local_maps_dir = output_dir / "local_maps"
        local_maps_dir.mkdir(exist_ok=True)
        self.kiss_slam.optimizer.write_graph(
            (local_maps_dir / "local_map_graph.g2o").as_posix()
        )
        plys_dir = local_maps_dir / "plys"
        plys_dir.mkdir(exist_ok=True)
        print("KissSLAM| Writing Local Maps on Disk")
        for local_map in tqdm(self.kiss_slam.local_map_graph.local_maps()):
            filename = plys_dir / f"{local_map.id:06d}.ply"
            local_map.write(str(filename))
