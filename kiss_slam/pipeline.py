import datetime
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from kiss_icp.tools.pipeline_results import PipelineResults
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
        self._evaluate_closures()
        self._create_output_dir()
        self._write_result_poses()
        self._write_cfg()
        self._write_log()
        self._write_graph()
        self._write_closures()
        self._write_local_maps()
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
        if self.refuse_scans:
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
            for idx in trange(
                self._first, self._last, unit=" frames", dynamic_ncols=True
            ):
                scan, timestamps = self._next(idx)
                deskewed_scan = preprocessor.preprocess(
                    scan, timestamps, deskewing_deltas[idx]
                )
                occupancy_grid_mapper.integrate_frame(
                    deskewed_scan, ref_ground_alignment @ self.poses[idx - self._first]
                )
            occupancy_grid_mapper.compute_3d_occupancy_information()
            occupancy_grid_mapper.compute_2d_occupancy_information()
            occupancy_dir = os.path.join(self.results_dir, "occupancy_grid")
            os.makedirs(occupancy_dir, exist_ok=True)
            occupancy_grid_mapper.write_3d_occupancy_grid(occupancy_dir)
            occupancy_2d_map_dir = os.path.join(occupancy_dir, "map2d")
            os.makedirs(occupancy_2d_map_dir, exist_ok=True)
            occupancy_grid_mapper.write_2d_occupancy_grid(occupancy_2d_map_dir)

    def _write_log(self):
        if not self.results.empty():
            self.results.log_to_file(
                f"{self.results_dir}/result_metrics.log",
                f"Results for {self.dataset_name} Sequence test",
            )

    def _write_cfg(self):
        write_config(self.config, os.path.join(self.results_dir, "config.yml"))

    @staticmethod
    def _get_results_dir(out_dir: str):
        def get_current_timestamp() -> str:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results_dir = os.path.join(os.path.realpath(out_dir), get_current_timestamp())
        latest_dir = os.path.join(os.path.realpath(out_dir), "latest")
        os.makedirs(results_dir, exist_ok=True)
        (
            os.unlink(latest_dir)
            if os.path.exists(latest_dir) or os.path.islink(latest_dir)
            else None
        )
        os.symlink(results_dir, latest_dir)
        return results_dir

    def _create_output_dir(self):
        self.results_dir = self._get_results_dir(self.config.out_dir)

    def _write_result_poses(self):
        np.save(f"{self.results_dir}/test_poses", self.poses)
        np.savetxt(
            f"{self.results_dir}/test_poses_kitti.txt",
            self.poses[:, :3].reshape(-1, 12),
        )

    def _write_local_maps(self):
        local_maps_dir = os.path.join(self.results_dir, "local_maps")
        os.makedirs(local_maps_dir, exist_ok=True)
        self.kiss_slam.optimizer.write_graph(
            os.path.join(local_maps_dir, "local_map_graph.g2o")
        )
        plys_dir = os.path.join(local_maps_dir, "plys")
        os.makedirs(plys_dir, exist_ok=True)
        print("KissSLAM| Writing Local Maps on Disk")
        for local_map in tqdm(self.kiss_slam.local_map_graph.local_maps()):
            filename = os.path.join(plys_dir, "{:06d}.ply".format(local_map.id))
            local_map.write(filename)

    def _evaluate_closures(self):
        self.results.append(
            desc="Number of closures found",
            units="closures",
            value=len(self.kiss_slam.closures),
        )

    def _write_closures(self):
        import matplotlib.pyplot as plt

        locations = [pose[:3, -1] for pose in self.poses]
        loc_x = [loc[0] for loc in locations]
        loc_y = [loc[1] for loc in locations]
        plt.scatter(loc_x, loc_y, s=0.1, color="black")
        key_poses = self.kiss_slam.get_keyposes()
        for closure in self.kiss_slam.closures:
            i, j = closure
            plt.plot(
                [key_poses[i][0, -1], key_poses[j][0, -1]],
                [key_poses[i][1, -1], key_poses[j][1, -1]],
                color="red",
                linewidth=1,
                markersize=1,
            )
        plt.savefig(os.path.join(self.results_dir, "trajectory.png"), dpi=2000)

    def _write_graph(self):
        self.pose_graph.write_graph(os.path.join(self.results_dir, "trajectory.g2o"))
