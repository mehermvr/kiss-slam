# MIT License

# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from kiss_icp.config.config import (
    AdaptiveThresholdConfig,
    DataConfig,
    MappingConfig,
    RegistrationConfig,
)
from kiss_icp.config.parser import KISSConfig
from map_closures.config.config import MapClosuresConfig
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from rko_lio.config import LIOConfig, PipelineConfig, TimestampConfig


class KissOdometryConfig(BaseModel):
    preprocessing: DataConfig = DataConfig()
    registration: RegistrationConfig = RegistrationConfig()
    mapping: MappingConfig = MappingConfig()
    adaptive_threshold: AdaptiveThresholdConfig = AdaptiveThresholdConfig()


class LoopCloserConfig(BaseModel):
    detector: MapClosuresConfig = MapClosuresConfig()
    overlap_threshold: float = 0.4


class LocalMapperConfig(BaseModel):
    voxel_size: float = 0.5
    splitting_distance: float = 100.0


class OccupancyMapperConfig(BaseModel):
    free_threshold: float = 0.2
    occupied_threshold: float = 0.65
    resolution: float = 0.5
    max_range: Optional[float] = None
    z_min: float = 0.1
    z_max: float = 0.5


class PoseGraphOptimizerConfig(BaseModel):
    max_iterations: int = 10


class RkoLioTimestampConfig(BaseModel):
    multiplier_to_seconds: float = 0.0
    force_absolute: bool = False
    force_relative: bool = False

    def to_rko_lio(self) -> TimestampConfig:
        return TimestampConfig(**self.model_dump())


class RkoLioLIOConfig(BaseModel):
    deskew: bool = True
    max_iterations: int = 100
    voxel_size: float = 1.0
    max_points_per_voxel: int = 20
    max_range: float = 100.0
    min_range: float = 1.0
    convergence_criterion: float = 1e-5
    max_correspondance_distance: float = 0.5
    max_num_threads: int = 0
    initialization_phase: bool = False
    max_expected_jerk: float = 3.0
    double_downsample: bool = True
    min_beta: float = 200.0

    def to_rko_lio(self) -> LIOConfig:
        return LIOConfig(**self.model_dump())


class RkoLioPipelineConfig(BaseModel):
    lio: RkoLioLIOConfig = RkoLioLIOConfig()
    timestamps: RkoLioTimestampConfig = RkoLioTimestampConfig()
    extrinsic_imu2base_quat_xyzw_xyz: Optional[List[float]] = None
    extrinsic_lidar2base_quat_xyzw_xyz: Optional[List[float]] = None
    viz: bool = False
    viz_every_n_frames: int = 20
    dump_deskewed_scans: bool = False
    log_dir: Path = Path("results")
    run_name: Optional[str] = None

    def to_rko_lio(self) -> PipelineConfig:
        return PipelineConfig(
            lio=self.lio.to_rko_lio(),
            timestamps=self.timestamps.to_rko_lio(),
            extrinsic_imu2base_quat_xyzw_xyz=self.extrinsic_imu2base_quat_xyzw_xyz,
            extrinsic_lidar2base_quat_xyzw_xyz=self.extrinsic_lidar2base_quat_xyzw_xyz,
            viz=self.viz,
            viz_every_n_frames=self.viz_every_n_frames,
            dump_deskewed_scans=self.dump_deskewed_scans,
            log_dir=self.log_dir,
            run_name=self.run_name,
        )


class KissSLAMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="kiss_slam_")
    out_dir: str = "slam_output"
    # odometry: KissOdometryConfig = KissOdometryConfig()
    rko_lio: RkoLioPipelineConfig = RkoLioPipelineConfig()
    local_mapper: LocalMapperConfig = LocalMapperConfig()
    occupancy_mapper: OccupancyMapperConfig = OccupancyMapperConfig()
    loop_closer: LoopCloserConfig = LoopCloserConfig()
    pose_graph_optimizer: PoseGraphOptimizerConfig = PoseGraphOptimizerConfig()

    # def kiss_icp_config(self) -> KISSConfig:
    #     return KISSConfig(
    #         out_dir=self.out_dir,
    #         data=self.odometry.preprocessing,
    #         registration=self.odometry.registration,
    #         mapping=self.odometry.mapping,
    #         adaptive_threshold=self.odometry.adaptive_threshold,
    #     )


class KissDumper(yaml.Dumper):
    # HACK: insert blank lines between top-level objects
    # inspired by https://stackoverflow.com/a/44284819/3786245
    def write_line_break(self, data=None):
        super().write_line_break(data)

        if len(self.indents) == 1:
            super().write_line_break()


def _yaml_source(config_file: Optional[Path]) -> Dict[str, Any]:
    data = None
    if config_file is not None:
        with open(config_file) as cfg_file:
            data = yaml.safe_load(cfg_file)
    return data or {}


def load_config(config_file: Optional[Path]) -> KissSLAMConfig:
    """Load configuration from an Optional yaml file. Additionally, deskew and max_range can be
    also specified from the CLI interface"""

    config = KissSLAMConfig(**_yaml_source(config_file))

    # Use specified voxel size or compute one using the max range
    # if config.odometry.mapping.voxel_size is None:
    #     config.odometry.mapping.voxel_size = float(
    #         config.odometry.preprocessing.max_range / 100.0
    #     )
    if config.rko_lio.lio.voxel_size is None:
        config.rko_lio.lio.voxel_size = float(config.rko_lio.lio.max_range / 100.0)

    if config.occupancy_mapper.max_range is None:
        # config.occupancy_mapper.max_range = config.odometry.preprocessing.max_range
        config.occupancy_mapper.max_range = config.rko_lio.lio.max_range

    return config


def write_config(
    config: KissSLAMConfig = KissSLAMConfig(), filename: str = "kiss_slam.yaml"
):
    with open(filename, "w") as outfile:
        yaml.dump(
            config.model_dump(),
            outfile,
            Dumper=KissDumper,
            default_flow_style=False,
            sort_keys=False,
            indent=4,
        )
