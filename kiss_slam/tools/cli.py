# MIT License
#
# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from pathlib import Path
from typing import Optional

import typer
import yaml
from rko_lio.config import PipelineConfig as RKO_LIO_Config
from rko_lio.dataloaders import available_dataloaders, dataloader_factory
from rko_lio.util import (
    error_and_exit,
    info,
    warning,
)


def name_callback(value: str):

    if not value:
        return value
    dl = available_dataloaders()
    if value.lower() not in [d.lower() for d in dl]:
        raise typer.BadParameter(f"Supported dataloaders are: {', '.join(dl)}")
    for d in dl:
        if value.lower() == d.lower():
            return d
    return value


app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@app.command()
def kiss_slam(
    data_path: Path = typer.Argument(
        ...,
        help="The data directory used by the specified dataloader",
        show_default=False,
    ),
    dataloader_name: str = typer.Option(
        None,
        show_default=False,
        case_sensitive=False,
        autocompletion=available_dataloaders,
        callback=name_callback,
        help="[Optional] Use a specific dataloader from those supported by KISS-ICP",
    ),
    visualize: bool = typer.Option(
        False,
        "--visualize",
        "-v",
        help="[Optional] Visualize Ground Truth Loop Closures in the data sequence",
        rich_help_panel="Additional Options",
    ),
    refuse_scans: bool = typer.Option(
        False,
        "--refuse-scans",
        "-rs",
        help="[Optional] At the end of the SLAM run, refuse all the scans into a Global Map using the Pose Estimates",
        rich_help_panel="Additional Options",
    ),
    sequence: Optional[str] = typer.Option(
        None,
        "--sequence",
        "-s",
        show_default=False,
        help="[Optional] For some dataloaders, you need to specify a given sequence",
        rich_help_panel="Additional Options",
    ),
    imu_topic: str | None = typer.Option(
        None,
        "--imu",
        help="Extra dataloader argument: imu topic",
        rich_help_panel="Rosbag dataloader options",
    ),
    lidar_topic: str | None = typer.Option(
        None,
        "--lidar",
        help="Extra dataloader argument: lidar topic",
        rich_help_panel="Rosbag dataloader options",
    ),
    base_frame: str | None = typer.Option(
        None,
        "--base_frame",
        help="Extra dataloader argument: base_frame for odometry estimation, default is lidar frame",
        rich_help_panel="Rosbag dataloader options",
    ),
    imu_frame: str | None = typer.Option(
        None,
        "--imu_frame",
        help="Extra dataloader argument: imu frame overload",
        rich_help_panel="Rosbag dataloader options",
    ),
    lidar_frame: str | None = typer.Option(
        None,
        "--lidar_frame",
        help="Extra dataloader argument: lidar frame overload",
        rich_help_panel="Rosbag dataloader options",
    ),
    config_fp: Optional[Path] = typer.Option(
        None,
        "--config",
        exists=True,
        show_default=False,
        help="[Optional] Path to the configuration file",
    ),
    log_results: bool = typer.Option(
        False,
        "--log",
        "-l",
        help="Log trajectory results to disk at 'log_dir' on completion",
        rich_help_panel="Disk logging options",
    ),
    log_dir: Path | None = typer.Option(
        None,
        "--log_dir",
        "-o",
        help="Where to dump LIO results if logging. If unspecified, and logging is enabled, a folder `results` will be created in the current directory.",
        file_okay=False,
        dir_okay=True,
        writable=True,
        rich_help_panel="Disk logging options",
    ),
    run_name: str | None = typer.Option(
        None,
        "--run_name",
        "-n",
        help="Name prefix for output files if logging. Leave empty to take the name from the data_path argument",
        rich_help_panel="Disk logging options",
    ),
):
    user_rko_lio_config = {}
    if config_fp:
        with open(config_fp, "r") as f:

            user_rko_lio_config.update(yaml.safe_load(f))
    user_rko_lio_config["log_dir"] = log_dir or user_rko_lio_config.get(
        "log_dir", "results"
    )
    user_rko_lio_config["run_name"] = run_name or user_rko_lio_config.get(
        "run_name", data_path.name
    )

    rko_lio_config = RKO_LIO_Config(**user_rko_lio_config)

    dataloader = dataloader_factory(
        name=dataloader_name,
        data_path=data_path,
        sequence=sequence,
        imu_topic=imu_topic,
        lidar_topic=lidar_topic,
        imu_frame_id=imu_frame,
        lidar_frame_id=lidar_frame,
        base_frame_id=base_frame,
        timestamp_processing_config=rko_lio_config.timestamps,
    )
    print("Loaded dataloader:", dataloader)

    from kiss_slam.pipeline import SlamPipeline

    SlamPipeline(
        dataset=dataloader,
        config=rko_lio_config.to_dict(),
        visualize=visualize,
        refuse_scans=refuse_scans,
    ).run().print()


def run():
    app()


if __name__ == "__main__":
    run()
