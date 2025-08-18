from __future__ import annotations
import os
import click
import yaml
import logging

from heatersim.facility.from_z5 import build_facility_from_z5

from heatersim.io.logging_config import setup_logging
from heatersim.io.recipe_csv import load_recipe_csv
from heatersim.core.timebase import expand_recipe_to_timeline
from heatersim.core.simulator import HeaterSimulator
from heatersim.core.models import SimConfig
from heatersim.io.outputs import save_csv
from heatersim.viz.plot import plot_zone


@click.group()
def main():
    """Heater Output Mask Simulator CLI"""
    pass


@main.command()
@click.option("--recipe", "recipe_path", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False, path_type=str), default=os.path.join("config", "default.yaml"))
@click.option("--outdir", type=click.Path(file_okay=False, path_type=str), default=os.path.join("data", "outputs"))
@click.option("--plot/--no-plot", default=True)
@click.option("--log-level", default="INFO")

def simulate(recipe_path: str, config_path: str, outdir: str, plot: bool, log_level: str):
    setup_logging(getattr(logging, log_level.upper(), logging.INFO))
    logger = logging.getLogger("TubeSimul")

    # 설정 로드
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)

    zones = int(cfg_yaml.get("zones", 8))
    cfg = SimConfig(
        dt_s=float(cfg_yaml["sampling"]["dt_s"]),
        total_padding_s=int(cfg_yaml["sampling"].get("total_padding_s", 0)),
        ambient_c=float(cfg_yaml["plant"]["ambient_c"]),
        initial_c_per_zone=list(cfg_yaml["plant"].get("initial_c_per_zone", [])),
        tau_s_per_zone=list(cfg_yaml["plant"].get("tau_s_per_zone", [])),
        gain_c_per_unit=float(cfg_yaml["plant"]["gain_c_per_unit"]),
        hysteresis_c=float(cfg_yaml["control"]["hysteresis_c"]),
        min_off_s=int(cfg_yaml["control"].get("min_off_s", 0)),
        min_on_s=int(cfg_yaml["control"].get("min_on_s", 0)),
        zones=zones,
        save_csv=bool(cfg_yaml["output"]["save_csv"]),
        save_plot=bool(cfg_yaml["output"]["save_plot"]),
    )

    logger.info(f"Loading recipe: {recipe_path}")
    recipe = load_recipe_csv(recipe_path, zones=zones)
    timeline = expand_recipe_to_timeline(recipe, dt_s=cfg.dt_s, total_padding_s=cfg.total_padding_s)

    sim = HeaterSimulator(cfg)
    result = sim.simulate(timeline)

    # 저장
    if cfg.save_csv:
        fname = os.path.splitext(os.path.basename(recipe_path))[0] + "__sim.csv"
        out_csv = save_csv(result, outdir, fname)
        logger.info(f"Saved CSV: {out_csv}")

    if plot and cfg.save_plot:
        for z in range(5, zones + 1):
            png = plot_zone(result, z, out_dir=outdir, fname_prefix=os.path.splitext(os.path.basename(recipe_path))[0])
            logger.info(f"Saved plot: {png}")

    logger.info("Simulation done.")

@main.command(name="facility-from-z5")
@click.option("--in-csv", type=click.Path(exists=True, dir_okay=False, path_type=str), required=True)
@click.option("--config", "facility_cfg", type=click.Path(exists=True, dir_okay=False, path_type=str),
              default=os.path.join("config", "facility.yaml"))
@click.option("--out", "out_csv", type=click.Path(dir_okay=False, path_type=str),
              default=os.path.join("data", "outputs", "test_log_sample__facility.csv"))
@click.option("--pre-s", type=int, default=None, help="앞쪽 0패딩(컨베이어 반입 + 리프터 이송) 초")
@click.option("--post-s", type=int, default=None, help="뒤쪽 0패딩(쿨링 스토어 안착) 초")

def facility_from_z5(in_csv: str, facility_cfg: str, out_csv: str, pre_s: int | None, post_s: int | None):
    """ZONE5 마스크→HEATER1 마스크로 매핑하고, 공정 외 구간 0을 앞/뒤에 붙여 facility CSV 생성."""
    with open(facility_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    steps = cfg.get("steps", {})
    default_pre = int(steps.get("conveyor_in_s", 0)) + int(steps.get("lifter_in_s", 0))
    default_post = int(steps.get("cooling_store_s", 0))

    pre = default_pre if pre_s is None else int(pre_s)
    post = default_post if post_s is None else int(post_s)

    out_path = build_facility_from_z5(in_csv=in_csv, out_csv=out_csv, pre_s=pre, post_s=post, multi_tube_running=True, repeat_n=2)
    click.echo(f"Saved facility CSV: {out_path}")


if __name__ == "__main__":
    main()