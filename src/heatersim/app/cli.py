from __future__ import annotations
import os
import click
import yaml
import logging

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
    logger = logging.getLogger("heatersim")

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
        for z in range(1, zones + 1):
            png = plot_zone(result, z, out_dir=outdir, fname_prefix=os.path.splitext(os.path.basename(recipe_path))[0])
            logger.info(f"Saved plot: {png}")

    logger.info("Simulation done.")


if __name__ == "__main__":
    main()