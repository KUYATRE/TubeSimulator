"""
Project entry point.

Usage examples:
  # 1) Launch UI (default command)
  python main.py
  python main.py ui --csv data/outputs/test_log_sample__facility.csv

  # 2) Run simulate (uses ./config/default.yaml)
  python main.py simulate --recipe data/recipes/example.csv --outdir data/outputs

  # 3) Build facility from Z5 (uses ./config/facility.yaml)
  python main.py facility --in-csv data/outputs/test_log_sample__sim.csv \
                         --out data/outputs/facility_from_z5.csv
  # (optional overrides)
  python main.py facility --in-csv ... --override-pre 600 --override-post 600 --override-repeat 10
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import Optional, List, Tuple, Dict, Any

# ---------- Config helpers (exe / python both supported) ----------
def _candidate_base_dirs() -> List[str]:
    dirs = [os.getcwd()]
    if getattr(sys, "frozen", False):  # PyInstaller
        dirs.append(os.path.dirname(sys.executable))
    try:
        dirs.append(os.path.dirname(os.path.abspath(sys.argv[0])))
    except Exception:
        pass
    seen = set(); uniq: List[str] = []
    for d in dirs:
        if d not in seen:
            seen.add(d); uniq.append(d)
    return uniq


def resolve_config_path(*names: str) -> Optional[str]:
    for base in _candidate_base_dirs():
        for name in names:
            p = os.path.join(base, "config", name)
            if os.path.exists(p):
                return p
    return None


def read_facility_yaml(cfg_path: str) -> Tuple[int, int, int, Dict[str, Any]]:
    import yaml  # local import to keep main import-light
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    steps = cfg.get("steps", {})
    default_pre = int(steps.get("conveyor_in_s", 0)) + int(steps.get("lifter_in_s", 0))
    default_post = int(steps.get("cooling_store_s", 0))
    fac = cfg.get("facility", {})
    pre = int(fac.get("pre_s", default_pre))
    post = int(fac.get("post_s", default_post))
    repeat = int(fac.get("repeat_n", 10))
    return pre, post, repeat, cfg


# ---------- Commands ----------
def cmd_ui(csv: Optional[str]) -> int:
    try:
        from src.heatersim.viz.heater_lamp_timeline import run as run_ui
    except Exception as e:
        print(f"[ERROR] UI 모듈 로드 실패: {e}")
        return 2
    run_ui(csv)
    return 0


def cmd_simulate(recipe: str, config: Optional[str], outdir: str, plot: bool, log_level: str) -> int:
    try:
        from src.heatersim.viz.heatersim_ui_bridge import simulate_ui
    except Exception as e:
        print(f"[ERROR] simulate_ui 로드 실패: {e}")
        return 2

    cfg_path = config or resolve_config_path("default.yaml")
    if not cfg_path:
        print("[ERROR] ./config/default.yaml을 찾을 수 없습니다. --config 로 직접 지정하세요.")
        return 1

    os.makedirs(outdir, exist_ok=True)
    payload = simulate_ui(recipe_path=recipe, config_path=cfg_path, outdir=outdir,
                          plot=plot, log_level=log_level)
    csv_path = payload.get("csv_path")
    plots = payload.get("plot_paths")
    print(f"[OK] simulate done. csv={csv_path}, plots={len(plots) if plots else 0}")
    return 0


def cmd_facility(in_csv: str, config: Optional[str], out_csv: str,
                 override_pre: Optional[int], override_post: Optional[int], override_repeat: Optional[int]) -> int:
    try:
        from src.heatersim.viz.heatersim_ui_bridge import facility_from_z5_ui
    except Exception as e:
        print(f"[ERROR] facility_from_z5_ui 로드 실패: {e}")
        return 2

    cfg_path = config or resolve_config_path("facility.yaml")
    if not cfg_path:
        print("[ERROR] ./config/facility.yaml을 찾을 수 없습니다. --config 로 직접 지정하세요.")
        return 1

    pre, post, repeat, _cfg = read_facility_yaml(cfg_path)
    if override_pre is not None:
        pre = int(override_pre)
    if override_post is not None:
        post = int(override_post)
    if override_repeat is not None:
        repeat = int(override_repeat)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_path = facility_from_z5_ui(
        in_csv=in_csv,
        facility_cfg=cfg_path,
        out_csv=out_csv,
        pre_s=pre,
        post_s=post,
        multi_tube_running=True,
        repeat_n=repeat,
    )
    print(f"[OK] facility csv saved: {out_path}")
    return 0


# ---------- Main ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TubeSimulator main entry point")
    sub = p.add_subparsers(dest="cmd")

    # ui (default)
    p_ui = sub.add_parser("ui", help="Launch PySide6 UI")
    p_ui.add_argument("--csv", dest="csv", default=None, help="CSV to open on start")

    # simulate
    p_sim = sub.add_parser("simulate", help="Run heater simulation (uses ./config/default.yaml)")
    p_sim.add_argument("--recipe", required=True, help="Recipe CSV path")
    p_sim.add_argument("--config", default=None, help="Override config yaml (defaults to ./config/default.yaml)")
    p_sim.add_argument("--outdir", default=os.path.join("data", "outputs"))
    p_sim.add_argument("--plot", dest="plot", action="store_true", default=True)
    p_sim.add_argument("--no-plot", dest="plot", action="store_false")
    p_sim.add_argument("--log-level", default="INFO")

    # facility
    p_fac = sub.add_parser("facility", help="Build facility csv from Z5 (uses ./config/facility.yaml)")
    p_fac.add_argument("--in-csv", required=True, help="Input CSV with MV_Z5/OUT_Z5")
    p_fac.add_argument("--config", default=None, help="Override facility.yaml path (defaults to ./config/facility.yaml)")
    p_fac.add_argument("--out", dest="out_csv", default=os.path.join("data", "outputs", "facility_from_z5.csv"))
    p_fac.add_argument("--override-pre", type=int, default=None)
    p_fac.add_argument("--override-post", type=int, default=None)
    p_fac.add_argument("--override-repeat", type=int, default=None)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # default command: ui
    if not args.cmd:
        return cmd_ui(csv=None)

    if args.cmd == "ui":
        return cmd_ui(csv=args.csv)
    if args.cmd == "simulate":
        return cmd_simulate(recipe=args.recipe, config=args.config, outdir=args.outdir, plot=args.plot, log_level=args.log_level)
    if args.cmd == "facility":
        return cmd_facility(in_csv=args.in_csv, config=args.config, out_csv=args.out_csv,
                            override_pre=args.override_pre, override_post=args.override_post, override_repeat=args.override_repeat)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
