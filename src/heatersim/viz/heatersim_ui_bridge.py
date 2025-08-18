"""
heatersim_ui_bridge.py

UI(예: PySide6)에서 CLI 기능(simulate, facility-from-z5)을 그대로 호출할 수 있도록
얇은 래퍼 함수와 비동기 워커를 제공합니다.

- 동기 함수
    - simulate_ui(...)
    - facility_from_z5_ui(...)

- 비동기 QRunnable 워커 (QThreadPool에 투입)
    - SimulateWorker
    - FacilityFromZ5Worker

사용 예 (PySide6 MainWindow 내부):

    from PySide6.QtCore import QThreadPool
    from heatersim_ui_bridge import SimulateParams, SimulateWorker

    self.pool = QThreadPool.globalInstance()
    params = SimulateParams(
        recipe_path=recipe_path, config_path=config_path, outdir=outdir,
        plot=True, log_level="INFO"
    )
    worker = SimulateWorker(params)
    worker.signals.finished.connect(self.on_sim_finished)
    worker.signals.error.connect(self.on_sim_error)
    worker.signals.message.connect(self.on_sim_message)
    self.pool.start(worker)

    def on_sim_finished(self, payload: dict):
        print("CSV:", payload.get("csv_path"))
        print("plots:", payload.get("plot_paths"))

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import os
import logging
import yaml

# --- TubeSimul 내부 모듈 ---
from heatersim.io.logging_config import setup_logging
from heatersim.io.recipe_csv import load_recipe_csv
from heatersim.core.timebase import expand_recipe_to_timeline
from heatersim.core.simulator import HeaterSimulator
from heatersim.core.models import SimConfig
from heatersim.io.outputs import save_csv
from heatersim.viz.plot import plot_zone
from heatersim.facility.from_z5 import build_facility_from_z5

# --- (선택) PySide6 비동기 실행용 ---
try:
    from PySide6.QtCore import QObject, Signal, QRunnable
    _HAS_QT = True
except Exception:  # PySide6 미존재 환경에서도 동작하도록
    _HAS_QT = False
    QObject = object  # type: ignore
    Signal = lambda *a, **k: None  # type: ignore
    class QRunnable:  # type: ignore
        def run(self):
            pass


# ===============================================================
# 동기 래퍼: simulate
# ===============================================================
@dataclass
class SimulateParams:
    recipe_path: str
    config_path: str
    outdir: str
    plot: bool = True
    log_level: str = "INFO"


def simulate_ui(recipe_path: str, config_path: str, outdir: str,
                 *, plot: bool = True, log_level: str = "INFO") -> Dict[str, Any]:
    """CLI의 simulate 로직을 UI/코드에서 직접 호출하기 위한 동기 래퍼.

    Returns:
        {
            "csv_path": Optional[str],
            "plot_paths": List[str],
            "result": Any (시뮬레이터 결과 객체)
        }
    """
    setup_logging(getattr(logging, log_level.upper(), logging.INFO))
    logger = logging.getLogger("TubeSimul")

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

    out_csv = None
    plot_paths: List[str] = []

    if cfg.save_csv:
        os.makedirs(outdir, exist_ok=True)
        fname = os.path.splitext(os.path.basename(recipe_path))[0] + "__sim.csv"
        out_csv = save_csv(result, outdir, fname)
        logger.info(f"Saved CSV: {out_csv}")

    if plot and cfg.save_plot:
        for z in range(5, zones + 1):
            png = plot_zone(result, z, out_dir=outdir,
                            fname_prefix=os.path.splitext(os.path.basename(recipe_path))[0])
            plot_paths.append(png)
            logger.info(f"Saved plot: {png}")

    logger.info("Simulation done.")
    return {"csv_path": out_csv, "plot_paths": plot_paths, "result": result}


# ===============================================================
# 동기 래퍼: facility-from-z5
# ===============================================================
@dataclass
class FacilityParams:
    in_csv: str
    facility_cfg: str
    out_csv: str
    pre_s: Optional[int] = None
    post_s: Optional[int] = None
    multi_tube_running: bool = True
    repeat_n: int = 2


def facility_from_z5_ui(in_csv: str, facility_cfg: str, out_csv: str,
                         *, pre_s: Optional[int] = None, post_s: Optional[int] = None,
                         multi_tube_running: bool = True, repeat_n: int = 2) -> str:
    """CLI의 facility-from-z5 로직을 UI/코드에서 직접 호출하기 위한 동기 래퍼.

    Returns:
        out_csv 경로 문자열
    """
    with open(facility_cfg, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    steps = cfg.get("steps", {})
    default_pre = int(steps.get("conveyor_in_s", 0)) + int(steps.get("lifter_in_s", 0))
    default_post = int(steps.get("cooling_store_s", 0))

    pre = default_pre if pre_s is None else int(pre_s)
    post = default_post if post_s is None else int(post_s)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_path = build_facility_from_z5(
        in_csv=in_csv,
        out_csv=out_csv,
        pre_s=pre,
        post_s=post,
        multi_tube_running=multi_tube_running,
        return_df=False,
        repeat_n=repeat_n,
    )
    return out_path


# ===============================================================
# (선택) PySide6 비동기 워커
# ===============================================================
if _HAS_QT:
    class _TaskSignals(QObject):
        finished = Signal(dict)
        error = Signal(str)
        message = Signal(str)

    class SimulateWorker(QRunnable):
        def __init__(self, params: SimulateParams):
            super().__init__()
            self.params = params
            self.signals = _TaskSignals()

        def run(self):
            try:
                self.signals.message.emit("Simulate: 시작")
                payload = simulate_ui(
                    recipe_path=self.params.recipe_path,
                    config_path=self.params.config_path,
                    outdir=self.params.outdir,
                    plot=self.params.plot,
                    log_level=self.params.log_level,
                )
                self.signals.message.emit("Simulate: 완료")
                self.signals.finished.emit(payload)
            except Exception as e:
                self.signals.error.emit(str(e))

    class FacilityFromZ5Worker(QRunnable):
        def __init__(self, params: FacilityParams):
            super().__init__()
            self.params = params
            self.signals = _TaskSignals()

        def run(self):
            try:
                self.signals.message.emit("Heater simulation file 생성: 시작")
                out_path = facility_from_z5_ui(
                    in_csv=self.params.in_csv,
                    facility_cfg=self.params.facility_cfg,
                    out_csv=self.params.out_csv,
                    pre_s=self.params.pre_s,
                    post_s=self.params.post_s,
                    multi_tube_running=self.params.multi_tube_running,
                    repeat_n=self.params.repeat_n,
                )
                self.signals.message.emit("Heater simulation file 생성: 완료")
                self.signals.finished.emit({"out_csv": out_path})
            except Exception as e:
                self.signals.error.emit(str(e))
