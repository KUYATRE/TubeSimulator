from __future__ import annotations
import pandas as pd
from src.heatersim.core.models import SimConfig
from src.heatersim.core.simulator import HeaterSimulator


def test_mask_binary():
    # 아주 짧은 타임라인
    df = pd.DataFrame({
        "t_s": [0, 1, 2, 3],
        "SP_Z1": [100, 100, 100, 100]
    })
    cfg = SimConfig(
        dt_s=1.0,
        total_padding_s=0,
        ambient_c=25.0,
        initial_c_per_zone=[25.0],
        tau_s_per_zone=[60.0],
        gain_c_per_unit=1.0,
        hysteresis_c=2.0,
        min_off_s=0,
        min_on_s=0,
        zones=1,
        save_csv=False,
        save_plot=False,
    )
    sim = HeaterSimulator(cfg)
    out = sim.simulate(df)
    assert set(out["OUT_Z1"].unique()).issubset({0, 1})