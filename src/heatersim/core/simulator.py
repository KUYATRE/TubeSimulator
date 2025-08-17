from __future__ import annotations
import pandas as pd

class HeaterSimulator:
    def __init__(self, cfg):
        self.cfg = cfg

    def simulate(self, timeline: pd.DataFrame) -> pd.DataFrame:
        out = timeline.copy()
        if "MV_Z1" not in out.columns:
            raise ValueError(f"'MV_Z1' column is missing. Available: {list(out.columns)}")
        out["OUT_Z1"] = (out["MV_Z1"].astype(float) != 0).astype(int)
        return out
