from __future__ import annotations
import pandas as pd

class HeaterSimulator:
    def __init__(self, cfg):
        self.cfg = cfg

    def simulate(self, timeline: pd.DataFrame) -> pd.DataFrame:
        out = timeline.copy()
        if "MV_Z5" not in out.columns:
            raise ValueError(f"'MV_Z5' column is missing. Available: {list(out.columns)}")
        out["OUT_Z5"] = (out["MV_Z5"].astype(float) != 0).astype(int)
        return out
