from __future__ import annotations
import os
import pandas as pd


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_csv(df: pd.DataFrame, out_dir: str, filename: str) -> str:
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, filename)
    df.to_csv(out_path, index=False)
    return out_path