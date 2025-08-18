from __future__ import annotations
import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_zone(df: pd.DataFrame, zone: int, out_dir: str | None = None, fname_prefix: str = "result") -> str | None:
    t = df["t_s"].values if "t_s" in df.columns else range(len(df))
    mv = df.get(f"MV_Z{zone}")
    out_mask = df.get(f"OUT_Z{zone}")

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(2, 1, 1)
    if mv is not None:
        ax1.step(t, mv, where="post", label=f"MV_Z{zone}")
    ax1.set_title(f"Zone {zone} MV")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("0/1")
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    if out_mask is not None:
        ax2.step(t, out_mask, where="post", label=f"OUT_Z{zone}")
    ax2.set_title(f"Zone {zone} Heater Output")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("0/1")
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()

    fig.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{fname_prefix}_Z{zone}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path
    else:
        plt.show()
        return None
