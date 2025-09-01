from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd


def _extract_z5_mask(df: pd.DataFrame) -> np.ndarray:
    """Return 1/0 mask from MV_Z5 or OUT_Z5."""
    if "MV_Z5" in df.columns:
        base = df["MV_Z5"].astype(float).to_numpy()
    elif "OUT_Z5" in df.columns:
        base = df["OUT_Z5"].astype(float).to_numpy()
    else:
        raise ValueError(
            f"Input CSV must contain MV_Z5 or OUT_Z5. Available: {list(df.columns)}"
        )
    return (base != 0).astype(int)


def build_facility_from_z5(
    in_csv: str,
    out_csv: Optional[str] = None,
    pre_s: int = 0,
    post_s: int = 0,
    *,
    multi_tube_running: bool = False,
    return_df: bool = False,
    repeat_n: int = 1,  # MV_H1 전체 repeat_n회 반복, MV_H2~H6은 1회 + MV_H1을 (repeat_n-1)회
) -> str | pd.DataFrame:
    """
    Z5 마스크를 H1으로 매핑하고, 공정 외 구간을 0으로 채워 앞/뒤로 붙여 CSV/DF 생성.

    - MV_H1: 기본 세그먼트를 시간축으로 repeat_n회 **연속 반복** 저장
    - MV_H2~H6 (multi_tube_running=True): 각 히터의 패딩 반영 세그먼트를 **첫 구간 1회** 두고,
      이후 구간(2..repeat_n)은 MV_H1과 동일한 세그먼트를 사용
    - 길이 보정은 **반복을 모두 구성한 뒤** 최종 길이(total_len)에 맞춰 한 번만 수행
    """
    df_in = pd.read_csv(in_csv)

    mv5 = _extract_z5_mask(df_in)

    pre_pad = np.zeros(int(pre_s), dtype=int)
    post_pad = np.zeros(int(post_s), dtype=int)

    # H1 base segment & repeated timeline
    mv_h1_seg = np.concatenate([pre_pad, mv5])
    mv_repeat = np.concatenate([post_pad, mv5])
    seg_len = len(mv_h1_seg)

    rep = int(repeat_n) if repeat_n is not None else 1
    total_len = seg_len + len(mv_repeat) * (rep-1)

    t_s_full = np.arange(total_len, dtype=int)
    mv_h1_full = np.concatenate([mv_h1_seg, np.tile(mv_repeat, rep-1)])

    data = {
        "t_s": t_s_full,
        "MV_H1": mv_h1_full,
    }

    if multi_tube_running:
        for i in range(2, 7):
            # 1) 히터별 패딩 세그먼트 (첫 구간)
            pre_pad_hi = np.zeros(int(pre_s) * i, dtype=int)
            mv_hi_first = np.concatenate([pre_pad_hi, mv5])

            # 2) 전체 반복 구성: 첫 구간은 mv_hi_first, 이후는 mv_h1_seg
            if rep <= 1:
                mv_hi_full = mv_hi_first
            else:
                tail_parts = [mv_repeat] * (rep - 1)
                mv_hi_full = np.concatenate([mv_hi_first] + tail_parts)

            # 3) 길이 보정은 여기서 **최종 길이에 대해 한 번만** 수행
            cur_len = len(mv_hi_full)
            if cur_len > total_len:
                mv_hi_full = mv_hi_full[:total_len]
            elif cur_len < total_len:
                mv_hi_full = np.pad(mv_hi_full, (total_len - cur_len, 0), mode="constant", constant_values=0)

            data[f"MV_H{i}"] = mv_hi_full

    out_df = pd.DataFrame(data)

    if return_df and out_csv is None:
        return out_df

    if out_csv is None:
        raise ValueError(
            "out_csv is None and return_df is False: 어디에도 출력하지 않습니다. 하나는 지정해주세요."
        )

    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out_csv
