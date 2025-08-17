from __future__ import annotations
import numpy as np
import pandas as pd
from heatersim.core.models import Recipe

def expand_recipe_to_timeline(recipe: Recipe, dt_s: float, total_padding_s: int = 0) -> pd.DataFrame:
    """
    각 스텝을 1초로 전개. 반환 컬럼: ["t_s", "MV_Z1"]
    (SP 관련 컬럼은 생성하지 않음)
    """
    total_s = int(np.ceil(recipe.total_duration_s + total_padding_s))
    n = max(total_s, 0)

    t = np.arange(0, n, dtype=float)

    # steps의 setpoints_c[0]에 이미 0/1 마스크가 들어있다고 가정
    mv = np.zeros(n, dtype=float)
    cursor = 0
    for step in recipe.steps:
        length = max(1, int(step.duration_s))
        end = min(cursor + length, n)
        val = float(step.setpoints_c[0]) if step.setpoints_c else 0.0
        mv[cursor:end] = val
        cursor = end

    df = pd.DataFrame({"t_s": t, "MV_Z1": mv.astype(int)})
    return df
