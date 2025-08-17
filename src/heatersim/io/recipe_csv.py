from __future__ import annotations
import re
import pandas as pd
from heatersim.core.models import Recipe, RecipeStep

# ZONE1 MV 컬럼 후보 패턴 (필요시 추가)
MV_PATS = [
    r"^ZONE\s*1\s*\(\s*MV\s*\)$",   # "ZONE1 (MV)"
    r"^ZONE1\(\s*MV\s*\)$",         # "ZONE1(MV)"
    r"^ZONE1[_\s-]*MV$",            # "ZONE1_MV", "ZONE1 MV"
    r"^MV[_\s-]*Z?1$",              # "MV_Z1", "MV 1"
    r".*\(MV\).*1.*",               # 임의 "(MV)" + '1' 포함
]

def _find_mv_col(df: pd.DataFrame) -> str | None:
    for pat in MV_PATS:
        rx = re.compile(pat, re.IGNORECASE)
        for c in df.columns:
            if rx.search(str(c)):
                return c
    # 최후의 보루: MV가 붙은 컬럼이 하나뿐이라면 허용
    mv_like = [c for c in df.columns if "(MV)" in str(c)]
    if len(mv_like) == 1:
        return mv_like[0]
    return None

def load_recipe_csv(path: str, zones: int) -> Recipe:
    df = pd.read_csv(path)

    mv_col = _find_mv_col(df)
    if mv_col is None:
        raise ValueError(f"CSV에 ZONE1의 MV 컬럼이 없습니다. Available: {list(df.columns)}")

    steps: list[RecipeStep] = []
    # 각 행이 1초로 간주되므로, 행 인덱스 i = 시간(초)
    for i, row in df.iterrows():
        raw = float(row[mv_col])
        mask = 1.0 if raw != 0.0 else 0.0  # 0 → 0, 비0 → 1
        steps.append(RecipeStep(step_name="AUTO", start_s=int(i), duration_s=1, setpoints_c=[mask]))

    steps.sort(key=lambda s: s.start_s)
    return Recipe(steps=steps, zones=1)
