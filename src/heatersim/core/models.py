from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class RecipeStep:
    step_name: str
    start_s: int
    duration_s: int
    setpoints_c: List[float]  # 길이 = zones

    @property
    def end_s(self) -> int:
        return self.start_s + self.duration_s


@dataclass
class Recipe:
    steps: List[RecipeStep]
    zones: int

    @property
    def total_duration_s(self) -> int:
        return max((s.end_s for s in self.steps), default=0)


@dataclass
class SimConfig:
    dt_s: float
    total_padding_s: int
    ambient_c: float
    initial_c_per_zone: List[float]
    tau_s_per_zone: List[float]
    gain_c_per_unit: float
    hysteresis_c: float
    min_off_s: int
    min_on_s: int
    zones: int
    save_csv: bool
    save_plot: bool

    def normalized(self) -> "SimConfig":
        # zones 길이에 맞추어 리스트 파라미터 보정
        def _fit_list(lst: Sequence[float], default: float) -> List[float]:
            if not lst:
                return [default] * self.zones
            if len(lst) >= self.zones:
                return list(lst)[: self.zones]
            return list(lst) + [lst[-1]] * (self.zones - len(lst))

        return SimConfig(
            dt_s=self.dt_s,
            total_padding_s=self.total_padding_s,
            ambient_c=self.ambient_c,
            initial_c_per_zone=_fit_list(self.initial_c_per_zone, self.ambient_c),
            tau_s_per_zone=_fit_list(self.tau_s_per_zone, max(1.0, self.tau_s_per_zone[0] if self.tau_s_per_zone else 60.0)),
            gain_c_per_unit=self.gain_c_per_unit,
            hysteresis_c=self.hysteresis_c,
            min_off_s=self.min_off_s,
            min_on_s=self.min_on_s,
            zones=self.zones,
            save_csv=self.save_csv,
            save_plot=self.save_plot,
        )