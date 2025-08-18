# heater_lamp_timeline.py (UI + Simulate integrated)
# PySide6 UI: drag timeline to blink red lamps for 6 heaters based on CSV (t_s, MV_H1..MV_H6)
# + Facility-from-Z5 생성 버튼 + Recipe 시뮬레이트/로그열기 버튼 (heatersim_ui_bridge 사용)

from __future__ import annotations
import sys
import os
from typing import List, Optional

import pandas as pd
from PySide6.QtCore import Qt, QTimer, QSize, QThreadPool
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QMessageBox, QFrame
)

# UI 브리지(동기/비동기 래퍼)
_HAS_BRIDGE = True
try:
    from heatersim_ui_bridge import (
        FacilityParams, FacilityFromZ5Worker,
        SimulateParams, SimulateWorker,
    )
except Exception:
    _HAS_BRIDGE = False


class LampWidget(QWidget):
    """Circular lamp that can blink when active during dragging."""
    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.title = title
        self.is_on = False
        self.is_dragging = False
        self._blink_phase = False
        self.setMinimumSize(70, 70)

    def sizeHint(self) -> QSize:
        return QSize(80, 90)

    def set_state(self, on: bool, dragging: bool):
        self.is_on = bool(on)
        self.is_dragging = bool(dragging)
        self.update()

    def set_blink_phase(self, phase: bool):
        # called by parent timer while dragging
        if self.is_dragging and self.is_on:
            self._blink_phase = phase
            self.update()
        elif not self.is_dragging:
            self._blink_phase = False

    def paintEvent(self, event):
        w = self.width(); h = self.height()
        radius = int(min(w, h) * 0.38)
        cx, cy = w // 2, int(h * 0.48)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # background
        painter.fillRect(self.rect(), QColor(20, 20, 24))

        # frame
        pen = QPen(QColor(60, 60, 70)); pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(2, 2, -2, -2), 10, 10)

        # lamp color logic
        if self.is_on:
            if self.is_dragging:
                # blink: alternate bright/dim red
                color = QColor(220, 40, 40) if self._blink_phase else QColor(120, 20, 20)
            else:
                # solid red when not dragging
                color = QColor(220, 40, 40)
        else:
            color = QColor(70, 70, 75)  # off

        painter.setPen(QPen(QColor(25, 25, 30), 2))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        # title
        painter.setPen(QPen(QColor(220, 220, 230)))
        painter.drawText(0, h - 18, w, 16, Qt.AlignCenter, self.title)


class HeaterLampPanel(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self); layout.setContentsMargins(8, 8, 8, 8); layout.setSpacing(10)
        self.lamps: List[LampWidget] = [LampWidget(f"H{i}") for i in range(1, 7)]
        for lamp in self.lamps:
            layout.addWidget(lamp)

    def set_states(self, states: List[int], dragging: bool):
        # states length should be 6; missing values treated as 0
        padded = (states + [0] * 6)[:6]
        for lamp, s in zip(self.lamps, padded):
            lamp.set_state(bool(s), dragging)

    def set_blink_phase(self, phase: bool):
        for lamp in self.lamps:
            lamp.set_blink_phase(phase)


class MainWindow(QMainWindow):
    def __init__(self, csv_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("Heater Lamp Timeline")
        self.resize(1200, 460)

        # Data members
        self.df: Optional[pd.DataFrame] = None
        self.cols = [f"MV_H{i}" for i in range(1, 7)]

        # UI
        central = QWidget(); self.setCentralWidget(central)
        vbox = QVBoxLayout(central); vbox.setContentsMargins(12, 10, 12, 10); vbox.setSpacing(8)

        # Top bar
        top = QHBoxLayout(); vbox.addLayout(top)
        self.path_label = QLabel("CSV: (not loaded)")
        btn_open = QPushButton("CSV 열기…"); btn_open.clicked.connect(self.browse_file)
        top.addWidget(self.path_label, 1)
        top.addWidget(btn_open)

        # Facility-from-Z5 버튼
        self.btn_facility = QPushButton("Facility 생성(Z5→H1..H6)")
        self.btn_facility.setEnabled(_HAS_BRIDGE)
        self.btn_facility.clicked.connect(self.on_click_facility)
        top.addWidget(self.btn_facility)

        # Recipe 로그/시뮬레이트 버튼
        self.btn_open_recipe_log = QPushButton("Recipe 로그 열기…")
        self.btn_open_recipe_log.clicked.connect(self.on_open_recipe_log)
        top.addWidget(self.btn_open_recipe_log)

        self.btn_simulate = QPushButton("Recipe 시뮬레이트…")
        self.btn_simulate.setEnabled(_HAS_BRIDGE)
        self.btn_simulate.clicked.connect(self.on_click_simulate)
        top.addWidget(self.btn_simulate)

        # 상태표시
        self.status_label = QLabel("")
        top.addWidget(self.status_label)

        # Panel
        self.panel = HeaterLampPanel(); vbox.addWidget(self.panel, 1)

        # Timeline
        line = QFrame(); line.setFrameShape(QFrame.HLine); line.setStyleSheet("color: #333")
        vbox.addWidget(line)

        bottom = QHBoxLayout(); vbox.addLayout(bottom)
        self.time_label = QLabel("t=0 (idx 0)")
        self.slider = QSlider(Qt.Horizontal); self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)

        bottom.addWidget(QLabel("Time"))
        bottom.addWidget(self.slider, 1)
        bottom.addWidget(self.time_label)

        # Blink timer (only animates while dragging)
        self._blink_timer = QTimer(self); self._blink_timer.setInterval(250)
        self._blink_timer.timeout.connect(self.on_blink)
        self._blink_phase = False

        # Keyboard shortcuts
        act_prev = QAction(self); act_prev.setShortcut(Qt.Key_Left); act_prev.triggered.connect(self.step_left)
        act_next = QAction(self); act_next.setShortcut(Qt.Key_Right); act_next.triggered.connect(self.step_right)
        self.addAction(act_prev); self.addAction(act_next)

        # Thread pool (for bridge workers)
        self.pool = QThreadPool.globalInstance()

        if csv_path and os.path.exists(csv_path):
            self.load_csv(csv_path)

    # --- Data loading ---
    def load_csv(self, path: str):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "CSV 오류", f"CSV를 읽는 중 오류: {e}")
            return

        # Detect heater columns
        available = [c for c in df.columns if c.upper().startswith("MV_H")]
        if not available:
            QMessageBox.critical(self, "컬럼 없음", "MV_H1~MV_H6 컬럼이 없습니다.")
            return

        # Keep only MV_H1..MV_H6 in order (missing -> fill later)
        self.cols = [f"MV_H{i}" for i in range(1, 7)]
        self.df = df
        n = len(df)
        self.slider.setRange(0, n - 1)
        self.slider.setEnabled(True)
        self.path_label.setText(f"CSV: {os.path.basename(path)}  (rows={n})")

        # Initial update
        self.update_lamps(index=0, dragging=False)

    # --- Slider handlers ---
    def on_slider_changed(self, value: int):
        dragging = self.slider.isSliderDown()
        self.update_lamps(index=value, dragging=dragging)
        if dragging and not self._blink_timer.isActive():
            self._blink_timer.start()
        elif not dragging and self._blink_timer.isActive():
            self._blink_timer.stop(); self._blink_phase = False; self.panel.set_blink_phase(False)

    def on_slider_pressed(self):
        # start blink on drag
        if self.df is not None:
            self._blink_timer.start()

    def on_slider_released(self):
        # stop blink after drag
        self._blink_timer.stop(); self._blink_phase = False; self.panel.set_blink_phase(False)
        # Final repaint (solid on/off)
        self.update_lamps(index=self.slider.value(), dragging=False)

    def on_blink(self):
        self._blink_phase = not self._blink_phase
        self.panel.set_blink_phase(self._blink_phase)

    # --- Lamp update ---
    def update_lamps(self, index: int, dragging: bool):
        if self.df is None:
            return
        n = len(self.df)
        idx = max(0, min(index, n - 1))

        states: List[int] = []
        for i in range(1, 7):
            col = f"MV_H{i}"
            if col in self.df.columns:
                try:
                    v = int(self.df.iloc[idx][col])
                except Exception:
                    v = 1 if self.df.iloc[idx][col] else 0
            else:
                # fallback: if MV_H2_doubled_pre exists and i==2
                if i == 2 and "MV_H2_doubled_pre" in self.df.columns:
                    v = int(self.df.iloc[idx]["MV_H2_doubled_pre"])
                else:
                    v = 0
            states.append(1 if v != 0 else 0)

        self.panel.set_states(states, dragging)

        if "t_s" in self.df.columns:
            tval = int(self.df.iloc[idx]["t_s"]) if pd.notna(self.df.iloc[idx]["t_s"]) else idx
        else:
            tval = idx
        self.time_label.setText(f"t={tval} (idx {idx})  |  states: {states}")

    # --- Key step ---
    def step_left(self):
        if self.df is None: return
        self.slider.setValue(max(0, self.slider.value() - 1))

    def step_right(self):
        if self.df is None: return
        self.slider.setValue(min(self.slider.maximum(), self.slider.value() + 1))

    # --- File dialogs ---
    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV 열기", "", "CSV Files (*.csv)")
        if path:
            self.load_csv(path)

    def on_open_recipe_log(self):
        """Recipe 시뮬 결과 CSV 등(= MV_H1..MV_H6를 갖춘 로그)을 직접 선택해서 로드."""
        path, _ = QFileDialog.getOpenFileName(self, "Recipe 로그 CSV 열기", "data/outputs", "CSV Files (*.csv)")
        if path:
            self.load_csv(path)

    # --- Facility-from-Z5: run via bridge worker and load result ---
    def on_click_facility(self):
        if not _HAS_BRIDGE:
            QMessageBox.warning(self, "기능 비활성화", "heatersim_ui_bridge 모듈을 찾을 수 없습니다.")
            return

        in_csv, _ = QFileDialog.getOpenFileName(self, "Z5 기반 원본 CSV 선택", "", "CSV Files (*.csv)")
        if not in_csv:
            return
        cfg_path, _ = QFileDialog.getOpenFileName(self, "facility.yaml 선택", "config", "YAML (*.yaml *.yml)")
        if not cfg_path:
            return
        out_csv, _ = QFileDialog.getSaveFileName(self, "생성될 facility CSV 저장 위치", os.path.join("data", "outputs", "facility_from_z5.csv"), "CSV Files (*.csv)")
        if not out_csv:
            return

        params = FacilityParams(
            in_csv=in_csv,
            facility_cfg=cfg_path,
            out_csv=out_csv,
            pre_s=600,
            post_s=600,
            multi_tube_running=True,
            repeat_n=10,
        )

        worker = FacilityFromZ5Worker(params)
        worker.signals.message.connect(lambda m: self.status_label.setText(m))
        worker.signals.error.connect(lambda e: QMessageBox.critical(self, "생성 오류", e))
        worker.signals.finished.connect(lambda d: self._on_facility_done(d.get("out_csv")))
        self.status_label.setText("FacilityFromZ5: 시작")
        self.pool.start(worker)

    def _on_facility_done(self, out_csv: Optional[str]):
        if not out_csv:
            self.status_label.setText("FacilityFromZ5: 실패")
            return
        self.status_label.setText(f"생성 완료: {os.path.basename(out_csv)}")
        self.load_csv(out_csv)

    # --- Simulate: run via bridge worker and load result ---
    def on_click_simulate(self):
        if not _HAS_BRIDGE:
            QMessageBox.warning(self, "기능 비활성화", "heatersim_ui_bridge 모듈을 찾을 수 없습니다.")
            return

        recipe_path, _ = QFileDialog.getOpenFileName(self, "Recipe CSV 선택", "data/recipes", "CSV Files (*.csv)")
        if not recipe_path:
            return
        config_path, _ = QFileDialog.getOpenFileName(self, "config yaml 선택", "config", "YAML (*.yaml *.yml)")
        if not config_path:
            return
        outdir = QFileDialog.getExistingDirectory(self, "출력 디렉터리 선택", "data/outputs")
        if not outdir:
            return

        params = SimulateParams(recipe_path=recipe_path, config_path=config_path, outdir=outdir,
                                plot=True, log_level="INFO")
        worker = SimulateWorker(params)
        worker.signals.message.connect(lambda m: self.status_label.setText(m))
        worker.signals.error.connect(lambda e: QMessageBox.critical(self, "시뮬 오류", e))
        worker.signals.finished.connect(self._on_simulate_done)
        self.status_label.setText("Simulate: 시작")
        self.pool.start(worker)

    def _on_simulate_done(self, payload: dict):
        self.status_label.setText("Simulate: 완료")
        csv_path = payload.get("csv_path")
        if csv_path:
            self.load_csv(csv_path)
        else:
            QMessageBox.information(self, "완료", "시뮬은 완료했지만 CSV 저장 설정(save_csv)이 비활성화되어 있습니다.")


def run(csv_path: Optional[str] = None):
    app = QApplication(sys.argv)
    win = MainWindow(csv_path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    run(csv_path)
