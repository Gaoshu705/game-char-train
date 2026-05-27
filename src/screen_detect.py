"""
实时屏幕检测
捕获屏幕内容 -> YOLO推理 -> 检测框标注在游戏画面上

支持两种显示模式:
  - overlay (默认): 透明覆盖层，框直接显示在屏幕上方，鼠标穿透
  - window: OpenCV 窗口，自带画面预览

用法:
  python src/screen_detect.py --model yolov8n.pt
  python src/screen_detect.py --model best.pt --conf 0.5 --fps 60 --window
"""

import os
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import draw_boxes, resolve_path


class ScreenDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        monitor: int = 1,
        capture_scale: float = 1.0,
        target_fps: int = 30,
        class_names: dict = None,
        overlay: bool = True,
    ):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.monitor = monitor
        self.capture_scale = capture_scale
        self.target_fps = target_fps
        self.class_names = class_names or {}
        self.overlay = overlay

        self.model = None
        self.sct = None
        self.running = False
        self.root = None
        self.canvas = None
        self.screen_w = 1920
        self.screen_h = 1080
        self.monitor_w = 1920
        self.monitor_h = 1080

    def _load_model(self):
        model_path = str(resolve_path(self.model_path))
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        self.model = YOLO(model_path)
        print(f"模型已加载: {model_path}")

    def _setup_capture(self):
        import mss

        self.sct = mss.mss()
        monitors = self.sct.monitors
        if self.monitor >= len(monitors):
            print(f"[WARNING] 显示器 {self.monitor} 不存在，使用主显示器")
            self.monitor = 1

        region = monitors[self.monitor]
        self.monitor_w = region["width"]
        self.monitor_h = region["height"]
        print(f"截屏区域: {self.monitor_w}x{self.monitor_h}  (left={region['left']}, top={region['top']})")

    def _setup_overlay(self):
        try:
            import tkinter as tk
        except ImportError:
            print("[ERROR] tkinter 不可用，回退到窗口模式")
            self.overlay = False
            self._setup_window()
            return

        self.root = tk.Tk()
        self.root.title("Game Detection Overlay")

        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)

        self.transparent_color = "#010101"
        self.root.configure(bg=self.transparent_color)
        self.root.attributes("-transparentcolor", self.transparent_color)

        self.canvas = tk.Canvas(
            self.root,
            bg=self.transparent_color,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.root.bind("<Escape>", lambda e: self.stop())
        self.root.bind("<q>", lambda e: self.stop())
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()

        self._set_click_through()
        print(f"覆盖层已创建: {self.screen_w}x{self.screen_h}")

    def _set_click_through(self):
        try:
            import ctypes
            from ctypes import wintypes

            self.root.update_idletasks()

            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED = 0x00080000

            hwnd = ctypes.windll.user32.FindWindowW(None, "Game Detection Overlay")
            if hwnd:
                ex_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                ctypes.windll.user32.SetWindowLongW(
                    hwnd,
                    GWL_EXSTYLE,
                    ex_style | WS_EX_TRANSPARENT | WS_EX_LAYERED,
                )
                print("覆盖层: 鼠标穿透已启用")
        except Exception as e:
            print(f"[WARNING] 鼠标穿透设置失败: {e}")
            print("鼠标在覆盖层区域可能无法穿透到游戏窗口")

    def _setup_window(self):
        cv2.namedWindow("Screen Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Screen Detection", cv2.WND_PROP_TOPMOST, 1)

    def _capture_frame(self):
        import mss

        sct_img = self.sct.grab(self.sct.monitors[self.monitor])
        frame = np.array(sct_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if self.capture_scale != 1.0:
            new_w = int(frame.shape[1] * self.capture_scale)
            new_h = int(frame.shape[0] * self.capture_scale)
            frame = cv2.resize(frame, (new_w, new_h))

        return frame

    def _inference(self, frame):
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)

        boxes_list = []
        class_ids_list = []
        scores_list = []

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_list.append(result.boxes.xyxy.cpu().numpy())
                class_ids_list.append(result.boxes.cls.cpu().numpy())
                scores_list.append(result.boxes.conf.cpu().numpy())

        if boxes_list:
            return (
                np.concatenate(boxes_list),
                np.concatenate(class_ids_list),
                np.concatenate(scores_list),
            )
        return np.array([]), np.array([]), np.array([])

    def _draw_on_overlay(self, frame, boxes, class_ids, scores):
        self.canvas.delete("all")

        if len(boxes) == 0:
            self.root.update()
            return

        scale_x = self.screen_w / frame.shape[1]
        scale_y = self.screen_h / frame.shape[0]

        colors = [
            "#00FF00", "#FF4444", "#44AAFF", "#FFAA00",
            "#FF44FF", "#44FFAA", "#AA44FF", "#FFFF44",
            "#FF8844", "#88FF44", "#4488FF", "#FFFFFF",
        ]

        for i in range(len(boxes)):
            if scores[i] < self.conf:
                continue

            x1 = int(boxes[i][0] * scale_x)
            y1 = int(boxes[i][1] * scale_y)
            x2 = int(boxes[i][2] * scale_x)
            y2 = int(boxes[i][3] * scale_y)

            cls_id = int(class_ids[i])
            color = colors[cls_id % len(colors)]

            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color,
                width=2,
            )

            label = self.class_names.get(cls_id, f"cls_{cls_id}")
            label = f"{label} {scores[i]:.2f}"

            text_id = self.canvas.create_text(
                x1 + 3, y1 + 1,
                text=label,
                anchor="nw",
                fill="white",
                font=("Consolas", 9, "bold"),
            )
            bbox = self.canvas.bbox(text_id)
            if bbox:
                bg_id = self.canvas.create_rectangle(
                    bbox[0] - 2, bbox[1] - 1,
                    bbox[2] + 2, bbox[3] + 1,
                    fill=color,
                    outline="",
                )
                self.canvas.tag_raise(text_id)

        self.root.update()

    def run(self):
        self._load_model()
        self._setup_capture()

        if self.overlay:
            self._setup_overlay()
        else:
            self._setup_window()

        self.running = True
        frame_time = 1.0 / self.target_fps if self.target_fps > 0 else 0

        print(f"\n屏幕检测已启动")
        print(f"显示模式: {'透明覆盖层' if self.overlay else 'OpenCV 窗口'}")
        print(f"屏幕分辨率: {self.monitor_w}x{self.monitor_h}")
        print(f"推理缩放: {self.capture_scale}x")
        print(f"目标帧率: {self.target_fps} FPS")
        print("按 Esc 或 窗口关闭 退出\n")

        last_log = time.time()

        try:
            while self.running:
                loop_start = time.time()

                frame = self._capture_frame()
                boxes, class_ids, scores = self._inference(frame)

                if self.overlay:
                    self._draw_on_overlay(frame, boxes, class_ids, scores)
                else:
                    if len(boxes) > 0:
                        frame = draw_boxes(
                            frame, boxes, class_ids, scores,
                            self.class_names,
                            conf_threshold=self.conf,
                        )
                    cv2.imshow("Screen Detection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord("q"), 27]:
                        self.stop()

                loop_elapsed = time.time() - loop_start
                sleep_time = max(0, frame_time - loop_elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

                now = time.time()
                if now - last_log >= 5.0:
                    actual_fps = 1.0 / max(loop_elapsed, 0.001)
                    print(
                        f"FPS: {actual_fps:.1f}  |  检测到: {len(boxes)} 个目标  "
                        f"|  推理耗时: {loop_elapsed*1000:.0f}ms"
                    )
                    last_log = now

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self.running = False
        if self.root:
            try:
                self.root.destroy()
            except Exception:
                pass
            self.root = None
        if not self.overlay:
            cv2.destroyAllWindows()
        if self.sct:
            self.sct.close()
        print("\n屏幕检测已停止")


def main():
    parser = argparse.ArgumentParser(description="实时屏幕检测 - 游戏角色识别")
    parser.add_argument("--model", default="yolov8n.pt", help="模型路径")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--monitor", type=int, default=1, help="显示器编号 (1=主显示器)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="截图缩放比例 (0.5=半分辨率, 加速推理)")
    parser.add_argument("--fps", type=int, default=30, help="目标帧率上限")
    parser.add_argument("--data", default="data/dataset.yaml",
                        help="数据集配置文件，用于加载类别名")
    parser.add_argument("--window", action="store_true",
                        help="使用 OpenCV 窗口模式 (默认透明覆盖层)")

    args = parser.parse_args()

    class_names = {}
    data_yaml = resolve_path(args.data)
    if data_yaml.exists():
        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            names = cfg.get("names", {})
            class_names = {int(k): v for k, v in names.items()}

    detector = ScreenDetector(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        monitor=args.monitor,
        capture_scale=args.scale,
        target_fps=args.fps,
        class_names=class_names,
        overlay=not args.window,
    )
    detector.run()


if __name__ == "__main__":
    main()
