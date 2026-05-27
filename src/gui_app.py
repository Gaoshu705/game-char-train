"""
游戏角色识别 - YOLOv8 图形界面
tkinter 多标签页GUI：训练 / 数据集 / 推理 / 屏幕检测 / 环境
"""
import os
import sys
import io

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONLEGACYWINDOWSFSENCODING"] = "0"
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import json
import time
import threading
import queue
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import resolve_path, PROJECT_ROOT


class RedirectText:
    """将 print 输出重定向到 tkinter Text 控件"""
    def __init__(self, text_widget, log_queue):
        self.text_widget = text_widget
        self.log_queue = log_queue

    def write(self, string):
        if string.strip():
            self.log_queue.put(string.rstrip())

    def flush(self):
        pass


class App(ttk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Game Character Detection - YOLOv8")
        self.master.geometry("1080x700")
        self.master.minsize(900, 650)

        self.log_queue = queue.Queue()
        self.train_thread = None
        self.predict_thread = None
        self.screen_detector = None
        self.screen_thread = None

        self._build_ui()
        self._redirect_stdout()
        self._poll_log_queue()

        os.chdir(str(PROJECT_ROOT))
        print(f"工作目录: {PROJECT_ROOT}")
        print(f"Python: {sys.version.split()[0]}")

    def _build_ui(self):
        self.master.configure(bg="#1e1e2e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 9))
        style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab", padding=[18, 6], font=("Microsoft YaHei UI", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", "#3b3b5c"), ("!selected", "#252540")],
                  foreground=[("selected", "#cdd6f4"), ("!selected", "#a6adc8")])
        style.configure("TLabelframe", background="#252540", foreground="#cdd6f4", borderwidth=1)
        style.configure("TLabelframe.Label", background="#252540", foreground="#cdd6f4",
                        font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("TButton", padding=[10, 4], font=("Microsoft YaHei UI", 9))
        style.configure("Green.TButton", background="#40a02b", foreground="white")
        style.configure("Red.TButton", background="#d20f39", foreground="white")
        style.configure("TLabel", background="#252540", foreground="#cdd6f4")
        style.configure("TEntry", fieldbackground="#313244", foreground="#cdd6f4",
                        font=("Microsoft YaHei UI", 10))
        style.configure("TCombobox", fieldbackground="#313244", foreground="#cdd6f4")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))

        self._build_train_tab()
        self._build_dataset_tab()
        self._build_predict_tab()
        self._build_screen_tab()
        self._build_env_tab()
        self._build_cleanup_tab()

        log_frame = ttk.LabelFrame(self, text=" 日志输出 ", padding=4)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.log_text = tk.Text(
            log_frame, height=8, bg="#1e1e2e", fg="#cdd6f4",
            font=("Consolas", 9), insertbackground="#cdd6f4",
            relief=tk.FLAT, borderwidth=0, wrap=tk.WORD,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.tag_configure("warn", foreground="#f9e2af")
        self.log_text.tag_configure("error", foreground="#f38ba8")
        self.log_text.tag_configure("ok", foreground="#a6e3a1")
        self.log_text.tag_configure("info", foreground="#89b4fa")

    def _build_train_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" 训练 ")

        f1 = ttk.LabelFrame(tab, text=" 模型 & 数据 ", padding=10)
        f1.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(f1, text="模型:").grid(row=0, column=0, sticky=tk.E, padx=(0, 8))
        self.train_model_var = tk.StringVar(value="runs/train/game_char_train/weights/best.pt")
        model_frame = ttk.Frame(f1)
        model_frame.grid(row=0, column=1, sticky=tk.EW, pady=3)
        ttk.Entry(model_frame, textvariable=self.train_model_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="浏览", width=6,
                   command=lambda: self._browse_file(self.train_model_var, "模型文件", [("PyTorch", "*.pt")])).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(f1, text="配置:").grid(row=1, column=0, sticky=tk.E, padx=(0, 8))
        self.train_config_var = tk.StringVar(value="config/train_config.yaml")
        cfg_frame = ttk.Frame(f1)
        cfg_frame.grid(row=1, column=1, sticky=tk.EW, pady=3)
        ttk.Entry(cfg_frame, textvariable=self.train_config_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(cfg_frame, text="加载", width=6,
                   command=self._load_train_config).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(f1, text="数据:").grid(row=2, column=0, sticky=tk.E, padx=(0, 8))
        self.train_data_var = tk.StringVar(value="data/dataset.yaml")
        data_frame = ttk.Frame(f1)
        data_frame.grid(row=2, column=1, sticky=tk.EW, pady=3)
        ttk.Entry(data_frame, textvariable=self.train_data_var, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)

        f1.columnconfigure(1, weight=1)

        f2 = ttk.LabelFrame(tab, text=" 训练参数 ", padding=10)
        f2.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        params = [
            ("epochs", "300"), ("batch", "8"), ("imgsz", "1280"),
            ("lr0", "0.001"), ("lrf", "0.01"), ("patience", "30"),
            ("device", "0"), ("workers", "4"),
        ]
        self.train_param_vars = {}
        for i, (key, default) in enumerate(params):
            row, col = i // 4, (i % 4) * 2
            ttk.Label(f2, text=f"{key}:").grid(row=row, column=col, sticky=tk.E, padx=(16, 4), pady=2)
            var = tk.StringVar(value=default)
            self.train_param_vars[key] = var
            ttk.Entry(f2, textvariable=var, width=14).grid(row=row, column=col + 1, sticky=tk.W, pady=2)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.train_btn = ttk.Button(btn_frame, text="▶  开始训练", command=self._start_training)
        self.train_btn.pack(side=tk.LEFT, padx=4)
        self.train_stop_btn = ttk.Button(btn_frame, text="■ 停止", command=self._stop_training, state=tk.DISABLED)
        self.train_stop_btn.pack(side=tk.LEFT, padx=4)

    def _build_dataset_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" 数据集 ")

        f1 = ttk.LabelFrame(tab, text=" 划分数据集 ", padding=10)
        f1.pack(fill=tk.X, padx=8, pady=8)

        self.ds_raw_images = tk.StringVar(value="data/raw/images")
        self.ds_raw_labels = tk.StringVar(value="data/raw/labels")
        self.ds_output = tk.StringVar(value="data/dataset")

        for label, var in [("图像目录:", self.ds_raw_images), ("标注目录:", self.ds_raw_labels), ("输出目录:", self.ds_output)]:
            row = ttk.Frame(f1)
            row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
            ttk.Button(row, text="...", width=3,
                       command=lambda v=var: self._browse_dir(v)).pack(side=tk.LEFT)

        btn_row = ttk.Frame(f1)
        btn_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_row, text="▶ 划分数据集", command=self._prepare_dataset).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="✓ 检查数据集", command=self._check_dataset).pack(side=tk.LEFT, padx=4)
        self.ds_stat_btn = ttk.Button(btn_row, text="📊 统计信息", command=self._dataset_stats)
        self.ds_stat_btn.pack(side=tk.LEFT, padx=4)

        f2 = ttk.LabelFrame(tab, text=" 从视频提取帧 ", padding=10)
        f2.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.ext_video = tk.StringVar()
        self.ext_interval = tk.StringVar(value="1.0")
        self.ext_count = tk.StringVar()
        self.ext_scale = tk.StringVar(value="1.0")

        r1 = ttk.Frame(f2)
        r1.pack(fill=tk.X, pady=3)
        ttk.Label(r1, text="视频:", width=10).pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.ext_video, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Button(r1, text="浏览", width=6,
                   command=lambda: self._browse_file(self.ext_video, "视频文件",
                       [("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])).pack(side=tk.LEFT)

        r2 = ttk.Frame(f2)
        r2.pack(fill=tk.X, pady=3)
        ttk.Label(r2, text="间隔(秒):", width=10).pack(side=tk.LEFT)
        ttk.Entry(r2, textvariable=self.ext_interval, width=10).pack(side=tk.LEFT)
        ttk.Label(r2, text="  或 均匀N帧:").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(r2, textvariable=self.ext_count, width=10).pack(side=tk.LEFT)
        ttk.Label(r2, text="  缩放:").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(r2, textvariable=self.ext_scale, width=8).pack(side=tk.LEFT)
        ttk.Button(r2, text="▶ 提取帧", command=self._extract_frames).pack(side=tk.LEFT, padx=(12, 0))

    def _build_predict_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" 推理 ")

        f1 = ttk.LabelFrame(tab, text=" 推理设置 ", padding=10)
        f1.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(f1, text="模型:").grid(row=0, column=0, sticky=tk.E, padx=(0, 8))
        self.pred_model = tk.StringVar(value="runs/train/game_char_train/weights/best.pt")
        mf = ttk.Frame(f1)
        mf.grid(row=0, column=1, sticky=tk.EW, pady=3)
        ttk.Entry(mf, textvariable=self.pred_model, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(mf, text="浏览", width=6,
                   command=lambda: self._browse_file(self.pred_model, "模型", [("PyTorch", "*.pt")])).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(f1, text="推理源:").grid(row=1, column=0, sticky=tk.E, padx=(0, 8))
        self.pred_source = tk.StringVar(value="0")
        sf = ttk.Frame(f1)
        sf.grid(row=1, column=1, sticky=tk.EW, pady=3)
        ttk.Entry(sf, textvariable=self.pred_source, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(sf, text="图片", width=6,
                   command=lambda: self._browse_file(self.pred_source, "图片",
                       [("Image", "*.jpg *.jpeg *.png *.bmp")])).pack(side=tk.LEFT, padx=2)
        ttk.Button(sf, text="视频", width=6,
                   command=lambda: self._browse_file(self.pred_source, "视频",
                       [("Video", "*.mp4 *.avi *.mov *.mkv")])).pack(side=tk.LEFT, padx=2)
        ttk.Button(sf, text="目录", width=6,
                   command=lambda: self._browse_dir(self.pred_source)).pack(side=tk.LEFT, padx=2)

        ttk.Label(f1, text="摄像头:").grid(row=2, column=0, sticky=tk.E, padx=(0, 8))
        cf = ttk.Frame(f1)
        cf.grid(row=2, column=1, sticky=tk.EW, pady=3)
        self.pred_camera = tk.StringVar(value="0")
        ttk.Label(cf, text="ID:").pack(side=tk.LEFT)
        ttk.Entry(cf, textvariable=self.pred_camera, width=6).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(cf, text="扫描摄像头", width=10,
                   command=self._scan_cameras).pack(side=tk.LEFT, padx=4)

        ttk.Label(f1, text="输出:").grid(row=3, column=0, sticky=tk.E, padx=(0, 8))
        self.pred_output = tk.StringVar()
        of = ttk.Frame(f1)
        of.grid(row=3, column=1, sticky=tk.EW, pady=3)
        ttk.Entry(of, textvariable=self.pred_output, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(f1, text="置信度:").grid(row=4, column=0, sticky=tk.E, padx=(0, 8))
        self.pred_conf = tk.StringVar(value="0.25")
        ttk.Entry(f1, textvariable=self.pred_conf, width=10).grid(row=4, column=1, sticky=tk.W, pady=3)

        f1.columnconfigure(1, weight=1)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(btn_frame, text="▶ 图片推理", command=lambda: self._run_predict("image")).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="▶ 视频推理", command=lambda: self._run_predict("video")).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="▶ 摄像头", command=lambda: self._run_predict("webcam")).pack(side=tk.LEFT, padx=4)

        prog_frame = ttk.Frame(tab)
        prog_frame.pack(fill=tk.X, padx=8, pady=(4, 2))
        self.pred_progress = ttk.Progressbar(prog_frame, mode="indeterminate", length=400)
        self.pred_progress.pack(fill=tk.X)
        self.pred_status = ttk.Label(prog_frame, text="")

        self.preview_label = ttk.Label(tab, text="推理结果将保存到 runs/ 目录")
        self.preview_label.pack(padx=8, pady=4)

    def _build_screen_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" 屏幕检测 ")

        f1 = ttk.LabelFrame(tab, text=" 屏幕检测设置 ", padding=10)
        f1.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(f1, text="模型:").grid(row=0, column=0, sticky=tk.E, padx=(0, 8))
        self.scr_model = tk.StringVar(value="runs/train/game_char_train/weights/best.pt")
        mf = ttk.Frame(f1)
        mf.grid(row=0, column=1, sticky=tk.EW, pady=3)
        ttk.Entry(mf, textvariable=self.scr_model, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(mf, text="浏览", width=6,
                   command=lambda: self._browse_file(self.scr_model, "模型", [("PyTorch", "*.pt")])).pack(side=tk.LEFT, padx=(4, 0))

        params = [
            ("置信度:", "scr_conf", "0.25"), ("IoU:", "scr_iou", "0.45"),
            ("屏幕编号:", "scr_monitor", "1"), ("缩放:", "scr_scale", "0.5"),
            ("目标帧率:", "scr_fps", "30"),
        ]
        for i, (label, attr, default) in enumerate(params):
            ttk.Label(f1, text=label).grid(row=i + 1, column=0, sticky=tk.E, padx=(0, 8), pady=2)
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(f1, textvariable=var, width=12).grid(row=i + 1, column=1, sticky=tk.W, pady=2)

        self.scr_window_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f1, text="窗口模式（非透明覆盖层）", variable=self.scr_window_var).grid(
            row=6, column=1, sticky=tk.W, pady=4)

        f1.columnconfigure(1, weight=1)

        btn_frame = ttk.Frame(tab)
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.scr_start_btn = ttk.Button(btn_frame, text="▶ 开始屏幕检测", command=self._start_screen)
        self.scr_start_btn.pack(side=tk.LEFT, padx=4)
        self.scr_stop_btn = ttk.Button(btn_frame, text="■ 停止", command=self._stop_screen, state=tk.DISABLED)
        self.scr_stop_btn.pack(side=tk.LEFT, padx=4)

        ttk.Label(tab, text="提示: 透明覆盖层模式下，按 Esc 退出。\n"
                 "框会直接绘制在屏幕上方，鼠标可穿透操作游戏。",
                 foreground="#a6adc8", background="#1e1e2e").pack(padx=8, pady=8)

    def _build_env_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" 环境 ")

        f1 = ttk.LabelFrame(tab, text=" 运行环境 ", padding=10)
        f1.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.env_text = tk.Text(f1, height=18, bg="#1e1e2e", fg="#cdd6f4",
                                font=("Consolas", 10), relief=tk.FLAT,
                                borderwidth=0, state=tk.DISABLED)
        self.env_text.pack(fill=tk.BOTH, expand=True)

        ttk.Button(tab, text="🔄 刷新环境信息", command=self._check_env).pack(padx=8, pady=(0, 8))
        self._check_env()

    def _build_cleanup_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" 清理 ")

        f1 = ttk.LabelFrame(tab, text=" 清理选项 ", padding=10)
        f1.pack(fill=tk.X, padx=8, pady=8)

        self.clean_runs = tk.BooleanVar(value=True)
        self.clean_dataset = tk.BooleanVar(value=False)
        self.clean_raw = tk.BooleanVar(value=False)
        self.clean_dry = tk.BooleanVar(value=False)

        ttk.Checkbutton(f1, text="训练缓存 (runs/)", variable=self.clean_runs).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(f1, text="数据集划分 (data/dataset/)", variable=self.clean_dataset).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(f1, text="⚠ 原始数据 (data/raw/) - 不可恢复!", variable=self.clean_raw).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(f1, text="预览模式 (不实际删除)", variable=self.clean_dry).pack(anchor=tk.W, pady=2)

        ttk.Button(f1, text="▶ 执行清理", command=self._run_cleanup).pack(anchor=tk.W, pady=(8, 0))

    # ── 动作函数 ──────────────────────────────────────────────

    def _load_train_config(self):
        from src.utils import load_config
        cfg_path = resolve_path(self.train_config_var.get())
        if not cfg_path.exists():
            print(f"[WARN] 配置文件不存在: {cfg_path}")
            return
        config = load_config(cfg_path)
        for key, var in self.train_param_vars.items():
            if key in config:
                var.set(str(config[key]))
        if "model" in config:
            self.train_model_var.set(config["model"])
        if "data" in config:
            self.train_data_var.set(config["data"])
        print(f"[OK] 已加载配置: {cfg_path}")

    def _start_training(self):
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showwarning("提示", "训练已在进行中")
            return
        self.train_btn.config(state=tk.DISABLED)
        self.train_stop_btn.config(state=tk.NORMAL)
        self.train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.train_thread.start()

    def _train_worker(self):
        try:
            from src.train import train as _train
            import warnings
            config_path = str(resolve_path(self.train_config_var.get()))
            with open(config_path, "r", encoding="utf-8") as f:
                import yaml
                config = yaml.safe_load(f) or {}

            config["data"] = self.train_data_var.get()
            config["model"] = self.train_model_var.get()
            for key, var in self.train_param_vars.items():
                val = var.get()
                if key == "device" and val != "cpu":
                    val = int(val) if val.isdigit() else val
                elif key in ("epochs", "batch", "imgsz", "patience", "workers"):
                    val = int(val)
                elif key in ("lr0", "lrf"):
                    val = float(val)
                config[key] = val

            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

            print("=" * 50)
            print("开始训练...")
            print(f"模型: {config.get('model')}")
            print(f"数据: {config.get('data')}")
            print(f"epochs: {config.get('epochs')}  batch: {config.get('batch')}  imgsz: {config.get('imgsz')}")
            print("=" * 50)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _train(config_path)
            print("[OK] 训练完成")
        except Exception as e:
            print(f"[ERROR] 训练失败: {e}")
        finally:
            self.master.after(0, self._on_train_done)

    def _on_train_done(self):
        self.train_btn.config(state=tk.NORMAL)
        self.train_stop_btn.config(state=tk.DISABLED)

    def _stop_training(self):
        print("[WARN] 停止训练请求...（请等待当前 epoch 完成）")

    def _prepare_dataset(self):
        self._run_in_thread("prepare", lambda: self._run_script("scripts/prepare_data.py", [
            "--raw_images", str(resolve_path(self.ds_raw_images.get())),
            "--raw_labels", str(resolve_path(self.ds_raw_labels.get())),
            "--output", str(resolve_path(self.ds_output.get())),
        ]))

    def _check_dataset(self):
        self._run_in_thread("check", lambda: self._run_script("scripts/prepare_data.py", [
            "--output", str(resolve_path(self.ds_output.get())),
            "--check_only",
        ]))

    def _dataset_stats(self):
        self._run_in_thread("stats", self._dataset_stats_worker)

    def _dataset_stats_worker(self):
        try:
            from src.dataset import GameCharacterDataset
            import yaml
            ds_path = resolve_path("data/dataset.yaml")
            class_names = {}
            if ds_path.exists():
                with open(ds_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                    names = cfg.get("names", {})
                    class_names = {int(k): v for k, v in names.items()}

            ds = GameCharacterDataset(str(resolve_path(self.ds_output.get())), class_names)
            ds.print_statistics()
        except Exception as e:
            print(f"[ERROR] 统计失败: {e}")

    def _extract_frames(self):
        self._run_in_thread("extract", lambda: self._run_script("scripts/extract_frames.py", [
            "--video", self.ext_video.get(),
            "--output", str(resolve_path(self.ds_raw_images.get())),
            "--interval", self.ext_interval.get(),
            "--scale", self.ext_scale.get(),
        ] + (["--count", self.ext_count.get()] if self.ext_count.get() else [])))

    def _scan_cameras(self):
        import cv2
        print("正在扫描可用摄像头...")
        found = []
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    found.append(f"  [✓] Camera {i} - {w}x{h}")
                else:
                    found.append(f"  [✗] Camera {i} - 可打开但无法读取帧")
                cap.release()
        if found:
            for line in found:
                print(line)
            if len(found) > 0 and "[✓]" in found[0]:
                first_id = found[0].split("]")[0].split("[")[1].replace("✓", "").strip()
                cam_id = first_id.split()[-1]
                self.pred_camera.set(cam_id)
                print(f"已自动选择 Camera {cam_id}")
        else:
            print("未检测到可用摄像头，请确认摄像头已连接")

    def _run_predict(self, mode):
        if mode == "image":
            source = self.pred_source.get()
            if not source or not Path(source).is_file():
                source = filedialog.askopenfilename(
                    title="选择图片",
                    filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp")]
                )
                if source:
                    self.pred_source.set(source)
        elif mode == "video":
            source = self.pred_source.get()
            if not source or not Path(source).is_file():
                source = filedialog.askopenfilename(
                    title="选择视频",
                    filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")]
                )
                if source:
                    self.pred_source.set(source)
        elif mode == "webcam":
            source = self.pred_camera.get().strip() or "0"

        if not source:
            print("[WARN] 请先选择推理源")
            return

        self.pred_status.config(text="推理中，请稍候...")
        self.pred_progress.start(15)
        self._run_in_thread("predict", lambda: self._predict_worker(source))

    def _predict_worker(self, source):
        try:
            model_path = str(resolve_path(self.pred_model.get()))
            if not Path(model_path).exists():
                print(f"[ERROR] 模型不存在: {model_path}")
                return

            from ultralytics import YOLO
            import cv2

            model = YOLO(model_path)
            conf = float(self.pred_conf.get() or 0.25)

            if source.strip().isdigit():
                camera_id = int(source.strip())
                print(f"启动摄像头推理 (ID={camera_id})，按 Q 退出...")
                cap = cv2.VideoCapture(camera_id)
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model(frame, conf=conf, verbose=False)
                    annotated = results[0].plot()
                    cv2.imshow("Camera - Press Q to quit", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                cap.release()
                cv2.destroyAllWindows()
            else:
                output = self.pred_output.get() or None
                if output:
                    output = str(resolve_path(output))
                results = model.predict(
                    source=source, conf=conf, save=True,
                    project=str(resolve_path("runs/predict")),
                    name="", exist_ok=True,
                )
                print(f"[OK] 推理完成，结果保存到 runs/predict/")
                if results and hasattr(results[0], "save_dir"):
                    print(f"       路径: {results[0].save_dir}")
        except Exception as e:
            print(f"[ERROR] 推理失败: {e}")
        finally:
            self.master.after(0, self._on_predict_done)

    def _on_predict_done(self):
        self.pred_progress.stop()
        self.pred_status.config(text="推理完成")

    def _start_screen(self):
        if self.screen_thread and self.screen_thread.is_alive():
            messagebox.showwarning("提示", "屏幕检测已在运行")
            return

        self.scr_start_btn.config(state=tk.DISABLED)
        self.scr_stop_btn.config(state=tk.NORMAL)
        self.screen_thread = threading.Thread(target=self._screen_worker, daemon=True)
        self.screen_thread.start()

    def _screen_worker(self):
        try:
            from src.screen_detect import ScreenDetector
            import yaml as _y

            class_names = {}
            data_yaml = resolve_path("data/dataset.yaml")
            if data_yaml.exists():
                with open(data_yaml, encoding="utf-8") as f:
                    cfg = _y.safe_load(f) or {}
                    names = cfg.get("names", {})
                    class_names = {int(k): v for k, v in names.items()}

            self.screen_detector = ScreenDetector(
                model_path=str(resolve_path(self.scr_model.get())),
                conf=float(self.scr_conf.get() or 0.25),
                iou=float(self.scr_iou.get() or 0.45),
                monitor=int(self.scr_monitor.get() or 1),
                capture_scale=float(self.scr_scale.get() or 0.5),
                target_fps=int(self.scr_fps.get() or 30),
                class_names=class_names,
                overlay=not self.scr_window_var.get(),
            )
            self.screen_detector.run()
        except Exception as e:
            print(f"[ERROR] 屏幕检测异常: {e}")
        finally:
            self.master.after(0, self._on_screen_done)

    def _on_screen_done(self):
        self.scr_start_btn.config(state=tk.NORMAL)
        self.scr_stop_btn.config(state=tk.DISABLED)

    def _stop_screen(self):
        if self.screen_detector:
            self.screen_detector.stop()

    def _check_env(self):
        import io
        old_stdout = sys.stdout
        buf = io.StringIO()
        sys.stdout = buf
        try:
            from src.utils import check_environment
            check_environment()
        finally:
            sys.stdout = old_stdout
        self.env_text.config(state=tk.NORMAL)
        self.env_text.delete("1.0", tk.END)
        self.env_text.insert("1.0", buf.getvalue())
        self.env_text.config(state=tk.DISABLED)

    def _run_cleanup(self):
        import shutil
        dry = self.clean_dry.get()
        targets = []

        if self.clean_runs.get():
            targets.append(("runs/", "训练缓存"))
        if self.clean_dataset.get():
            targets.append(("data/dataset/images/", "数据集图像"))
            targets.append(("data/dataset/labels/", "数据集标注"))
        if self.clean_raw.get():
            targets.append(("data/raw/images/", "原始图像"))
            targets.append(("data/raw/labels/", "原始标注"))

        if not targets:
            print("[WARN] 未选择清理目标")
            return

        print(f"\n{'[预览] ' if dry else ''}清理以下内容:")
        for path, desc in targets:
            print(f"  - {path} ({desc})")

        if dry:
            print("[预览模式] 未实际删除")
            return

        if self.clean_raw.get():
            if not messagebox.askyesno("确认", "包含原始数据，删除后不可恢复！确认清理？"):
                return
        elif not messagebox.askyesno("确认", "确认清理所选内容？"):
            return

        base = PROJECT_ROOT
        for path, _desc in targets:
            full = base / path
            if full.exists():
                for item in full.iterdir():
                    if item.name == ".gitkeep":
                        continue
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
        print("[OK] 清理完成")

    # ── 辅助函数 ──────────────────────────────────────────────

    def _run_in_thread(self, name, target):
        t = threading.Thread(target=target, daemon=True)
        t.start()

    def _run_script(self, script_path, args):
        import subprocess
        cmd = [sys.executable, str(resolve_path(script_path))] + args
        print(f"[CMD] {' '.join(cmd)}")
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env["PYTHONLEGACYWINDOWSFSENCODING"] = "0"
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace",
                                    env=env, cwd=str(PROJECT_ROOT))
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    print(f"  {line}")
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    print(f"  {line}")
        except Exception as e:
            print(f"[ERROR] 脚本执行失败: {e}")

    def _browse_file(self, var, title, filetypes):
        path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if path:
            var.set(str(Path(path)))

    def _browse_dir(self, var):
        path = filedialog.askdirectory(title="选择目录")
        if path:
            var.set(str(Path(path)))

    def _redirect_stdout(self):
        self._orig_stdout = sys.stdout
        sys.stdout = RedirectText(self.log_text, self.log_queue)

    def _poll_log_queue(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
            except queue.Empty:
                break
        self.master.after(100, self._poll_log_queue)


def main():
    root = tk.Tk()

    try:
        root.iconbitmap(default="")
    except Exception:
        pass

    app = App(master=root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()

    sys.stdout = getattr(app, "_orig_stdout", sys.__stdout__)


if __name__ == "__main__":
    main()
