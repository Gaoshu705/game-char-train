# 🎮 Game Character Detection - YOLOv8

基于 YOLOv8 的游戏角色识别与区分系统，支持训练、推理、实时屏幕检测。

---

## 目录

- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
  - [标注工具](#标注工具)
  - [从视频提取帧](#从视频提取帧)
  - [划分数据集](#划分数据集)
- [训练模型](#训练模型)
- [模型推理](#模型推理)
- [实时屏幕检测](#实时屏幕检测)
- [图形界面](#图形界面)
- [常用命令](#常用命令)

---

## 快速开始

```bash
# 1. 创建 conda 环境
conda create -n yolo-game python=3.10 -y
conda activate yolo-game

# 2. 安装 PyTorch（CUDA 12.4）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 3. 安装依赖
pip install -r requirements.txt

# 4. 启动图形界面
python main.py gui
```

---

## 项目结构

```
game-collimation/
├── main.py                     # 项目统一入口
├── requirements.txt            # Python 依赖
│
├── config/
│   └── train_config.yaml       # 训练超参数配置
│
├── data/
│   ├── dataset.yaml            # YOLO 数据集配置（类别名、路径）
│   ├── classes.txt             # 类别名列表（labelImg 自动读取）
│   ├── raw/                    # 原始数据
│   │   ├── images/             #   原始截图
│   │   └── labels/             #   原始 YOLO 标注
│   ├── processed/              # 中间处理数据
│   └── dataset/                # 划分后的训练/验证/测试集
│       ├── images/  {train,val,test}/
│       └── labels/  {train,val,test}/
│
├── src/
│   ├── train.py                # 训练脚本
│   ├── predict.py              # 推理脚本（图片/视频/摄像头）
│   ├── screen_detect.py        # 实时屏幕检测（透明覆盖层）
│   ├── dataset.py              # 数据集管理工具
│   ├── utils.py                # 通用工具函数
│   └── gui_app.py              # 图形界面
│
├── scripts/
│   ├── prepare_data.py         # 数据集划分 & 检查
│   ├── label_converter.py      # 标注格式转换（VOC/COCO → YOLO）
│   └── extract_frames.py       # 视频帧提取
│
├── models/                     # 模型权重文件
└── runs/                       # 训练/推理输出
    ├── train/                  #   训练结果
    ├── predict/                #   推理结果
    └── logs/                   #   训练日志
```

---

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖：

| 库 | 用途 |
|---|---|
| `ultralytics` | YOLOv8 训练推理 |
| `torch` / `torchvision` | 深度学习框架 |
| `opencv-python` | 图像视频处理 |
| `mss` | 屏幕截取（高性能） |
| `PyYAML` | 配置文件解析 |
| `matplotlib` | 训练图表 |

### 检查环境

```bash
python main.py env
```

---

## 数据准备

### 标注工具

推荐使用 **labelImg**：

```bash
pip install labelImg
labelImg
```

启动后配置：
1. **Open Dir** → 选择 `data/raw/images/`
2. **Change Save Dir** → 选择 `data/raw/labels/`
3. 点击 **PascalVOC** 切换为 **YOLO** 格式

快捷键：

| 按键 | 功能 |
|------|------|
| `W` | 创建标注框 |
| `A` / `D` | 上一张 / 下一张 |
| `Ctrl + S` | 保存 |
| `Ctrl + D` | 复制上一张的框 |
| `Del` | 删除选中框 |

`data/raw/labels/classes.txt` 会在标注时自动生成，文件内容会被 `prepare` 命令同步到 `dataset.yaml`。

### 从视频提取帧

录制游戏视频后提取标注帧：

```bash
# 每1秒提取1帧
python main.py extract --video gameplay.mp4

# 每0.5秒提取1帧
python main.py extract --video gameplay.mp4 --interval 0.5

# 均匀提取100帧
python main.py extract --video gameplay.mp4 --count 100
```

### 划分数据集

```bash
# 划分 train/val/test（同时同步类别到 dataset.yaml）
python main.py prepare

# 仅检查数据集
python main.py check
```

---

## 训练模型

### 配置训练参数

编辑 [`config/train_config.yaml`](config/train_config.yaml)，关键参数：

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `epochs` | 训练轮数 | 数据少 200~300，数据多 80~100 |
| `batch` | 批大小 | 4GB 显存=4，8GB=8，12GB=16 |
| `imgsz` | 输入尺寸 | 1440p 游戏建议 960~1280 |
| `patience` | 早停耐心值 | 30（连续30轮不提升则停止） |
| `lr0` | 初始学习率 | 0.001 |

### 开始训练

```bash
# 使用默认配置训练
python main.py train

# 在 GUI 中训练
python main.py gui
```

训练输出保存在 `runs/train/game_char_train/`：
- `weights/best.pt` — 最佳模型
- `weights/last.pt` — 最后一轮模型
- `results.png` — 训练曲线图
- `confusion_matrix.png` — 混淆矩阵

---

## 模型推理

```bash
# 图片推理
python main.py predict --source test.jpg --show

# 视频推理并保存结果
python main.py predict --source video.mp4 --output result.mp4 --show

# 批量推理目录
python main.py predict --source data/test_images/

# 摄像头实时推理
python main.py predict --source 0 --show

# 使用自定义模型
python main.py predict --source test.jpg --model runs/train/game_char_train/weights/best.pt
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `yolov8n.pt` | 模型权重路径 |
| `--source` | `0` | 推理源（图片/视频/目录/摄像头ID） |
| `--conf` | `0.25` | 置信度阈值 |
| `--output` | — | 输出路径 |
| `--show` | — | 弹窗显示结果 |

---

## 实时屏幕检测

在游戏画面上叠加透明检测框，鼠标可穿透操作：

```bash
# 透明覆盖层模式（默认）
python main.py screen

# 窗口模式（调试用）
python main.py screen --window

# 自定义参数
python main.py screen --model best.pt --conf 0.4 --scale 0.5 --fps 60
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `yolov8n.pt` | 模型权重 |
| `--conf` | `0.25` | 置信度阈值 |
| `--scale` | `1.0` | 截图缩放（0.5=半分辨率加速） |
| `--fps` | `30` | 目标帧率上限 |
| `--monitor` | `1` | 显示器编号 |
| `--window` | — | 使用 OpenCV 窗口模式 |

按 **Esc** 退出。

---

## 图形界面

```bash
python main.py gui
```

| 标签页 | 功能 |
|--------|------|
| **训练** | 加载配置、设置参数、启动训练 |
| **数据集** | 划分数据集、检查完整性、统计信息、提取帧 |
| **推理** | 图片/视频/目录/摄像头推理 |
| **屏幕检测** | 配置透明覆盖层参数、启动实时检测 |
| **环境** | 查看 Python/PyTorch/CUDA 版本 |
| **清理** | 清理训练缓存/数据集/原始数据 |

---

## 常用命令

```bash
python main.py train                  # 训练模型
python main.py predict                # 模型推理
python main.py screen                 # 实时屏幕检测
python main.py prepare                # 划分数据集
python main.py check                  # 检查数据集
python main.py extract --video v.mp4  # 从视频提取帧
python main.py convert                # 标注格式转换
python main.py env                    # 检查运行环境
python main.py cleanup                # 清理缓存
python main.py gui                    # 启动图形界面
```

---

## 标注格式转换

支持 VOC XML 和 COCO JSON 转为 YOLO 格式：

```bash
# VOC XML → YOLO
python main.py convert --format voc2yolo \
    --label_dir data/voc_labels \
    --output data/raw/labels \
    --class_map '{"player":0,"enemy":1}'

# COCO JSON → YOLO
python main.py convert --format coco2yolo \
    --coco_json annotations.json \
    --output data/raw/labels
```

---

## 标注与训练完整流程

```
录制游戏视频 → 提取帧 → labelImg 标注 → 划分数据集 → 训练 → 推理/屏幕检测

  extract      labelImg    prepare       train      predict/screen
     ↓            ↓           ↓            ↓             ↓
  data/raw/   data/raw/   data/        runs/train/   实时检测框
  images/     labels/     dataset/     best.pt       覆盖在游戏上
```

---

## License

MIT
