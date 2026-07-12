# JPG IQ Evaluator

**基于成品 JPG 图像的相机画质自动化评测系统**

替代人工主观评图，用于手机相机成品输出效果回归测试，大幅降低测试人力成本。

---

## 背景

传统相机画质评测依赖 RAW 原始数据与专业测试图卡（ISO 12233、ColorChecker 24 色卡等），操作门槛高、人工成本大。本项目另辟蹊径——**直接以成品 JPG 图像为输入**，结合 EXIF 曝光元数据，利用计算机视觉算法实现全流程自动化客观评测。

适用于：手机相机成品效果回归测试、多算法版本横向对比、镜头模组批次抽检。

---

## 评测指标

### 客观画质指标（OpenCV 传统视觉）

| 指标 | 方法 | 说明 |
|------|------|------|
| **SFR 锐度 MTF50** | ISO 12233 倾斜边缘法 | 边缘扩散函数 → FFT → 调制传递函数，插值求 MTF50 频率 |
| **色彩准确度 ΔE** | 24 色卡自动检测 + CIE Lab | 6x4 网格拟合定位色块，CIELAB 色差公式，输出均值/最大值/中位数 |
| **信噪比 SNR** | 平坦区域统计 | 20×log₁₀(均值/标准差)，中心 ROI 计算 |
| **纹理噪声** | 高通滤波能量 | 拉普拉斯算子提取高频分量标准差 |
| **暗角** | 四角/中心亮度比 | 四角 ROI 均值 / 中心 ROI 均值，输出比值与 dB |
| **JPG 块效应** | 8×8 DCT 块边界检测 | 块边界灰度差 / 块内部灰度差比值，量化压缩伪影严重程度 |
| **光学畸变** | 棋盘格角点检测 | 理想网格 vs 实际角点偏移百分比 |

### AI 缺陷检测（YOLOv8 + 传统方法辅助）

| 缺陷类型 | 检测方式 |
|----------|----------|
| **色散 / 紫边** | HSV 色相聚类（传统） + YOLOv8 模型 |
| **脏点 / 污渍** | YOLOv8 目标检测 |
| **模糊 / 失焦** | 拉普拉斯方差法（传统） + YOLOv8 |
| **鬼影 / 眩光** | YOLOv8 目标检测 |

YOLOv8 模型可选加载，模型不存在时自动降级为纯传统方法。

### EXIF 联动分析

自动提取 JPG EXIF 元数据（ISO、焦距、光圈、快门速度），支持散点图分析 **ISO / 焦距对画质（SNR / ΔE / 暗角）的影响趋势**，量化关键参数与画质的关联关系。

---

## 项目架构

```
Auto_Evaluator/
│
├── main.py                  # 统一入口（CLI / GUI / A/B对比）
│
├── config/
│   └── config.yaml          # 指标开关、场景分组、缺陷阈值、图表参数
│
├── core/                    # 核心算法
│   ├── image_loader.py      # JPG加载 + piexif EXIF提取
│   ├── roi_detector.py      # 24色卡/ISO12233/灰阶卡ROI自动定位
│   ├── iq_metrics.py        # MTF50/ΔE/SNR/暗角/畸变/块效应
│   ├── defect_detect.py     # YOLOv8 + 传统方法缺陷检测
│   ├── batch_processor.py   # ThreadPoolExecutor多线程批量调度
│   └── comparison.py        # A/B版本画质指标对比引擎
│
├── gui/                     # PyQt5图形界面
│   ├── main_window.py       # 四标签主窗口
│   ├── batch_tab.py         # 批量评测标签
│   ├── compare_tab.py       # 版本对比标签
│   ├── exif_analysis_tab.py # EXIF联动分析标签
│   └── report_tab.py        # PDF报表生成标签
│
├── visualization/
│   ├── plot_drawer.py       # 雷达图/折线图/散点图/柱状图
│   └── pdf_generator.py     # ReportLab标准化PDF报表
│
├── utils/
│   ├── file_io.py           # 目录遍历、CSV读写
│   ├── exif_utils.py        # EXIF解析辅助
│   └── common.py            # 图像归一化、颜色空间转换
│
├── models/
│   └── yolov8_defect.pt     # 预训练缺陷检测模型（占位）
│
├── data/
│   ├── jpg_input/           # 按场景类型分文件夹存放JPG
│   ├── test/                # 29个品牌手机实拍测试图
│   └── meta_info.csv        # 补充元数据
│
└── output/                  # 自动生成结果
    ├── metrics_csv/         # 批量结果CSV
    ├── charts/              # 可视化图表
    ├── defect_vis/          # 缺陷检测标注图
    └── report/              # PDF评测报表
```

---

## 使用方式

支持三种运行模式：

**CLI 评测模式**
直接指定 JPG 输入目录，自动完成全流程并输出 CSV + 图表 + PDF。

**版本对比模式**
传入新旧两个版本的指标 CSV，自动对齐文件名、计算 Δ 变化、统计胜出方。

**GUI 交互模式**
PyQt5 四标签界面：批量评测 / 版本对比 / EXIF 分析 / 报表生成，所见即所得。

---

## 关键技术选型

| 技术 | 用途 |
|------|------|
| OpenCV 5.x | 图像加载、颜色空间转换、边缘检测、形态学操作、棋盘格角点 |
| scikit-image | CIE Lab 颜色空间转换、参考色差计算 |
| piexif | EXIF 元数据提取（ISO、焦距、光圈、快门） |
| ultralytics (YOLOv8) | 缺陷目标检测（色散、脏点、模糊、鬼影） |
| PyQt5 | 可视化交互界面 |
| ReportLab | 标准化 PDF 报表生成 |
| Matplotlib + Seaborn | 雷达图、散点图、折线图、柱状图 |

---

## 输出产物

```
output/
├── metrics_csv/
│   └── batch_results.csv      ← 每张图的全部指标（25+列）
├── charts/
│   ├── radar_comparison.png   ← 多分组雷达对比图
│   ├── snr_db_line.png        ← SNR折线图
│   ├── delta_e_mean_line.png  ← ΔE折线图
│   └── defect_bar.png         ← 缺陷统计柱状图
├── defect_vis/
│   └── *_defects.jpg          ← 缺陷检测框标注图
└── report/
    └── iq_evaluation_report.pdf ← 完整标准化评测报表
```
