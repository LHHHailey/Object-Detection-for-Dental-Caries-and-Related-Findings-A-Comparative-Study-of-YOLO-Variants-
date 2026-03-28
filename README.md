# 龋齿检测（YOLOv8）完整项目

基于 [AndreyGermanov/yolov8_caries_detector](https://github.com/AndreyGermanov/yolov8_caries_detector) 的牙齿龋病/洞/裂纹检测：数据转换、训练、推理与简易 Web 服务。

## 本仓库里有什么 / 没有什么（重要）

为减小体积，**本仓库通常只包含 `yolo_dataset/data.yaml`（请保持此相对路径，勿放在仓库根目录）**，不包含 `yolo_dataset/**/images` 与 `labels`，也**不包含**原始 `dataset/` 或大型 `.tar`。**不能直接训练**：必须先在本机准备好原始数据，再运行转换脚本生成完整 `yolo_dataset/`。

**推荐流程概览：**

1. 自备 DentalAI 风格数据（见下文「数据集来源」），在项目根目录得到 `dataset/train|valid|test/{img,ann}` 与 `dataset/meta.json`。  
2. 安装依赖后执行：`python run_convert.py`  
3. 脚本会**清空并重建** `yolo_dataset/`，写入 `images/`、`labels/` 以及新的 `data.yaml`（类别与仓库里提交的 `data.yaml` 一致；路径由脚本生成）。  
4. 再运行 `run_yolo_compare.ps1` 或自行调用 Ultralytics 训练。

若你使用 `run_all.ps1` 从 `.tar` 解压，会自动完成「解压 → `dataset/` → `run_convert.py`」。

### `.gitignore` 传不上 GitHub？

网页 **Upload files** 可能提示以 `.` 开头的文件为 *hidden* 无法上传。可以任选其一：

1. 在 GitHub 网页：**Add file → Create new file**，文件名手动输入 **`.gitignore`**，把本地 `.gitignore` 内容粘贴进去后提交。  
2. 上传本仓库中的 **`gitignore.txt`**，克隆到本地后执行：  
   `Rename-Item gitignore.txt .gitignore`（PowerShell）或手动重命名。  
3. 使用 **Git 命令行**：`git add .gitignore && git commit`。

**说明：** 默认的忽略规则里包含 `logs/`、`runs/`。若你**故意要把训练日志或 `runs` 一并推到 GitHub**，请在生成 `.gitignore` 前删掉对应行，或改用 Release/网盘存放大文件。

## 目录说明

| 路径 | 说明 |
|------|------|
| `dataset/` | 原始数据（`train`/`valid`/`test` 各含 `img`+`ann`，及 `meta.json`）——**一般需自备，不在精简仓库中** |
| `yolo_dataset/` | `run_convert.py` 生成的 YOLO 格式与 `data.yaml`；**精简仓库可能仅有 `data.yaml` 作参考** |
| `run_convert.py` | Supervisely 风格 JSON → YOLO txt（等价于 `convert.ipynb`） |
| `run_all.ps1` | 一键：解压 tar → 安装依赖 → 转换 → 可选烟测训练/预测/Web |
| `train_yolo_compare.py` | 多版本 YOLO 统一超参对比训练 |
| `run_yolo_compare.ps1` | 对比训练封装（写日志） |
| `object_detector.py` + `index.html` | 本地 Web：`http://127.0.0.1:8080` |
| `best.pt` | 默认推理权重（可替换为你训练得到的 `weights/best.pt`） |

## 环境

- Python 3.10+（推荐；需与 `torch` 轮子匹配）
- Windows 自带 `tar` 用于解压 `.tar` 数据包

```powershell
cd <本仓库根目录>   # 与 run_convert.py 同级
python -m pip install -r requirements.txt
```

## 一键流程（推荐）

将完整数据包（`.tar`，顶层含 `train/`、`valid/`、`test/`、`meta.json` 等）放在本机任意路径，例如 `C:\data\dentalai_sample.tar`：

```powershell
cd <本仓库根目录>
.\run_all.ps1 -TarPath "C:\data\dentalai_sample.tar"
```

已有 `dataset\` 时：

```powershell
.\run_all.ps1 -SkipExtract
```

参数：` -SkipTrain` ` -SkipPredict` ` -SkipWeb` ` -Device 0`（GPU）

日志：`logs\run_all_*.log`

## 手动步骤

1. 准备 `dataset/`（与上表结构一致）。**仅克隆到 `yolo_dataset/data.yaml` 时，此步为必须；**不可在未转换的情况下直接训练。
2. `python run_convert.py` → 生成完整 `yolo_dataset/`（含 `images/`、`labels/` 与 `data.yaml`）。
3. 训练（与 `train.ipynb` 一致，示例）：

```python
from ultralytics import YOLO
import os
model = YOLO("yolov8m.pt")
model.train(
    data=os.path.join(os.getcwd(), "yolo_dataset", "data.yaml"),
    epochs=30,
    imgsz=640,
    batch=16,
    device="cpu",  # 或 0
)
```

4. 最优权重一般在：`runs/detect/train/weights/best.pt`（名称随 `project`/`name` 变化）。复制到项目根目录 `best.pt` 即可被 Web 与默认推理使用。

## Web 服务

```powershell
cd <本仓库根目录>
python object_detector.py
```

浏览器打开 `http://127.0.0.1:8080` 上传图片。  
指定权重（环境变量）：

```powershell
$env:YOLO_MODEL_PATH = "runs\detect\train\weights\best.pt"
python object_detector.py
```

API：`POST /detect`，表单字段名 `image_file`，返回 JSON 框数组。当前示例接口仅返回 **Caries（类别 id 0）** 的框。

## 多模型对比训练

需先存在 `yolo_dataset\data.yaml`：

```powershell
.\run_yolo_compare.ps1 -Device cpu -Epochs 100 -Batch 16
```

结果与汇总：`runs\compare\_compare_logs\summary_all.json`。  
监控：`tensorboard --logdir runs`（另开终端）。

## 数据集来源

DentalAI 等见原仓库说明：[datasetninja.com/dentalai](https://datasetninja.com/dentalai)（与上游 README 一致）。

## 许可

以仓库内 `LICENSE` / `LICENSE.md` 为准。
