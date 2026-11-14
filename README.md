# PCLHD AG-ReID.v1 / AG-ReID.v2 Baseline Release

本发布版用于在 **两个训练阶段均关闭通道转换增强** 的条件下，对
D-AGVA 与三域匹配进行受控的 2×2 消融实验。代码只面向 AG-ReID.v1。

## 固定实验矩阵

| 实验名 | Stage 1 D-AGVA | Stage 2 三域匹配 | 两阶段通道增强 |
|---|---:|---:|---:|
| `baseline` | 关闭 | 关闭 | 关闭 |
| `dynamic` | 开启 | 关闭 | 关闭 |
| `three_domain` | 关闭 | 开启 | 关闭 |
| `full` | 开启 | 开启 | 关闭 |

其中，通道增强关闭是指训练代码不调用 `ChannelAdapGray`、
`ChannelExchange` 或 `Gray`。`ChannelRandomErasing` 属于空间随机擦除，
四组实验均保留。D-AGVA 只作用于 Stage 1 的 aerial training branch；
Stage 2 始终使用原始图像。三域匹配只作用于 Stage 2。

四组实验还统一使用以下设置：

- Stage 2 prototype memory：`CMhard`；
- Stage 2 的 `loss_ema` 不参与反向传播；
- EMA encoder 参数更新保留；
- ALL-memory 第二次优化保留；
- 默认使用 Stage 1 的 `model_best.pth.tar` 初始化 Stage 2；
- Stage 1 best 由修正 P/T/A 身份后的 G$\rightarrow$A Rank-1 选取。

## 数据目录

默认训练根目录为：

```text
/home/lab338/Jaxon/dataset/AGreid_train
├── aerial_modify/1/bounding_box_train/
└── ground_modify/1/bounding_box_train/
```

D-AGVA 还需要与 aerial images 一一对应的 SCHP 前景掩码，默认位置：

```text
/home/lab338/Jaxon/dataset/AGreid_train_SCHP/aerial_modify/1/bounding_box_train/
```

三域匹配使用发布包中的：

```text
view_scores/agreid_aerial_view_scores_foreground.csv
```

数据集、掩码、模型权重和训练日志均未包含在发布包中。

### AG-ReID.v1 身份解析

AG-ReID.v1 的官方身份由文件名中的 P、T、A 三段拼接得到，不能只使用
P 字段。发布版同时支持原始名称 `P...T...A...C...F...jpg` 和训练目录中
将 A/C 合并为 `C{A}{C}` 的改写名称。完整训练集必须解析出 199 个身份，
完整测试集必须解析出 189 个身份；数量不一致时程序会直接报错。训练过程
仍只使用 DBSCAN 伪标签，真实身份仅用于数据完整性审计和测试评价。

## 环境

推荐 Python 3.10。安装依赖：

```bash
pip install -r requirements.txt
```

## 先做配置检查

下面的命令不会启动训练，只打印四组实验实际调用的子命令：

```bash
python train_agreid.py --experiment baseline --run-mode full --dry-run
python train_agreid.py --experiment dynamic --run-mode full --dry-run
python train_agreid.py --experiment three_domain --run-mode full --dry-run
python train_agreid.py --experiment full --run-mode full --dry-run
```

输出中应明确显示：

```text
ChannelAdapGray: False (Stage 1 and Stage 2)
ChannelExchange/Gray: False (Stage 1 and Stage 2)
Stage 2 memory: CMhard
Stage 2 EMA loss included: False
```

## 批量运行四组实验

批量脚本只训练两份 Stage 1（D=0、D=1），再分别启动四个 Stage 2，
避免重复训练相同的 Stage 1。

```bash
cd /path/to/PCLHD_AGReID_DT_release_20261110
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python scripts/run_all_ablations.py \
  --gpu 0 \
  --experiments baseline dynamic three_domain full \
  --data-dir /home/lab338/Jaxon/dataset/AGreid_train \
  --logs-dir /home/lab338/Jaxon/PCLHD_AGReID_DT_release/logs_agreid_dt \
  --mask-dir /home/lab338/Jaxon/dataset/AGreid_train_SCHP/aerial_modify/1/bounding_box_train \
  --aerial-image-root /home/lab338/Jaxon/dataset/AGreid_train/aerial_modify/1/bounding_box_train \
  --dynamic-agva-strict-masks \
  --epochs 50 --iters 400 --seed 1 --workers 8 --print-freq 50
```

建议先在同一命令末尾添加 `--dry-run` 核查路径。

## 单独运行

完整方法（D=1, T=1）：

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 \
python train_agreid.py \
  --experiment full --run-mode full \
  --data-dir /home/lab338/Jaxon/dataset/AGreid_train \
  --logs-dir /home/lab338/Jaxon/PCLHD_AGReID_DT_release/logs_agreid_dt \
  --dynamic-agva-mask-dir /home/lab338/Jaxon/dataset/AGreid_train_SCHP/aerial_modify/1/bounding_box_train \
  --dynamic-agva-image-root /home/lab338/Jaxon/dataset/AGreid_train/aerial_modify/1/bounding_box_train \
  --dynamic-agva-strict-masks \
  --epochs 50 --iters 400 --seed 1 --workers 8 --print-freq 50
```

只运行 Stage 2 时，必须显式给出 Stage 1 checkpoint：

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0 \
python train_agreid.py \
  --experiment three_domain --run-mode stage2 \
  --stage1-checkpoint /path/to/stage1/model_best.pth.tar \
  --data-dir /home/lab338/Jaxon/dataset/AGreid_train \
  --logs-dir /home/lab338/Jaxon/PCLHD_AGReID_DT_release/logs_agreid_dt
```

## 日志与权重命名

Stage 1：

```text
logs_agreid_dt/agreid_dt_s1_d0_channel_off_seed1/1/model_best.pth.tar
logs_agreid_dt/agreid_dt_s1_d1_channel_off_seed1/1/model_best.pth.tar
```

Stage 2：

```text
logs_agreid_dt/agreid_dt_baseline_d0_t0_channel_off_s2_seed1/1/checkpoint.pth.tar
logs_agreid_dt/agreid_dt_dynamic_d1_t0_channel_off_s2_seed1/1/checkpoint.pth.tar
logs_agreid_dt/agreid_dt_three_domain_d0_t1_channel_off_s2_seed1/1/checkpoint.pth.tar
logs_agreid_dt/agreid_dt_full_d1_t1_channel_off_s2_seed1/1/checkpoint.pth.tar
```

完整流程默认通过 `--checkpoint-policy best` 使用 Stage 1 的
`model_best.pth.tar` 初始化 Stage 2。若需要固定使用 Stage 1 最后一个 epoch，
可显式传入 `--checkpoint-policy final`。需要注意，当前 best 依据正式测试集
G$\rightarrow$A Rank-1 选择；虽然 P/T/A 身份已修正，这种选择方式仍可能产生
test-set leakage，论文正式实验应明确披露或改用独立 validation split。

DBSCAN 的 `--eps` 默认值为 `0.3`，Stage 1、Stage 2 以及三域引擎均使用该
参数。非默认值会自动加入日志名，例如 `--eps 0.6` 生成 `_eps0.6` 后缀；
默认 `0.3` 不加后缀，从而保持既有日志路径不变。若某域找不到簇，引擎仍会
逐次增加 `0.05` 并打印每次调整以及每轮最终实际使用的 eps。

Stage 2 的联合（ALL）聚类可用 `--eps-all` 单独设置；未提供时继承
`--eps`，因此默认命令、日志名和数值行为均不变。仅当两者不同时，Stage 2
日志名才在现有 eps 后缀后追加 `_epsall{值}`，例如
`--eps 0.6 --eps-all 0.3` 对应 `_eps0.6_epsall0.3`。Stage 1 不接收
`--eps-all`，其日志名和权重查找路径不受影响。

### 快速 DBSCAN（显式启用）

默认 `--dbscan-backend sklearn` 保持原有子命令和结果不变。大规模稠密
预计算距离矩阵可显式改用分块实现：

```bash
python train_agreid.py \
  --dataset-key agreidv2 \
  --experiment baseline --run-mode full \
  --eps 0.6 \
  --dbscan-backend fast \
  --data-dir <data-dir> --logs-dir <logs-dir>
```

快速实现不改变 `eps`、`min_samples` 或任何训练超参，并复现 sklearn 的
簇编号与边界点归属。实现及 sklearn 1.8.0 源码语义核对见
`FAST_DBSCAN_NOTES.md`。

如需用真实 Stage 2 联合距离矩阵逐元素验证，可在训练时保存一个零基 epoch：

```bash
python train_agreid.py \
  --dataset-key agreidv2 \
  --experiment baseline --run-mode stage2 \
  --stage1-checkpoint <stage1-model-best> \
  --eps 0.6 --dbscan-backend fast \
  --dump-cluster-dist-epoch 1 \
  --data-dir <data-dir> --logs-dir <logs-dir>

python tools/verify_fast_dbscan.py \
  <stage2-output>/cluster_dist_all_epoch_1.npy \
  --eps 0.6 --min-samples 4
```

## 批量测试

```bash
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python scripts/test_all_checkpoints.py \
  --gpu 0 \
  --test-data-dir /home/lab338/Jaxon/dataset/AGreid_test \
  --logs-dir /home/lab338/Jaxon/PCLHD_AGReID_DT_release/logs_agreid_dt \
  --experiments baseline dynamic three_domain full
```

每个测试日志开头都会记录 D/T 开关、通道增强状态和 checkpoint 路径。
测试程序会输出 A$\rightarrow$G、G$\rightarrow$A，并保留现有脚本中的
G$\rightarrow$A+G 结果。论文主表应仅采用预先声明的正式评价协议；
G$\rightarrow$A+G 若非官方协议，应标为自定义补充实验。

## 本地验证

```bash
python -m unittest discover -s tests -v
python tools/verify_release.py
```

验证包括 2×2 开关、D-AGVA 仅在 Stage 1、三域匹配仅在 Stage 2、
CMhard/no-EMA-loss 固定设置、通道转换调用清除，以及发布包中不存在权重、
checkpoint、日志和 Python cache。

## 主要入口

- `train_agreid.py`：单组实验统一入口；
- `experiment_runner.py`：固定实验矩阵与子命令构造；
- `_train_agreid_engine.py`：Stage 1 与原始双域 Stage 2；
- `_train_agreid_three_domain_engine.py`：三域 Stage 2；
- `scripts/run_all_ablations.py`：四组消融批量训练；
- `scripts/test_all_checkpoints.py`：四组权重批量测试；
- `tests/test_release_configuration.py`：配置回归测试。

## AG-ReID.v2 baseline

AG-ReID.v2 目前只开放 `baseline`（D=0、T=0）的 Stage 1、Stage 2 和四个
官方测试协议。v1 的 2×2 消融入口、默认值和日志名保持不变。v2 的
`dynamic`、`three_domain`、`full` 会明确报错，因为尚未准备 v2 的 D-AGVA
掩码与三域视角分数。

### 目录要求

`--data-dir` 必须指向外层目录：

```text
<data-dir>/
├── exp1_aerial_to_cctv.txt
├── exp2_aerial_to_wearable.txt
├── exp4_cctv_to_aerial.txt
├── exp5_wearable_to_aerial.txt
└── AG-ReID.v2/
    ├── train_all/<ID目录>/*.jpg
    ├── query/<ID目录>/*.jpg
    └── gallery/<ID目录>/*.jpg
```

数据不会被移动、改名或复制。训练图像按文件名解析 P/T/A 仅用于完整性
断言；训练元组使用未标注占位符，优化过程只使用 DBSCAN 伪标签。C0 进入
IR 分支（`modal=2`），C2/C3 进入 RGB 分支（`modal=1`）；采样器相机号为
`int(A + C)`，与 v1 的高度档+相机定义一致。

### 预检

训练前先在 CPU 上核对 51,530 张训练图、807 个身份、各相机数量及四份
协议清单：

```bash
python tools/preflight_agreidv2.py --data-dir <data-dir>
```

任何图像数或身份数不一致都会打印实际数字并退出。

### 训练

完整 baseline（Stage 2 默认加载 Stage 1 `model_best.pth.tar`）：

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python train_agreid.py \
  --dataset-key agreidv2 \
  --experiment baseline \
  --run-mode full \
  --checkpoint-policy best \
  --data-dir <data-dir> \
  --logs-dir <logs-dir>
```

v2 默认日志目录名带 `agreidv2_dt_` 前缀，不会与 v1 冲突。批量入口也接受
数据集键，但 v2 必须只选择 baseline：

```bash
python scripts/run_all_ablations.py \
  --dataset-key agreidv2 \
  --experiments baseline \
  --data-dir <data-dir> \
  --logs-dir <logs-dir>
```

### 四协议测试

以下命令依次测试 A→C、A→W、C→A、W→A，并输出 Rank-1/5/10、mAP、
mINP，同时生成每协议一行的 CSV：

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
python examples/PCLHD_AGReIDv2_test.py \
  --data-dir <data-dir> \
  --resume <stage2-checkpoint.pth.tar> \
  --seed 1 \
  --output-csv <results.csv>
```

批量测试脚本默认选择最终 `checkpoint.pth.tar`；需要最佳权重时显式传入
`--checkpoint-name model_best.pth.tar`：

```bash
python scripts/test_all_checkpoints.py \
  --dataset-key agreidv2 \
  --experiments baseline \
  --test-data-dir <data-dir> \
  --logs-dir <logs-dir>
```
