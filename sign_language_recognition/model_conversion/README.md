# 模型转换 (ONNX → RKNN)

把训练好的 `best_model.onnx` 转换成 RK3588 NPU 用的 `.rknn`，并对照训练流程
验证转换无误。独立于 `rk3588_deploy/`（部署代码），这里只管"把模型变成 .rknn"。

## 预处理版本（2026-06-06 起）

`dataset.py` 的预处理改版：直接读**原始绝对坐标** `tracking.csv`（不再读派生的
`tracking_relative_to_right_palm.csv`），在 `load_session` 内部把**两只手统一**
锚定到右手首帧 palm 坐标系，并丢弃右手首次出现之前的帧。

- `gen_golden.py` 已同步改用 `tracking.csv`（否则会双重坐标变换）。
- **右手处理与旧版数学等价；左手坐标系完全不同**（旧版相对自己手掌、新版相对右手）。
  因此模型**必须用新版 dataset.py 重新训练**后再转换，否则双手类（如「忙」）会预测错乱。
- C 端实时采集（若仍维护）也需同步成"两手统一锚定到右手 palm"，否则与训练分布不一致。

## 当前模型（第三版，15 类）

- 输入 `[1, 6, 60, 26, 2]`（6 通道 pos+vel、60 帧、26 节点、2 手）—— 历代不变
- 输出 `[1, 15]` —— **15 类**（类别顺序 = 数据集目录名的 `sorted()`）：

  | 索引 | 类别 | 索引 | 类别 | 索引 | 类别 |
  |---|---|---|---|---|---|
  | 0 | 伤心 | 5 | 大家好 | 10 | 感动 |
  | 1 | 你好 | 6 | 害怕 | 11 | 未知 |
  | 2 | 去 | 7 | 工作 | 12 | 来 |
  | 3 | 吃 | 8 | 开心 | 13 | 现在 |
  | 4 | 名字 | 9 | 忙 | 14 | 看 |

  > 注意：加入新类后，`sorted()` 顺序整体重排（新类按拼音/笔画插进中间），
  > 索引和上一版不同。部署端标签表务必按 `output/labels.txt` 全量更新。

## opset 重导出（第三版必做）

第三版的 `best_model.onnx` 是 PyTorch 2.12 的新 dynamo 导出器在 **opset 18**
下导出的（70 节点、权重存在 `best_model.onnx.data` 外部文件），它会触发
RKNN-Toolkit 2.3.2 常量折叠器的 bug（`MatMul` 与邻接矩阵相乘时 shape 广播报错），
**无法直接转换**。

解决：用 `reexport_onnx.py` 从 `best_model.pth` 以 **opset 12 + 经典 TorchScript
导出器**重新导出（产出 `best_model_opset12.onnx`，82 节点、单文件、与前两版同构）。
`gen_golden.py` 和 `convert.py` 都已指向这个重导出文件。

> 若以后训练侧把导出 opset 改回 12（修改 `train.py` 的 `opset_version`），
> 可直接用 `best_model.onnx`，跳过重导出这步。

## 用法

在 `RKNN-Toolkit-2.3.2` conda 环境里运行（含 numpy/scipy/onnxruntime/rknn-toolkit2/torch）：

```bash
conda activate RKNN-Toolkit-2.3.2
cd model_conversion

# 0) 从 .pth 重导出 opset-12 ONNX (绕过 RKNN 2.3.2 的 opset18 折叠 bug)
python reexport_onnx.py

# 1) 生成 golden 参照 (复用 dataset.py 预处理 + ONNX 推理)，每类各一条样本
python gen_golden.py

# 2) ONNX → RKNN (rk3588, fp16 非量化) + 用 golden 在模拟器上校验
python convert.py
```

产物在 `output/`：

```
output/
├── sign_rk3588.rknn        转换好的 NPU 模型
├── labels.txt              9 类标签 (索引 / 汉字)，供部署端核对
└── golden/<类别>/          每类的 golden_input.bin + golden_logits.txt
```

## 验证标准

`convert.py` 对每个 golden 样本比较 ONNX 与 RKNN 模拟器输出：
- 余弦相似度 > 0.999
- argmax 一致

全部通过才算转换成功。

## 部署

把 `output/sign_rk3588.rknn` 拷到 `rk3588_deploy/model/sign_rk3588.rknn`。
类别数变化时，部署端 `src/labels.c` 也要同步更新成 9 类（见 `output/labels.txt`）。
