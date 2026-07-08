"""Re-export best_model.pth -> ONNX with the legacy exporter (opset 12).

The v3 ONNX shipped with the project was produced by PyTorch 2.12's new dynamo
exporter at opset 18 (70 nodes, external-data weights, batched MatMul with the
adjacency matrix). That graph trips two separate bugs in RKNN-Toolkit 2.3.2's
constant folder / einsum-to-matmul rule, so it cannot be converted.

The first two model versions converted fine because they were exported at
opset 12 with the classic TorchScript exporter, producing a graph RKNN
accepts. This script reproduces that path from the .pth weights, yielding a
single-file ONNX that converts cleanly.

Run in the RKNN-Toolkit-2.3.2 conda env (has torch + onnxruntime).
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)
from model import SignLanguageModel  # noqa: E402

PTH = os.path.join(PROJECT_ROOT, 'best_model.pth')
OUT = os.path.join(HERE, 'best_model_opset12.onnx')


def num_classes_from_pth(state):
    """Infer class count from the final fc/Gemm weight in the checkpoint."""
    for k, v in state.items():
        if k.endswith('fc.weight'):
            return v.shape[0]
    raise RuntimeError('could not find fc.weight in checkpoint')


def main():
    state = torch.load(PTH, map_location='cpu', weights_only=True)
    n_cls = num_classes_from_pth(state)
    print(f'checkpoint fc -> {n_cls} classes')

    model = SignLanguageModel(num_classes=n_cls)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 6, 60, 26, 2)
    torch.onnx.export(
        model, dummy, OUT,
        opset_version=12,            # legacy opset that RKNN 2.3.2 handles
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,
        do_constant_folding=True,
    )
    print('exported ->', OUT, f'({os.path.getsize(OUT)} bytes)')

    # sanity: onnxruntime output dim
    import onnxruntime as ort
    s = ort.InferenceSession(OUT, providers=['CPUExecutionProvider'])
    print('output shape:', s.get_outputs()[0].shape)


if __name__ == '__main__':
    main()
