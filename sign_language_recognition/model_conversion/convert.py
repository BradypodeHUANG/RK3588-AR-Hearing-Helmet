"""Convert best_model.onnx -> output/sign_rk3588.rknn and validate.

Full 5D model conversion (input [1,6,60,26,2], output [1,N]). fp16, no
quantization — the net is tiny and fp16 preserves ONNX accuracy. Validates the
RKNN simulator against every per-class golden produced by gen_golden.py
(cosine > 0.999 and argmax match).

Run in the RKNN-Toolkit-2.3.2 conda env, after gen_golden.py.
"""
import os
import sys
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
# Use the opset-12 re-export (see reexport_onnx.py). The project's
# best_model.onnx is an opset-18 dynamo export that RKNN 2.3.2 cannot fold;
# the re-export is structurally identical to the weights but converts cleanly.
ONNX = os.path.join(HERE, 'best_model_opset12.onnx')
OUT_DIR = os.path.join(HERE, 'output')
RKNN_OUT = os.path.join(OUT_DIR, 'sign_rk3588.rknn')
GOLDEN = os.path.join(OUT_DIR, 'golden')
PLATFORM = 'rk3588'


def class_names():
    data_dir = os.path.join(PROJECT_ROOT, 'dataset')
    return sorted(d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d)))


def load_logits(path):
    for line in open(path):
        if line.startswith('logits'):
            return np.array([float(v) for v in line.split()[1:]], dtype=np.float32)
    raise ValueError('no logits in ' + path)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def write_labels(names):
    with open(os.path.join(OUT_DIR, 'labels.txt'), 'w') as f:
        for i, n in enumerate(names):
            f.write(f'{i} {n}\n')
    print(f'wrote labels.txt ({len(names)} classes)')


def main():
    from rknn.api import RKNN
    names = class_names()
    os.makedirs(OUT_DIR, exist_ok=True)
    write_labels(names)

    rknn = RKNN(verbose=False)
    print('--> config')
    rknn.config(target_platform=PLATFORM, optimization_level=3)
    print('--> load_onnx')
    if rknn.load_onnx(model=ONNX) != 0:
        print('load_onnx failed'); sys.exit(1)
    print('--> build (fp16, no quant)')
    if rknn.build(do_quantization=False) != 0:
        print('build failed'); sys.exit(1)
    print('--> export', RKNN_OUT)
    if rknn.export_rknn(RKNN_OUT) != 0:
        print('export failed'); sys.exit(1)

    print('--> init_runtime (simulator)')
    if rknn.init_runtime() != 0:
        print('init_runtime failed'); sys.exit(1)

    print('--> validate against golden')
    n_dim = len(names)
    all_ok = True
    checked = 0
    for cls in names:
        gin = os.path.join(GOLDEN, cls, 'golden_input.bin')
        glog = os.path.join(GOLDEN, cls, 'golden_logits.txt')
        if not (os.path.exists(gin) and os.path.exists(glog)):
            print(f'  {cls}: no golden, skip')
            continue
        x = np.fromfile(gin, dtype='<f4').reshape(1, 6, 60, 26, 2)
        gl = load_logits(glog)
        out = np.array(rknn.inference(inputs=[x])[0]).reshape(-1)[:n_dim]
        cos = cosine(out, gl)
        a_r, a_g = int(out.argmax()), int(gl.argmax())
        ok = cos > 0.999 and a_r == a_g
        all_ok &= ok
        checked += 1
        print(f'  {cls:6s}: cosine={cos:.6f} argmax rknn={a_r} golden={a_g}  '
              f'{"OK" if ok else "FAIL"}')

    rknn.release()
    print(f'\n{checked} classes checked. VALIDATION {"PASS" if all_ok else "FAIL"}')
    print('rknn model ->', RKNN_OUT)
    sys.exit(0 if all_ok else 2)


if __name__ == '__main__':
    main()
