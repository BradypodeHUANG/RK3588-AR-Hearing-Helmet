"""Generate golden references for model conversion verification.

Auto-discovers all dataset classes (sorted() = label order), picks one session
per class, reproduces dataset.py preprocessing to build the [1,6,60,26,2]
tensor, runs ONNX, and dumps per-class golden tensors + logits. convert.py
then checks the RKNN model reproduces these.

Run in the RKNN-Toolkit-2.3.2 conda env.
"""
import os
import sys
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PROJECT_ROOT)
import dataset as ds  # noqa: E402  (reuse the exact training preprocessing)

DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset')
# opset-12 re-export (see reexport_onnx.py); same weights, RKNN-convertible.
ONNX = os.path.join(HERE, 'best_model_opset12.onnx')
OUT = os.path.join(HERE, 'output', 'golden')
# dataset.py now reads the RAW absolute-coordinate CSV and does the
# right-palm anchoring itself inside load_session. Feeding the already-relative
# CSV here would double-apply the transform, so use tracking.csv.
CSV_NAME = 'tracking.csv'


def class_names():
    return sorted(d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d)))


def first_session_csv(cls):
    """First session CSV for a class, or None."""
    pattern = os.path.join(DATA_DIR, cls, '*', CSV_NAME)
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def build_input(csv_path):
    # dataset.py preprocessing is now two-stage: deterministic
    # _preprocess_one_hand (normalize+resample) then _finalize_one_hand
    # (augment+velocity). Mirror the val/no-augment path exactly.
    # load_session reads the raw absolute CSV and anchors to the first right palm.
    right, left, _, _ = ds.load_session(csv_path, skip_first_n=0)
    rc = ds._preprocess_one_hand(right, ds.TARGET_T)
    lc = ds._preprocess_one_hand(left, ds.TARGET_T)
    rf = ds._finalize_one_hand(rc, ds.TARGET_T, False)
    lf = ds._finalize_one_hand(lc, ds.TARGET_T, False)
    x = np.stack([rf, lf], axis=-1)[None]   # [1,6,60,26,2]
    return x.astype(np.float32)


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main():
    import onnxruntime as ort
    names = class_names()
    print(f'classes ({len(names)}): {names}')

    sess = ort.InferenceSession(ONNX, providers=['CPUExecutionProvider'])
    in_name = sess.get_inputs()[0].name
    out_dim = sess.get_outputs()[0].shape[-1]
    print(f'ONNX output dim = {out_dim}')
    if out_dim != len(names):
        print(f'WARNING: ONNX output {out_dim} != #classes {len(names)}')

    os.makedirs(OUT, exist_ok=True)
    n_ok = 0
    for idx, cls in enumerate(names):
        csv = first_session_csv(cls)
        if not csv:
            print(f'  [{idx}] {cls}: no session found, skip')
            continue
        x = build_input(csv)
        logits = sess.run(None, {in_name: x})[0][0]
        probs = softmax(logits)
        pred = int(np.argmax(logits))
        mark = 'OK' if pred == idx else f'!! pred={pred}({names[pred]})'
        print(f'  [{idx}] {cls}: argmax={pred} conf={probs[pred]:.4f}  {mark}')

        d = os.path.join(OUT, cls)
        os.makedirs(d, exist_ok=True)
        x.astype('<f4').tofile(os.path.join(d, 'golden_input.bin'))
        with open(os.path.join(d, 'golden_logits.txt'), 'w') as f:
            f.write('logits ' + ' '.join(f'{v:.6f}' for v in logits) + '\n')
            f.write('argmax %d\n' % pred)
            f.write('expected %d\n' % idx)
        n_ok += 1

    print(f'wrote golden for {n_ok}/{len(names)} classes -> {OUT}')


if __name__ == '__main__':
    main()
