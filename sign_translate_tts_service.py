#!/usr/bin/env python3
import argparse
import json
import os
import sys
import wave

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def build_tts(args):
    import sherpa_onnx as so

    matcha_cls = getattr(so, "OfflineTtsMatchaModelConfig", None)
    model_cls = getattr(so, "OfflineTtsModelConfig", None)
    config_cls = getattr(so, "OfflineTtsConfig", None)
    tts_cls = getattr(so, "OfflineTts", None)
    if not all([matcha_cls, model_cls, config_cls, tts_cls]):
        raise RuntimeError("Required sherpa_onnx OfflineTts classes are unavailable")

    matcha_attempts = [
        dict(
            acoustic_model=args.acoustic_model,
            vocoder=args.vocoder,
            lexicon=args.lexicon,
            tokens=args.tokens,
            data_dir="",
            dict_dir="",
            noise_scale=1.0,
            length_scale=1.0,
        ),
        dict(
            acoustic_model=args.acoustic_model,
            vocoder=args.vocoder,
            lexicon=args.lexicon,
            tokens=args.tokens,
            data_dir="",
            dict_dir="",
        ),
        dict(
            acoustic_model=args.acoustic_model,
            vocoder=args.vocoder,
            lexicon=args.lexicon,
            tokens=args.tokens,
        ),
    ]

    matcha_cfg = None
    matcha_errors = []
    for kwargs in matcha_attempts:
        try:
            matcha_cfg = matcha_cls(**kwargs)
            break
        except TypeError as ex:
            matcha_errors.append(str(ex))
    if matcha_cfg is None:
        raise RuntimeError("Failed to construct OfflineTtsMatchaModelConfig: " + " | ".join(matcha_errors))

    model_attempts = [
        dict(matcha=matcha_cfg, provider="cpu", num_threads=2, debug=False),
        dict(matcha=matcha_cfg, provider="cpu", num_threads=2),
        dict(matcha=matcha_cfg, provider="cpu"),
        dict(matcha=matcha_cfg),
    ]

    model_cfg = None
    model_errors = []
    for kwargs in model_attempts:
        try:
            model_cfg = model_cls(**kwargs)
            break
        except TypeError as ex:
            model_errors.append(str(ex))
    if model_cfg is None:
        raise RuntimeError("Failed to construct OfflineTtsModelConfig: " + " | ".join(model_errors))

    config_attempts = [
        dict(model=model_cfg, rule_fsts=args.rule_fsts, max_num_sentences=1),
        dict(model=model_cfg, rule_fsts=args.rule_fsts),
        dict(model=model_cfg),
    ]

    config = None
    config_errors = []
    for kwargs in config_attempts:
        try:
            config = config_cls(**kwargs)
            break
        except TypeError as ex:
            config_errors.append(str(ex))
    if config is None:
        raise RuntimeError("Failed to construct OfflineTtsConfig: " + " | ".join(config_errors))

    try:
        tts = tts_cls(config)
    except TypeError:
        tts = tts_cls(config=config)

    return tts


def generate_audio(tts, text):
    attempts = [
        {"text": text, "sid": 0, "speed": 1.0},
        {"text": text, "speaker_id": 0, "speed": 1.0},
        {"text": text, "sid": 0},
        {"text": text, "speaker_id": 0},
        {"text": text},
    ]

    last_error = None
    for kwargs in attempts:
        try:
            return tts.generate(**kwargs)
        except Exception as ex:  # pragma: no cover
            last_error = ex
    raise last_error or RuntimeError("TTS generation failed")


def normalize_samples(samples):
    if np is not None:
        arr = np.asarray(samples).reshape(-1)
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, -1.0, 1.0)
            arr = (arr * 32767.0).astype(np.int16)
        else:
            arr = arr.astype(np.int16, copy=False)
        return arr.tobytes()

    raw = list(samples)
    data = bytearray()
    for sample in raw:
        value = float(sample)
        if value > 1.0 or value < -1.0:
            value = max(-32768.0, min(32767.0, value))
            ivalue = int(value)
        else:
            ivalue = int(max(-1.0, min(1.0, value)) * 32767.0)
        data.extend(int(ivalue).to_bytes(2, byteorder="little", signed=True))
    return bytes(data)


def write_wave(path, audio):
    sample_rate = getattr(audio, "sample_rate", 22050)
    samples = getattr(audio, "samples", None)
    if samples is None and isinstance(audio, tuple) and len(audio) >= 2:
        samples, sample_rate = audio[0], audio[1]
    if samples is None:
        raise RuntimeError("Unsupported TTS audio result: missing samples")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(normalize_samples(samples))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-model", required=True)
    parser.add_argument("--vocoder", required=True)
    parser.add_argument("--lexicon", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--rule-fsts", required=True)
    args = parser.parse_args()

    try:
        tts = build_tts(args)
        # Warm the model once so first real request is faster.
        try:
            _ = generate_audio(tts, "你好")
        except Exception:
            pass
        emit({"ok": True, "type": "ready"})
    except Exception as ex:
        emit({"ok": False, "type": "init_error", "error": str(ex)})
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as ex:
            emit({"ok": False, "error": f"invalid_json: {ex}"})
            continue

        cmd = req.get("cmd", "synthesize")
        if cmd == "shutdown":
            emit({"ok": True, "type": "shutdown"})
            return 0
        if cmd != "synthesize":
            emit({"ok": False, "error": f"unknown_cmd: {cmd}"})
            continue

        text = str(req.get("text", "")).strip()
        output = str(req.get("output", "")).strip()
        if not text or not output:
            emit({"ok": False, "error": "missing_text_or_output"})
            continue

        try:
            audio = generate_audio(tts, text)
            write_wave(output, audio)
            emit({"ok": True, "output": output})
        except Exception as ex:
            emit({"ok": False, "error": str(ex)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
