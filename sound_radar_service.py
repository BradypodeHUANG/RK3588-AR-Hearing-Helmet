import csv
import json
import math
import os
import queue
import socket
import sys
import time
from collections import deque

import librosa
import numpy as np
import scipy.fftpack as fft
import scipy.signal as signal
import sounddevice as sd
from rknnlite.api import RKNNLite


MODEL_PATH = "yamnet_rk3588_int8.rknn"
CLASS_MAP_PATH = "yamnet_class_map.csv"
RADAR_HOST = "127.0.0.1"
RADAR_PORT = 19091

RAW_SAMPLE_RATE = 48000
RAW_CHANNELS = 8
MODEL_SAMPLE_RATE = 16000
MIC_CHANNEL = 7

WINDOW_DURATION = 0.975
STEP_DURATION = 0.25
WINDOW_SAMPLES = int(MODEL_SAMPLE_RATE * WINDOW_DURATION)
STEP_SAMPLES = int(MODEL_SAMPLE_RATE * STEP_DURATION)
RAW_BLOCKSIZE = int(RAW_SAMPLE_RATE * STEP_DURATION)

HORN_THRESHOLD = 0.20
DOG_THRESHOLD = 0.30
HORN_LIKE_THRESHOLD = 0.12
MIN_AI_RMS = 0.0010
OUTPUT_INTERVAL_SEC = 0.25
SCORE_MEMORY_DECAY = 0.65
SCORE_QUIET_DECAY = 0.30
FIXED_AUDIO_GAIN = float(os.getenv("SOUND_RADAR_GAIN", "40.0"))
DEBUG_PRINT_EVERY_N_BLOCKS = 10
TOP_K = 5
TOP_K_SCORE_THRESHOLD = 0.10

MIC_DISTANCE = 0.040
SPEED_OF_SOUND = 343.0
DOA_CHANNELS = (0, 1, 2, 3, 4, 5)
MIC_ARRAY_RADIUS = MIC_DISTANCE
DOA_FRAME_SIZE = 1024
DOA_HOP_SIZE = 256
DOA_SEARCH_SEC = 0.65
DOA_MIN_FRAME_PEAK = 0.0010
DOA_MIN_CONFIDENCE = 0.06
DOA_WEAK_CONFIDENCE = 0.04
DOA_STRONG_CONFIDENCE = 0.35
DOA_CLUSTER_DEG = 30.0
DOA_MAX_STABLE_SPREAD_DEG = 25.0
DOA_MIN_STABLE_FRAMES = 1
DOA_HISTORY_SIZE = 3
DOA_DIRECT_SOUND_WINDOW_MS = 55.0
DOA_ECHO_LOOKAHEAD_MS = 140.0
DOA_MIN_DIRECT_ECHO_RATIO = 0.40
DOA_DIRECTION_RESET_DEG = 55.0
DOA_FAST_ADAPT_DEG = 25.0
DOA_RECENT_PRIORITY_SEC = 0.35
DOA_OUTLIER_REJECT_DEG = 45.0
DOA_OUTLIER_CONFIRM_DEG = 20.0
DOA_OUTLIER_CONFIRM_COUNT = 2
DOA_OUTLIER_ACCEPT_CONFIDENCE = 0.45
DOA_OUTLIER_ACCEPT_FRAMES = 5
DOA_OUTLIER_ACCEPT_SPREAD_DEG = 10.0
DOA_COLD_START_CONFIDENCE = 0.08
DOA_COLD_START_FRAMES = 1
DOA_COLD_START_SPREAD_DEG = 12.0
DOA_COLD_START_CONFIRM_COUNT = 1
DOA_COLD_START_CONFIRM_DEG = 15.0
SRP_ANGLE_STEP_DEG = 2

HORN_CLASS_NAMES = (
    "Vehicle horn, car horn, honking",
    "Air horn, truck horn",
    "Train horn",
    "Toot",
    "Honk",
)
HORN_LIKE_CLASS_NAMES = (
    "Buzzer",
    "Beep, bleep",
    "Alarm",
    "Siren",
    "Alarm clock",
    "Ringtone",
    "Telephone",
    "Telephone bell ringing",
    "Doorbell",
    "Bell",
    "Ding-dong",
)
DOG_CLASS_NAMES = (
    "Bark",
    "Yip",
    "Howl",
    "Bow-wow",
    "Growling",
    "Whimper (dog)",
)

audio_buffer = np.zeros((int(RAW_SAMPLE_RATE * 1.2), RAW_CHANNELS), dtype=np.float32)
audio_window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
callback_count = 0
audio_queue = queue.Queue()


def load_class_names(class_map_csv):
    with open(class_map_csv, encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        return np.array([display_name for (_, _, display_name) in reader])


def list_audio_devices():
    print("\n========== Audio Devices ==========")
    print(sd.query_devices())
    print("===================================\n")


def find_micarray_device():
    devices = sd.query_devices()
    keywords = ("SipeedUSB MicArray", "MicArray")
    for idx, dev in enumerate(devices):
        name = dev["name"]
        max_input_channels = int(dev["max_input_channels"])
        if max_input_channels >= RAW_CHANNELS and any(keyword in name for keyword in keywords):
            return idx, name
    for idx, dev in enumerate(devices):
        name = dev["name"]
        max_input_channels = int(dev["max_input_channels"])
        if max_input_channels >= RAW_CHANNELS and "USB" in name:
            return idx, name
    return None, None


def compute_mel_spectrogram(audio_data, sr=16000):
    audio_data = np.asarray(audio_data, dtype=np.float32).reshape(-1)
    target_len = 15600
    if len(audio_data) < target_len:
        audio_data = np.pad(audio_data, (0, target_len - len(audio_data)), mode="constant")
    elif len(audio_data) > target_len:
        audio_data = audio_data[:target_len]

    audio_padded = np.pad(audio_data, (0, 112), mode="constant")
    mel_spec = librosa.feature.melspectrogram(
        y=audio_padded,
        sr=sr,
        n_fft=512,
        win_length=400,
        hop_length=160,
        window="hann",
        center=False,
        n_mels=64,
        fmin=125.0,
        fmax=7500.0,
        power=1.0,
        htk=True,
        norm=None,
    )
    log_mel_spec = np.log(mel_spec + 0.001).astype(np.float32).T
    if log_mel_spec.shape != (96, 64):
        pad_value = np.log(0.001).astype(np.float32)
        log_mel_spec = log_mel_spec[:96, :64]
        pad_t = max(0, 96 - log_mel_spec.shape[0])
        pad_f = max(0, 64 - log_mel_spec.shape[1])
        log_mel_spec = np.pad(log_mel_spec, ((0, pad_t), (0, pad_f)), mode="constant", constant_values=pad_value)
    return log_mel_spec.reshape(1, 96, 64, 1).astype(np.float32)


def normalize_angle(angle):
    return (angle + 180.0) % 360.0 - 180.0


def angle_delta(a, b):
    return normalize_angle(a - b)


def robust_channel_level(values):
    values = np.asarray(values, dtype=np.float32)
    top_count = min(4, values.size)
    return float(np.mean(np.sort(values)[-top_count:])) if values.size else 0.0


def circular_mean(angles, weights):
    radians = np.radians(angles)
    x = float(np.sum(weights * np.cos(radians)))
    y = float(np.sum(weights * np.sin(radians)))
    if x == 0.0 and y == 0.0:
        return float(angles[0])
    return normalize_angle(math.degrees(math.atan2(y, x)))


OUTER_MIC_ANGLES_DEG = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
MIC_POSITIONS = {
    channel: np.array(
        (
            MIC_ARRAY_RADIUS * math.cos(math.radians(angle)),
            MIC_ARRAY_RADIUS * math.sin(math.radians(angle)),
        ),
        dtype=np.float32,
    )
    for channel, angle in zip(DOA_CHANNELS, OUTER_MIC_ANGLES_DEG)
}
SRP_ANGLES = np.arange(-180.0, 180.0, SRP_ANGLE_STEP_DEG, dtype=np.float32)
SRP_DIRECTIONS = np.column_stack((np.cos(np.deg2rad(SRP_ANGLES)), np.sin(np.deg2rad(SRP_ANGLES))))
SRP_PAIRS = [(a, b) for i, a in enumerate(DOA_CHANNELS) for b in DOA_CHANNELS[i + 1 :]]
SOS_FILTER = signal.butter(4, [300, 4000], btype="band", fs=RAW_SAMPLE_RATE, output="sos")


def apply_bandpass(data):
    axis = 0 if np.ndim(data) > 1 else -1
    return signal.sosfilt(SOS_FILTER, data, axis=axis)


def gcc_phat_curve(sig1, sig2, fs, max_shift_samples, beta=0.75):
    sig1 = apply_bandpass(sig1) - np.mean(sig1)
    sig2 = apply_bandpass(sig2) - np.mean(sig2)
    window = np.hanning(len(sig1))
    sig1 = sig1 * window
    sig2 = sig2 * window

    n = len(sig1) + len(sig2)
    sig1_fft = fft.fft(sig1, n=n)
    sig2_fft = fft.fft(sig2, n=n)
    cross = sig1_fft * np.conj(sig2_fft)
    cross_abs = np.abs(cross)
    max_val = np.max(cross_abs)
    if max_val > 0:
        cross_abs = np.maximum(cross_abs, 0.05 * max_val)
        cross_abs = np.power(cross_abs, beta)
    else:
        cross_abs[cross_abs == 0] = 1e-10

    cc = np.real(fft.ifft(cross / cross_abs))
    cc_shifted = np.concatenate((cc[n // 2 :], cc[: n // 2]))
    center = n // 2
    search_start = max(0, center - max_shift_samples)
    search_end = min(n, center + max_shift_samples + 1)
    curve = cc_shifted[search_start:search_end].astype(np.float32)
    shifts = np.arange(search_start - center, search_end - center, dtype=np.float32)
    scale = float(np.max(np.abs(curve)))
    if scale > 1e-8:
        curve = curve / scale
    return shifts, curve


def raw_frame_channels(frame_audio):
    valid_channels = [ch for ch in DOA_CHANNELS if ch < frame_audio.shape[1]]
    return frame_audio[:, valid_channels] if valid_channels else None


def raw_frame_peak(frame_raw):
    return robust_channel_level(np.max(np.abs(frame_raw), axis=0))


def raw_frame_rms(frame_raw):
    return robust_channel_level(np.sqrt(np.mean(frame_raw ** 2, axis=0)))


def build_candidate_starts(audio_48k):
    if len(audio_48k) < DOA_FRAME_SIZE:
        return []
    valid_channels = [ch for ch in DOA_CHANNELS if ch < audio_48k.shape[1]]
    if not valid_channels:
        return []

    filtered = apply_bandpass(audio_48k[:, valid_channels])
    starts = list(range(0, len(filtered) - DOA_FRAME_SIZE + 1, DOA_HOP_SIZE))
    peaks = np.array(
        [robust_channel_level(np.max(np.abs(filtered[start : start + DOA_FRAME_SIZE, :]), axis=0)) for start in starts]
    )
    max_peak = float(np.max(peaks))
    if max_peak < DOA_MIN_FRAME_PEAK:
        return []

    threshold = max(DOA_MIN_FRAME_PEAK, max_peak * 0.35, float(np.percentile(peaks, 20)) * 2.5)
    onset_hits = np.where(peaks >= threshold)[0]
    if len(onset_hits) == 0:
        return []

    recent_start_sample = max(0, len(filtered) - int(RAW_SAMPLE_RATE * DOA_RECENT_PRIORITY_SEC))
    recent_hit_ids = [int(idx) for idx in onset_hits if starts[int(idx)] >= recent_start_sample]
    onset_idx = recent_hit_ids[0] if recent_hit_ids else int(onset_hits[-1])
    direct_hops = max(2, int((DOA_DIRECT_SOUND_WINDOW_MS / 1000.0) * RAW_SAMPLE_RATE / DOA_HOP_SIZE))
    candidate_ids = set(range(max(0, onset_idx - 1), min(len(starts), onset_idx + direct_hops + 2)))
    return [starts[idx] for idx in sorted(candidate_ids)]


def direct_echo_ratio(audio_48k, start):
    frame_raw = raw_frame_channels(audio_48k[start : start + DOA_FRAME_SIZE, :])
    if frame_raw is None or len(frame_raw) < DOA_FRAME_SIZE:
        return 0.0

    direct_energy = raw_frame_rms(frame_raw)
    echo_offset = int(RAW_SAMPLE_RATE * DOA_DIRECT_SOUND_WINDOW_MS / 1000.0)
    echo_length = int(RAW_SAMPLE_RATE * DOA_ECHO_LOOKAHEAD_MS / 1000.0)
    echo_start = start + echo_offset
    echo_end = min(len(audio_48k), echo_start + echo_length)
    if echo_end - echo_start < DOA_FRAME_SIZE:
        return 1.0

    echo_raw = raw_frame_channels(audio_48k[echo_start:echo_end, :])
    if echo_raw is None:
        return 1.0

    echo_rms_values = []
    for echo_frame_start in range(0, len(echo_raw) - DOA_FRAME_SIZE + 1, DOA_HOP_SIZE):
        echo_frame = echo_raw[echo_frame_start : echo_frame_start + DOA_FRAME_SIZE, :]
        echo_rms_values.append(raw_frame_rms(echo_frame))

    late_energy = max(echo_rms_values) if echo_rms_values else 0.0
    return direct_energy / (late_energy + 1e-8)


def srp_phat_frame_direction(frame_audio):
    if frame_audio.shape[1] <= max(DOA_CHANNELS):
        return None

    max_spacing = max(float(np.linalg.norm(MIC_POSITIONS[a] - MIC_POSITIONS[b])) for a, b in SRP_PAIRS)
    max_shift_samples = int(max_spacing / SPEED_OF_SOUND * RAW_SAMPLE_RATE) + 4

    scores = np.zeros(len(SRP_ANGLES), dtype=np.float32)
    pair_count = 0
    for ch_a, ch_b in SRP_PAIRS:
        shifts, curve = gcc_phat_curve(frame_audio[:, ch_a], frame_audio[:, ch_b], RAW_SAMPLE_RATE, max_shift_samples)
        pos_delta = MIC_POSITIONS[ch_a] - MIC_POSITIONS[ch_b]
        expected_samples = -(SRP_DIRECTIONS @ pos_delta) / SPEED_OF_SOUND * RAW_SAMPLE_RATE
        scores += np.interp(expected_samples, shifts, curve, left=0.0, right=0.0).astype(np.float32)
        pair_count += 1

    scores /= max(pair_count, 1)
    best_idx = int(np.argmax(scores))
    best_angle = float(SRP_ANGLES[best_idx])
    best_score = float(scores[best_idx])
    outside_peak = [float(score) for angle, score in zip(SRP_ANGLES, scores) if abs(angle_delta(float(angle), best_angle)) > DOA_CLUSTER_DEG]
    second_score = max(outside_peak) if outside_peak else float(np.median(scores))
    dominance = (best_score - second_score) / (abs(best_score) + 1e-6)
    return {"angle": best_angle, "confidence": float(dominance), "pair_count": pair_count}


def estimate_direction(audio_48k):
    candidates = []
    weak_candidates = []
    for start in build_candidate_starts(audio_48k):
        frame_audio = audio_48k[start : start + DOA_FRAME_SIZE, :]
        frame_raw = raw_frame_channels(frame_audio)
        if frame_raw is None:
            continue
        frame_peak = raw_frame_peak(frame_raw)
        if frame_peak < DOA_MIN_FRAME_PEAK:
            continue
        echo_ratio = direct_echo_ratio(audio_48k, start)
        if echo_ratio < DOA_MIN_DIRECT_ECHO_RATIO:
            continue
        frame_direction = srp_phat_frame_direction(frame_audio)
        if frame_direction is None:
            continue

        energy = raw_frame_rms(frame_raw)
        echo_weight = min(1.0, max(0.35, echo_ratio))
        item = {
            "angle": frame_direction["angle"],
            "confidence": frame_direction["confidence"],
            "score": max(frame_direction["confidence"], 0.0) * energy * echo_weight,
            "pair_count": frame_direction["pair_count"],
            "echo_ratio": echo_ratio,
        }
        if frame_direction["confidence"] >= DOA_MIN_CONFIDENCE:
            candidates.append(item)
        elif frame_direction["confidence"] >= DOA_WEAK_CONFIDENCE:
            weak_candidates.append(item)

    if not candidates:
        if not weak_candidates:
            return None
        best = sorted(weak_candidates, key=lambda item: item["score"], reverse=True)[0]
        return {"angle": best["angle"], "confidence": best["confidence"], "spread": 0.0, "frames": 1, "stable": False}

    candidates.sort(key=lambda item: item["score"], reverse=True)
    anchor = candidates[0]
    cluster = [item for item in candidates if abs(angle_delta(item["angle"], anchor["angle"])) <= DOA_CLUSTER_DEG] or [anchor]
    angles = np.array([item["angle"] for item in cluster], dtype=np.float32)
    weights = np.array([item["score"] for item in cluster], dtype=np.float32)
    final_angle = circular_mean(angles, weights)
    spread = max(abs(angle_delta(float(angle), final_angle)) for angle in angles)
    confidence = float(np.mean([item["confidence"] for item in cluster]))
    echo_ratio = float(np.mean([item["echo_ratio"] for item in cluster]))
    stable = spread <= DOA_MAX_STABLE_SPREAD_DEG and echo_ratio >= DOA_MIN_DIRECT_ECHO_RATIO and (
        (len(cluster) >= DOA_MIN_STABLE_FRAMES and confidence >= DOA_MIN_CONFIDENCE) or confidence >= DOA_STRONG_CONFIDENCE
    )
    return {"angle": final_angle, "confidence": confidence, "spread": spread, "frames": len(cluster), "stable": stable}


def smooth_angle(raw_angle, history, confidence, spread):
    if not history:
        history.append(raw_angle)
        return raw_angle
    current_smoothed = circular_mean(np.array(history, dtype=np.float32), np.ones(len(history), dtype=np.float32))
    shift = abs(angle_delta(raw_angle, current_smoothed))
    if shift >= DOA_DIRECTION_RESET_DEG and confidence >= DOA_MIN_CONFIDENCE and spread <= DOA_MAX_STABLE_SPREAD_DEG:
        history.clear()
        history.append(raw_angle)
        return raw_angle
    history.append(raw_angle)
    angles = np.array(history, dtype=np.float32)
    weights = np.ones(len(angles), dtype=np.float32)
    weights[-1] = 4.0 if shift >= DOA_FAST_ADAPT_DEG else 2.0
    return circular_mean(angles, weights)


def should_accept_direction(direction, history, pending_direction):
    if not history:
        good_seed = direction["confidence"] >= DOA_COLD_START_CONFIDENCE and direction["frames"] >= DOA_COLD_START_FRAMES and direction["spread"] <= DOA_COLD_START_SPREAD_DEG
        if good_seed:
            pending_direction["angle"] = None
            pending_direction["count"] = 0
            return True
        pending_angle = pending_direction.get("angle")
        if pending_angle is None or abs(angle_delta(direction["angle"], pending_angle)) > DOA_COLD_START_CONFIRM_DEG:
            pending_direction["angle"] = direction["angle"]
            pending_direction["count"] = 1
            return False
        pending_direction["count"] = int(pending_direction.get("count", 0)) + 1
        if pending_direction["count"] >= DOA_COLD_START_CONFIRM_COUNT:
            pending_direction["angle"] = None
            pending_direction["count"] = 0
            return True
        return False

    baseline = circular_mean(np.array(history, dtype=np.float32), np.ones(len(history), dtype=np.float32))
    shift = abs(angle_delta(direction["angle"], baseline))
    if shift < DOA_OUTLIER_REJECT_DEG:
        pending_direction["angle"] = None
        pending_direction["count"] = 0
        return True
    strong_jump = direction["confidence"] >= DOA_OUTLIER_ACCEPT_CONFIDENCE and direction["frames"] >= DOA_OUTLIER_ACCEPT_FRAMES and direction["spread"] <= DOA_OUTLIER_ACCEPT_SPREAD_DEG
    if strong_jump:
        pending_direction["angle"] = None
        pending_direction["count"] = 0
        return True
    pending_angle = pending_direction.get("angle")
    if pending_angle is None or abs(angle_delta(direction["angle"], pending_angle)) > DOA_OUTLIER_CONFIRM_DEG:
        pending_direction["angle"] = direction["angle"]
        pending_direction["count"] = 1
        return False
    pending_direction["count"] = int(pending_direction.get("count", 0)) + 1
    if pending_direction["count"] >= DOA_OUTLIER_CONFIRM_COUNT:
        history.clear()
        pending_direction["angle"] = None
        pending_direction["count"] = 0
        return True
    return False


def classify_target_name(class_name):
    lower = class_name.lower()
    if (
        "horn" in lower
        or "honk" in lower
        or "toot" in lower
        or "buzzer" in lower
        or "beep" in lower
        or "alarm" in lower
        or "siren" in lower
        or "ringtone" in lower
        or "telephone" in lower
        or "doorbell" in lower
        or "bell" in lower
        or "ding-dong" in lower
    ):
        return "horn"
    if any(key in lower for key in ("bark", "howl", "yip", "bow-wow", "growl", "whimper", "dog")):
        return "dog"
    return None


def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    global callback_count
    if status:
        print(status, file=sys.stderr)
    callback_count += 1
    if indata.ndim != 2:
        print(f"[audio] unexpected input shape: {indata.shape}")
        return
    if indata.shape[1] <= MIC_CHANNEL:
        print(f"[audio] input channels={indata.shape[1]} but MIC_CHANNEL={MIC_CHANNEL}")
        return
    if callback_count % DEBUG_PRINT_EVERY_N_BLOCKS == 0:
        rms_list = np.sqrt(np.mean(indata * indata, axis=0) + 1e-12)
        rms_str = " ".join([f"ch{i}:{rms_list[i]:.6f}" for i in range(indata.shape[1])])
        print(f"[8ch RMS] {rms_str}")
    gained = np.clip(indata.astype(np.float32) * FIXED_AUDIO_GAIN, -1.0, 1.0)
    ch_raw = gained[:, MIC_CHANNEL].copy()
    ch_16k = librosa.resample(ch_raw, orig_sr=RAW_SAMPLE_RATE, target_sr=MODEL_SAMPLE_RATE).astype(np.float32)
    audio_queue.put(ch_16k)
    audio_buffer = np.roll(audio_buffer, -frames, axis=0)
    audio_buffer[-frames:, :] = gained


def connect_retry():
    while True:
        try:
            sock = socket.create_connection((RADAR_HOST, RADAR_PORT), timeout=2.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return sock
        except OSError:
            time.sleep(0.5)


def main():
    global audio_window

    list_audio_devices()
    input_device, input_device_name = find_micarray_device()
    if input_device is None:
        raise RuntimeError("SipeedUSB MicArray device not found")

    class_names = load_class_names(CLASS_MAP_PATH)
    name_to_index = {name: idx for idx, name in enumerate(class_names)}
    target_indices = [name_to_index[name] for name in HORN_CLASS_NAMES + HORN_LIKE_CLASS_NAMES + DOG_CLASS_NAMES if name in name_to_index]
    target_thresholds = {name_to_index[name]: HORN_THRESHOLD for name in HORN_CLASS_NAMES if name in name_to_index}
    target_thresholds.update({name_to_index[name]: HORN_LIKE_THRESHOLD for name in HORN_LIKE_CLASS_NAMES if name in name_to_index})
    target_thresholds.update({name_to_index[name]: DOG_THRESHOLD for name in DOG_CLASS_NAMES if name in name_to_index})

    rknn = RKNNLite()
    if rknn.load_rknn(MODEL_PATH) != 0:
        raise RuntimeError(f"load_rknn failed: {MODEL_PATH}")
    if rknn.init_runtime(core_mask=0) != 0:
        raise RuntimeError("init_runtime failed")

    print("[radar] config:")
    print(
        f"  device={input_device} ({input_device_name}), raw_rate={RAW_SAMPLE_RATE}, "
        f"raw_channels={RAW_CHANNELS}, mic_channel={MIC_CHANNEL}"
    )
    print(f"  model_rate={MODEL_SAMPLE_RATE}, window={WINDOW_SAMPLES}, step={STEP_SAMPLES}")
    print(
        f"  horn_threshold={HORN_THRESHOLD}, horn_like_threshold={HORN_LIKE_THRESHOLD}, "
        f"dog_threshold={DOG_THRESHOLD}, min_ai_rms={MIN_AI_RMS}, fixed_audio_gain={FIXED_AUDIO_GAIN}"
    )
    print("[radar] waiting for HUD socket...")

    sock = connect_retry()
    print("[radar] connected to HUD socket.")
    score_memory = np.zeros(len(class_names), dtype=np.float32)
    direction_history = deque(maxlen=DOA_HISTORY_SIZE)
    pending_direction = {"angle": None, "count": 0}
    last_output_time = 0.0

    stream = sd.InputStream(
        device=input_device,
        samplerate=RAW_SAMPLE_RATE,
        channels=RAW_CHANNELS,
        blocksize=RAW_BLOCKSIZE,
        dtype="float32",
        callback=audio_callback,
    )

    try:
        with stream:
            while True:
                new_chunk = audio_queue.get()
                if len(new_chunk) > STEP_SAMPLES:
                    new_chunk = new_chunk[:STEP_SAMPLES]
                elif len(new_chunk) < STEP_SAMPLES:
                    new_chunk = np.pad(new_chunk, (0, STEP_SAMPLES - len(new_chunk)), mode="constant")

                audio_window = np.roll(audio_window, -STEP_SAMPLES)
                audio_window[-STEP_SAMPLES:] = new_chunk

                rms = float(np.sqrt(np.mean(audio_window ** 2)))
                if rms < MIN_AI_RMS:
                    print(f"[ai] skipped: window rms too low ({rms:.6f})")
                    score_memory *= SCORE_QUIET_DECAY
                    continue

                peak = float(np.percentile(np.abs(audio_window), 99.9))
                ai_audio = np.clip(audio_window, -1.0, 1.0)
                print(f"[ai] input rms={rms:.6f}, peak={peak:.6f}, fixed_gain={FIXED_AUDIO_GAIN:.3f}")
                mel_input = compute_mel_spectrogram(ai_audio, sr=MODEL_SAMPLE_RATE)
                outputs = rknn.inference(inputs=[mel_input])
                if not outputs:
                    print("[ai] RKNN outputs empty")
                    continue

                prediction = outputs[0][0]
                current_scores = np.zeros(len(class_names), dtype=np.float32)
                current_scores[: len(prediction)] = prediction[: len(class_names)]
                score_memory = np.maximum(current_scores, score_memory * SCORE_MEMORY_DECAY)

                top_indices = np.argsort(score_memory)[::-1][:TOP_K]
                top_items = []
                for idx in top_indices:
                    score = float(score_memory[idx])
                    if score >= TOP_K_SCORE_THRESHOLD:
                        top_items.append(f"{class_names[idx]}:{score:.3f}")
                if top_items:
                    print("[ai] topk:", " | ".join(top_items))
                else:
                    idx = int(top_indices[0])
                    print(f"[ai] top1 below threshold: {class_names[idx]}:{float(score_memory[idx]):.3f}")

                passed_indices = [idx for idx in target_indices if float(score_memory[idx]) >= target_thresholds.get(idx, HORN_THRESHOLD)]
                if not passed_indices:
                    print("[ai] no horn/dog target passed threshold")
                    continue

                best_idx = max(passed_indices, key=lambda idx: score_memory[idx])
                kind = classify_target_name(class_names[best_idx])
                if kind is None:
                    print(f"[ai] target matched threshold but type unsupported: {class_names[best_idx]}")
                    continue

                print(f"[ai] target hit: {kind} | {class_names[best_idx]} | score={float(score_memory[best_idx]):.3f}")

                now = time.monotonic()
                if now - last_output_time < OUTPUT_INTERVAL_SEC:
                    print("[doa] throttled by output interval")
                    continue

                latest_audio = audio_buffer[-int(RAW_SAMPLE_RATE * DOA_SEARCH_SEC) :, :]
                direction = estimate_direction(latest_audio)
                if direction is None or not direction["stable"]:
                    if direction is None:
                        print("[doa] no direction estimated")
                    else:
                        print(
                            f"[doa] unstable direction: angle={direction['angle']:.1f}, "
                            f"conf={direction['confidence']:.3f}, spread={direction['spread']:.1f}, frames={direction['frames']}"
                        )
                    continue
                if not should_accept_direction(direction, direction_history, pending_direction):
                    print("[doa] direction rejected by temporal filter")
                    continue

                final_angle = smooth_angle(direction["angle"], direction_history, direction["confidence"], direction["spread"])
                payload = {
                    "type": kind,
                    "angle": round(float(final_angle), 2),
                    "confidence": round(float(score_memory[best_idx]), 3),
                    "class_name": class_names[best_idx],
                }
                print(
                    f"[radar] emit type={payload['type']} angle={payload['angle']:.2f} "
                    f"confidence={payload['confidence']:.3f} class={payload['class_name']}"
                )

                try:
                    sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
                except OSError:
                    try:
                        sock.close()
                    except OSError:
                        pass
                    sock = connect_retry()
                    sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))

                last_output_time = now
    finally:
        try:
            sock.close()
        except OSError:
            pass
        rknn.release()


if __name__ == "__main__":
    main()
