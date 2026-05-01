"""
Analyse audio via Essentia — tonalité, BPM, structure, loudness.
Optimisé mémoire : 22050 Hz, streaming MFCC, troncature 15 min.
"""

import numpy as np
import essentia.standard as es
from camelot import to_camelot

ANALYSIS_VERSION = 3
SAMPLE_RATE = 22050          # 44100 → 22050 : divise par 2 l'empreinte mémoire
MAX_DURATION_SEC = 15 * 60   # troncature de sécurité pour les très longs tracks
KEY_CONFIDENCE_THRESHOLD = 0.5

# ── Constantes structure ───────────────────────────────────────────────────────
MIN_DIST_SEC       = 16
NOVELTY_THRESHOLD  = 0.25
MFCC_WINDOW_SEC    = 3
ENERGY_WINDOW_SEC  = 4
BUILDUP_WINDOW_SEC = 12
DROP_RATIO         = 1.35
BREAK_RATIO        = 0.65
BEAT_SNAP_SEC      = 0.25


def _snap_to_beat(pos_sec: float, beats: np.ndarray) -> float:
    if len(beats) == 0:
        return pos_sec
    idx = int(np.argmin(np.abs(beats - pos_sec)))
    if abs(beats[idx] - pos_sec) <= BEAT_SNAP_SEC:
        return round(float(beats[idx]), 3)
    return pos_sec


def analyze_audio(filepath: str) -> dict:
    loader = es.MonoLoader(filename=filepath, sampleRate=SAMPLE_RATE)
    audio = loader()
    duration = len(audio) / float(SAMPLE_RATE)

    # Troncature de sécurité
    if duration > MAX_DURATION_SEC:
        audio = audio[:int(MAX_DURATION_SEC * SAMPLE_RATE)]
        duration = float(MAX_DURATION_SEC)

    # ── Tonalité ──────────────────────────────────────────────────────────────
    try:
        key, scale, key_strength = es.KeyExtractor(
            profileType="edma", sampleRate=SAMPLE_RATE
        )(audio)
    except Exception:
        key, scale, key_strength = es.KeyExtractor(sampleRate=SAMPLE_RATE)(audio)
    camelot_key = to_camelot(key, scale) if key_strength >= KEY_CONFIDENCE_THRESHOLD else "?"

    # ── BPM + beats ───────────────────────────────────────────────────────────
    bpm, beats, bpm_confidence, _, _ = es.RhythmExtractor2013(
        method="multifeature", sampleRate=SAMPLE_RATE
    )(audio)
    beats_arr = np.array(beats, dtype=np.float32)

    # ── Loudness RMS ──────────────────────────────────────────────────────────
    loudness_rms_db = None
    try:
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
        if rms > 1e-10:
            loudness_rms_db = round(20.0 * np.log10(rms), 1)
    except Exception:
        pass

    # ── Danceability ──────────────────────────────────────────────────────────
    danceability = None
    try:
        d, _ = es.Danceability(sampleRate=SAMPLE_RATE)(audio)
        danceability = round(float(d), 3)
    except Exception:
        pass

    # ── Structure ─────────────────────────────────────────────────────────────
    cue_points = detect_cue_points(audio, bpm, duration, beats_arr)

    return {
        "key":             camelot_key,
        "key_raw":         key,
        "key_scale":       scale,
        "key_confidence":  round(float(key_strength), 3),
        "bpm":             round(float(bpm), 2),
        "bpm_confidence":  round(float(bpm_confidence), 3),
        "beats":           [round(float(b), 3) for b in beats_arr],
        "loudness_rms_db": loudness_rms_db,
        "danceability":    danceability,
        "cue_points":      cue_points,
        "duration":        round(duration, 2),
        "analysis_version": ANALYSIS_VERSION,
    }


def gaussian_smooth(arr, sigma):
    half = int(round(sigma * 3))
    kernel = np.array([np.exp(-k**2 / (2 * sigma**2)) for k in range(-half, half + 1)])
    kernel /= kernel.sum()
    n = len(arr)
    result = np.zeros(n)
    for t in range(n):
        acc = 0.0
        for ki, w in enumerate(kernel):
            idx = max(0, min(n - 1, t + ki - half))
            acc += w * arr[idx]
        result[t] = acc
    return result


def detect_cue_points(audio, bpm, duration: float, beats_arr: np.ndarray) -> list:
    if duration < 20 or bpm <= 0:
        return []

    FRAME_SIZE = 2048
    HOP_SIZE   = 512

    windowing    = es.Windowing(type='hann')
    spectrum_alg = es.Spectrum()
    mfcc_alg     = es.MFCC(numberCoefficients=13)
    rms_alg      = es.RMS()

    # Streaming : agrégation par seconde — évite de stocker ~23K frames en mémoire
    frames_per_sec = max(1, SAMPLE_RATE // HOP_SIZE)
    n_secs = int(duration)

    mfcc_secs: list[np.ndarray] = []
    rms_secs: list[float] = []
    mfcc_acc  = np.zeros(13, dtype=np.float64)
    rms_acc   = 0.0
    count     = 0
    cur_sec   = 0

    for frame_idx, frame in enumerate(
        es.FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE, startFromZero=True)
    ):
        sec = frame_idx // frames_per_sec
        if sec != cur_sec:
            if count > 0 and cur_sec < n_secs:
                mfcc_secs.append((mfcc_acc / count).astype(np.float32))
                rms_secs.append(rms_acc / count)
            mfcc_acc[:] = 0.0
            rms_acc = 0.0
            count = 0
            cur_sec = sec

        spec = spectrum_alg(windowing(frame))
        _, coeffs = mfcc_alg(spec)
        mfcc_acc += coeffs
        rms_acc  += float(rms_alg(frame))
        count    += 1

    if count > 0 and cur_sec < n_secs:
        mfcc_secs.append((mfcc_acc / count).astype(np.float32))
        rms_secs.append(rms_acc / count)

    n = len(mfcc_secs)
    if n < 8:
        return []

    mfcc_matrix = np.array(mfcc_secs, dtype=np.float32)
    rms_array   = np.array(rms_secs,  dtype=np.float32)

    norms = np.linalg.norm(mfcc_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    mfcc_norm = mfcc_matrix / norms

    w = min(MFCC_WINDOW_SEC, n // 4)
    novelty = np.zeros(n)
    for t in range(w, n - w):
        before = np.mean(mfcc_norm[t - w:t], axis=0)
        after  = np.mean(mfcc_norm[t:t + w], axis=0)
        novelty[t] = float(np.linalg.norm(after - before))

    novelty_smooth = gaussian_smooth(novelty, sigma=3)
    max_n = novelty_smooth.max()
    if max_n > 0:
        novelty_smooth /= max_n

    min_dist = max(1, int(MIN_DIST_SEC))
    candidates = []
    last = -min_dist
    for t in range(1, n - 1):
        if (novelty_smooth[t] > novelty_smooth[t - 1] and
            novelty_smooth[t] > novelty_smooth[t + 1] and
            novelty_smooth[t] > NOVELTY_THRESHOLD and
            t - last >= min_dist):
            candidates.append(t)
            last = t

    mean_rms = float(rms_array.mean())
    ew = max(1, int(ENERGY_WINDOW_SEC))
    bw = max(1, int(BUILDUP_WINDOW_SEC))

    def mean_range(arr, a, b):
        a, b = max(0, int(a)), min(len(arr), int(b))
        return float(arr[a:b].mean()) if a < b else 0.0

    cue_points = []
    for ci, t in enumerate(candidates):
        pos_sec  = float(t)
        pos_snap = _snap_to_beat(pos_sec, beats_arr)
        rel      = pos_sec / n

        e_before = mean_range(rms_array, t - ew, t)
        e_after  = mean_range(rms_array, t, t + ew)
        conf     = round(float(novelty_smooth[t]), 3)

        if rel < 0.15 and e_before < mean_rms * 0.6:
            ctype = "intro"
        elif rel > 0.82 and e_after < e_before * 0.75:
            ctype = "outro"
        elif e_after > e_before * DROP_RATIO:
            ctype = "drop"
        elif e_after < e_before * BREAK_RATIO:
            ctype = "break"
        else:
            e_earlier = mean_range(rms_array, t - bw, t - ew)
            ctype = "buildup" if e_earlier < e_before * 0.75 and e_after >= e_before * 0.9 else "drop"

        next_pos = float(candidates[ci + 1]) if ci < len(candidates) - 1 else float(n)

        cue_points.append({
            "position":     pos_snap,
            "position_raw": round(pos_sec, 2),
            "snapped":      pos_snap != pos_sec,
            "type":         ctype,
            "confidence":   conf,
            "duration":     round(next_pos - pos_sec, 2),
            "energyBefore": round(e_before, 6),
            "energyAfter":  round(e_after, 6),
        })

    return cue_points
