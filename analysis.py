"""
Analyse audio via Essentia — tonalité, BPM, structure.
"""

import numpy as np
import essentia.standard as es
from camelot import to_camelot

ANALYSIS_VERSION = 2
KEY_CONFIDENCE_THRESHOLD = 0.5

# ── Constantes structure ───────────────────────────────────────────────────────
MIN_DIST_SEC       = 16     # distance minimale entre deux cue points
NOVELTY_THRESHOLD  = 0.25   # seuil de détection (0–1)
MFCC_WINDOW_SEC    = 3      # fenêtre de comparaison MFCC (secondes de chaque côté)
ENERGY_WINDOW_SEC  = 4      # fenêtre de mesure énergie avant/après
BUILDUP_WINDOW_SEC = 12     # fenêtre de détection buildup
DROP_RATIO         = 1.35
BREAK_RATIO        = 0.65


def analyze_audio(filepath: str) -> dict:
    loader = es.MonoLoader(filename=filepath, sampleRate=44100)
    audio = loader()
    duration = len(audio) / 44100.0

    # ── Tonalité ──────────────────────────────────────────────────────────────
    key_extractor = es.KeyExtractor()
    key, scale, key_strength = key_extractor(audio)
    camelot_key = to_camelot(key, scale) if key_strength >= KEY_CONFIDENCE_THRESHOLD else "?"

    # ── BPM ───────────────────────────────────────────────────────────────────
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, bpm_confidence, _, _ = rhythm_extractor(audio)

    # ── Structure (MFCC Essentia) ─────────────────────────────────────────────
    cue_points = detect_cue_points(audio, bpm, duration)

    return {
        "key": camelot_key,
        "key_confidence": round(float(key_strength), 3),
        "bpm": round(float(bpm), 2),
        "bpm_confidence": round(float(bpm_confidence), 3),
        "cue_points": cue_points,
        "duration": round(duration, 2),
        "analysis_version": ANALYSIS_VERSION,
    }


def gaussian_smooth(arr, sigma):
    """Lissage gaussien sans scipy."""
    half = int(round(sigma * 3))
    kernel = np.array([np.exp(-k**2 / (2 * sigma**2)) for k in range(-half, half + 1)])
    kernel /= kernel.sum()
    n = len(arr)
    result = np.zeros(n)
    for t in range(n):
        acc = 0.0
        for ki, w in enumerate(kernel):
            idx = t + ki - half
            idx = max(0, min(n - 1, idx))
            acc += w * arr[idx]
        result[t] = acc
    return result


def detect_cue_points(audio, bpm, duration: float) -> list:
    if duration < 20 or bpm <= 0:
        return []

    SAMPLE_RATE = 44100
    FRAME_SIZE = 2048
    HOP_SIZE = 512

    # ── Extraction MFCC + RMS par frame ───────────────────────────────────────
    frame_cutter = es.FrameCutter(frameSize=FRAME_SIZE, hopSize=HOP_SIZE, startFromZero=True)
    windowing    = es.Windowing(type='hann')
    spectrum_alg = es.Spectrum()
    mfcc_alg     = es.MFCC(numberCoefficients=13)
    rms_alg      = es.RMS()

    mfcc_frames, rms_frames = [], []
    for frame in frame_cutter(audio):
        spec = spectrum_alg(windowing(frame))
        _, coeffs = mfcc_alg(spec)
        mfcc_frames.append(coeffs)
        rms_frames.append(float(rms_alg(frame)))

    if not mfcc_frames:
        return []

    # ── Agrégation par seconde ────────────────────────────────────────────────
    frames_per_sec = max(1, SAMPLE_RATE // HOP_SIZE)
    n_secs = int(duration)

    mfcc_secs, rms_secs = [], []
    for s in range(n_secs):
        start = s * frames_per_sec
        end = min(start + frames_per_sec, len(mfcc_frames))
        if start >= len(mfcc_frames):
            break
        mfcc_secs.append(np.mean(mfcc_frames[start:end], axis=0))
        rms_secs.append(float(np.mean(rms_frames[start:end])))

    n = len(mfcc_secs)
    if n < 8:
        return []

    mfcc_matrix = np.array(mfcc_secs)
    rms_array   = np.array(rms_secs)

    # Normaliser MFCC (cosine similarity)
    norms = np.linalg.norm(mfcc_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    mfcc_norm = mfcc_matrix / norms

    # ── Courbe de nouveauté par distance MFCC entre sections adjacentes ───────
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

    # ── Détection de pics ─────────────────────────────────────────────────────
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

    # ── Classification ────────────────────────────────────────────────────────
    mean_rms = float(rms_array.mean())
    ew = max(1, int(ENERGY_WINDOW_SEC))
    bw = max(1, int(BUILDUP_WINDOW_SEC))

    def mean_range(arr, a, b):
        a, b = max(0, int(a)), min(len(arr), int(b))
        return float(arr[a:b].mean()) if a < b else 0.0

    cue_points = []
    for ci, t in enumerate(candidates):
        pos = float(t)
        rel = pos / n

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
            "position":      round(pos, 2),
            "type":          ctype,
            "confidence":    conf,
            "duration":      round(next_pos - pos, 2),
            "energyBefore":  round(e_before, 6),
            "energyAfter":   round(e_after, 6),
        })

    return cue_points
