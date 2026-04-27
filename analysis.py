"""
Logique d'analyse audio via Essentia.
"""

import essentia.standard as es
import numpy as np
from camelot import to_camelot

ANALYSIS_VERSION = 1

# Seuil de confiance minimum pour retourner une tonalité (vs "?")
KEY_CONFIDENCE_THRESHOLD = 0.5


def analyze_audio(filepath: str) -> dict:
    # Chargement mono 44100 Hz
    loader = es.MonoLoader(filename=filepath, sampleRate=44100)
    audio = loader()
    duration = len(audio) / 44100.0

    # ── Tonalité ──────────────────────────────────────────────
    key_extractor = es.KeyExtractor()
    key, scale, key_strength = key_extractor(audio)

    if key_strength >= KEY_CONFIDENCE_THRESHOLD:
        camelot_key = to_camelot(key, scale)
    else:
        camelot_key = "?"

    # ── BPM ───────────────────────────────────────────────────
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, bpm_confidence, _, beats_intervals = rhythm_extractor(audio)

    # ── Cue points (segmentation structurelle) ────────────────
    cue_points = detect_cue_points(audio, beats, bpm, duration)

    return {
        "key": camelot_key,
        "key_confidence": round(float(key_strength), 3),
        "bpm": round(float(bpm), 2),
        "bpm_confidence": round(float(bpm_confidence), 3),
        "cue_points": cue_points,
        "duration": round(duration, 2),
        "analysis_version": ANALYSIS_VERSION,
    }


def detect_cue_points(audio, beats, bpm, duration: float) -> list:
    """
    Détection de structure par analyse de l'énergie RMS par segment.
    Identifie les transitions significatives (drops, breaks, intros, outros).
    """
    if bpm <= 0 or len(beats) < 4:
        return []

    # Découper l'audio en segments d'une mesure (4 beats)
    samples_per_beat = int(44100 * 60 / bpm)
    samples_per_bar = samples_per_beat * 4
    n_bars = max(1, len(audio) // samples_per_bar)

    # Énergie RMS par mesure
    rms_per_bar = []
    for i in range(n_bars):
        start = i * samples_per_bar
        end = min(start + samples_per_bar, len(audio))
        segment = audio[start:end]
        rms = float(np.sqrt(np.mean(segment ** 2))) if len(segment) > 0 else 0.0
        rms_per_bar.append(rms)

    if not rms_per_bar:
        return []

    rms_array = np.array(rms_per_bar)
    mean_rms = np.mean(rms_array)
    std_rms = np.std(rms_array)

    cue_points = []

    for i in range(1, n_bars - 1):
        delta = rms_array[i] - rms_array[i - 1]
        position = i * samples_per_bar / 44100.0

        # Intro : début du track avec énergie croissante
        if i <= max(2, int(n_bars * 0.10)) and rms_array[i] < mean_rms * 0.6:
            if not any(c["type"] == "intro" for c in cue_points):
                cue_points.append({
                    "position": round(position, 2),
                    "type": "intro",
                    "confidence": round(0.7 + min(0.25, (mean_rms - rms_array[i]) / (mean_rms + 1e-6) * 0.5), 3),
                })

        # Drop : montée significative d'énergie
        elif delta > std_rms * 1.2 and rms_array[i] > mean_rms:
            cue_points.append({
                "position": round(position, 2),
                "type": "drop",
                "confidence": round(min(0.99, 0.65 + delta / (std_rms * 3)), 3),
            })

        # Break : chute significative d'énergie
        elif delta < -std_rms * 1.2 and rms_array[i] < mean_rms:
            cue_points.append({
                "position": round(position, 2),
                "type": "break",
                "confidence": round(min(0.99, 0.65 + abs(delta) / (std_rms * 3)), 3),
            })

        # Outro : fin du track avec énergie décroissante
        elif i >= int(n_bars * 0.85) and rms_array[i] < mean_rms * 0.6:
            if not any(c["type"] == "outro" for c in cue_points):
                cue_points.append({
                    "position": round(position, 2),
                    "type": "outro",
                    "confidence": round(0.7 + min(0.25, (mean_rms - rms_array[i]) / (mean_rms + 1e-6) * 0.5), 3),
                })

    # Déduplique : min 8 secondes entre deux cue points
    cue_points.sort(key=lambda c: c["position"])
    filtered = []
    last_pos = -999.0
    for cp in cue_points:
        if cp["position"] - last_pos >= 8.0:
            filtered.append(cp)
            last_pos = cp["position"]

    return filtered
