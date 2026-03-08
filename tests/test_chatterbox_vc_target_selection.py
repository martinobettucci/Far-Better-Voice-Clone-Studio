from __future__ import annotations

import numpy as np
import soundfile as sf

from modules.chatterbox.vc import ChatterboxVC


def _make_fake_speech(sr: int, duration_s: float, base_freq: float = 180.0) -> np.ndarray:
    t = np.arange(int(sr * duration_s), dtype=np.float32) / float(sr)
    carrier = (
        0.36 * np.sin(2.0 * np.pi * base_freq * t)
        + 0.10 * np.sin(2.0 * np.pi * base_freq * 2.1 * t)
        + 0.05 * np.sin(2.0 * np.pi * base_freq * 3.3 * t)
    )
    syllable_env = np.maximum(0.0, np.sin(2.0 * np.pi * 4.2 * t))
    phrase_gate = (np.sin(2.0 * np.pi * 0.55 * t) > -0.1).astype(np.float32)
    return (carrier * (syllable_env ** 1.5) * phrase_gate).astype(np.float32)


class _FakeS3Gen:
    def __init__(self):
        self.last_ref = None

    def embed_ref(self, ref_wav, ref_sr, device="auto"):
        self.last_ref = np.asarray(ref_wav, dtype=np.float32).copy()
        return {
            "prompt_token": np.zeros((1, 1), dtype=np.int64),
            "prompt_token_len": np.array([1], dtype=np.int64),
            "prompt_feat": np.zeros((1, 1, 80), dtype=np.float32),
            "prompt_feat_len": None,
            "embedding": np.zeros((1, 80), dtype=np.float32),
        }


def test_target_selection_prefers_clean_speech_window(tmp_path):
    sr = 24_000
    rng = np.random.default_rng(1234)

    noise = (0.025 * rng.standard_normal(sr * 12)).astype(np.float32)
    speech = _make_fake_speech(sr, 10.0)
    tail = np.zeros(sr * 4, dtype=np.float32)
    full = np.concatenate([noise, speech, tail])

    path = tmp_path / "target.wav"
    sf.write(path, full, sr)

    fake = _FakeS3Gen()
    vc = ChatterboxVC(fake, device="cpu")
    vc.set_target_voice(path)

    selected = fake.last_ref
    assert selected is not None
    assert selected.shape[0] == vc.DEC_COND_LEN

    window = vc.DEC_COND_LEN
    step = sr
    candidate_scores = {}
    for start in range(0, full.shape[0] - window + 1, step):
        candidate_scores[start] = vc._score_target_window(full[start:start + window], sr)

    best_start = max(candidate_scores, key=candidate_scores.get)
    assert abs(best_start - (12 * sr)) <= sr
    assert vc._score_target_window(selected, sr) >= candidate_scores[best_start] - 0.1


def test_target_selection_trims_short_clip_boundaries(tmp_path):
    sr = 24_000
    speech = _make_fake_speech(sr, 4.0, base_freq=210.0)
    full = np.concatenate([
        np.zeros(sr * 2, dtype=np.float32),
        speech,
        np.zeros(sr * 2, dtype=np.float32),
    ])

    path = tmp_path / "short_target.wav"
    sf.write(path, full, sr)

    fake = _FakeS3Gen()
    vc = ChatterboxVC(fake, device="cpu")
    vc.set_target_voice(path)

    selected = fake.last_ref
    assert selected is not None
    assert selected.shape[0] < full.shape[0]
    assert int(3.0 * sr) <= selected.shape[0] <= int(5.0 * sr)
    assert vc._score_target_window(selected, sr) > vc._score_target_window(full, sr)
