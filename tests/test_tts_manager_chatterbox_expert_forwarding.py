from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

import modules.core_components.ai_models.tts_manager as tts_manager_mod
from modules.core_components.ai_models.tts_manager import TTSManager


def test_chatterbox_generate_forwards_and_clamps_expert_params(monkeypatch):
    mgr = TTSManager(user_config={})
    captured = {}

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 32), dtype=torch.float32)

    monkeypatch.setattr(tts_manager_mod, "set_seed", lambda _seed: None)
    monkeypatch.setattr(mgr, "get_chatterbox_tts", lambda: FakeModel())

    audio, sr = mgr.generate_voice_clone_chatterbox(
        text="hello",
        voice_sample_path="/tmp/sample.wav",
        min_p=9.9,
        max_new_tokens=99999,
    )

    assert isinstance(audio, np.ndarray)
    assert sr == 24000
    assert captured["min_p"] == 0.30
    assert captured["max_new_tokens"] == 4096


def test_chatterbox_multilingual_forwards_and_clamps_expert_params(monkeypatch):
    mgr = TTSManager(user_config={})
    captured = {}

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 16), dtype=torch.float32)

    monkeypatch.setattr(tts_manager_mod, "set_seed", lambda _seed: None)
    monkeypatch.setattr(mgr, "get_chatterbox_multilingual", lambda: FakeModel())

    audio, sr = mgr.generate_voice_clone_chatterbox_multilingual(
        text="hello",
        language_code="en",
        voice_sample_path="/tmp/sample.wav",
        min_p=-3.0,
        max_new_tokens=1,
    )

    assert isinstance(audio, np.ndarray)
    assert sr == 24000
    assert captured["min_p"] == 0.0
    assert captured["max_new_tokens"] == 256


def test_chatterbox_vc_forwards_steps_and_auto_none(monkeypatch):
    mgr = TTSManager(user_config={})
    captured = {}

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 24), dtype=torch.float32)

    monkeypatch.setattr(mgr, "get_chatterbox_vc", lambda: FakeModel())

    audio, sr = mgr.generate_voice_convert_chatterbox(
        source_audio_path="/tmp/src.wav",
        target_voice_path="/tmp/target.wav",
        n_cfm_timesteps=0,
    )
    assert isinstance(audio, np.ndarray)
    assert sr == 24000
    assert captured["n_cfm_timesteps"] is None

    captured.clear()
    audio, sr = mgr.generate_voice_convert_chatterbox(
        source_audio_path="/tmp/src.wav",
        target_voice_path="/tmp/target.wav",
        n_cfm_timesteps=999,
    )
    assert isinstance(audio, np.ndarray)
    assert sr == 24000
    assert captured["n_cfm_timesteps"] == 100


def test_chatterbox_vc_applies_chunk_overrides_without_mutating_user_config(monkeypatch, tmp_path):
    mgr = TTSManager(
        user_config={
            "voice_changer_chunk_seconds": 12.0,
            "voice_changer_chunk_overlap_seconds": 1.0,
        }
    )
    captured = {}

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 24), dtype=torch.float32)

    monkeypatch.setattr(mgr, "get_chatterbox_vc", lambda: FakeModel())

    src_path = tmp_path / "src.wav"
    target_path = tmp_path / "target.wav"
    sf.write(src_path, np.zeros(16_000, dtype=np.float32), 16_000)
    sf.write(target_path, np.zeros(16_000, dtype=np.float32), 16_000)
    original_config = dict(mgr.user_config)

    audio, sr = mgr.generate_voice_convert_chatterbox(
        source_audio_path=src_path,
        target_voice_path=target_path,
        n_cfm_timesteps=25,
        chunk_seconds=6.0,
        overlap_seconds=1.5,
    )

    assert isinstance(audio, np.ndarray)
    assert sr == 24000
    assert captured["n_cfm_timesteps"] == 25
    assert mgr.user_config == original_config
    assert mgr._get_voice_changer_chunk_settings() == (12.0, 1.0)
    assert mgr._get_voice_changer_chunk_settings(chunk_seconds=6.0, overlap_seconds=1.5) == (6.0, 1.5)


def test_chatterbox_vc_short_source_with_chunk_override_stays_single_pass(monkeypatch, tmp_path):
    mgr = TTSManager(user_config={})

    class FakeModel:
        sr = 24000

        def __init__(self):
            self.generate_calls = 0

        def generate(self, **kwargs):
            self.generate_calls += 1
            return torch.zeros((1, 24), dtype=torch.float32)

        def generate_from_waveform(self, audio, source_sr, n_cfm_timesteps=None):
            raise AssertionError("short source should use single-pass generate")

    fake = FakeModel()
    monkeypatch.setattr(mgr, "get_chatterbox_vc", lambda: fake)

    src_path = tmp_path / "short.wav"
    target_path = tmp_path / "target.wav"
    sf.write(src_path, np.zeros(16_000 * 2, dtype=np.float32), 16_000)
    sf.write(target_path, np.zeros(16_000, dtype=np.float32), 16_000)

    audio, sr = mgr.generate_voice_convert_chatterbox(
        source_audio_path=src_path,
        target_voice_path=target_path,
        n_cfm_timesteps=25,
        chunk_seconds=6.0,
        overlap_seconds=1.5,
    )

    assert isinstance(audio, np.ndarray)
    assert sr == 24000
    assert fake.generate_calls == 1


def test_chatterbox_vc_chunks_long_sources_and_crossfades(monkeypatch, tmp_path):
    mgr = TTSManager(
        user_config={
            "voice_changer_chunk_seconds": 1.0,
            "voice_changer_chunk_overlap_seconds": 0.25,
        }
    )
    progress_updates = []

    class FakeModel:
        sr = 24000

        def __init__(self):
            self.target_voice_paths = []
            self.chunk_calls = []

        def set_target_voice(self, wav_fpath):
            self.target_voice_paths.append(wav_fpath)

        def generate_from_waveform(self, audio, source_sr, n_cfm_timesteps=None):
            arr = np.asarray(audio)
            frames = int(arr.shape[0]) if arr.ndim > 1 else int(arr.size)
            self.chunk_calls.append((frames, int(source_sr), n_cfm_timesteps))
            out_len = int(round((frames / float(source_sr)) * self.sr))
            value = float(len(self.chunk_calls))
            return torch.full((1, out_len), value, dtype=torch.float32)

        def generate(self, **kwargs):
            raise AssertionError("long source should use chunked waveform generation")

    fake = FakeModel()
    monkeypatch.setattr(mgr, "get_chatterbox_vc", lambda: fake)

    src_path = tmp_path / "source.wav"
    target_path = tmp_path / "target.wav"
    sf.write(src_path, np.zeros(16_000 * 3, dtype=np.float32), 16_000)
    sf.write(target_path, np.zeros(16_000, dtype=np.float32), 16_000)

    audio, sr = mgr.generate_voice_convert_chatterbox(
        source_audio_path=src_path,
        target_voice_path=target_path,
        n_cfm_timesteps=5,
        progress_callback=lambda current, total: progress_updates.append((current, total)),
    )

    assert sr == 24000
    assert audio.shape == (72_000,)
    assert fake.target_voice_paths == [str(target_path)]
    assert fake.chunk_calls == [
        (16_000, 16_000, 5),
        (16_000, 16_000, 5),
        (16_000, 16_000, 5),
        (12_000, 16_000, 5),
    ]
    assert progress_updates == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert audio[10_000] == 1.0
    assert 1.0 < audio[21_000] < 2.0
