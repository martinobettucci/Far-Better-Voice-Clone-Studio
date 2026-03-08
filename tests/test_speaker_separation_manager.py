from pathlib import Path
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from modules.core_components.ai_models.speaker_separation_manager import (
    SpeakerSeparationManager,
)


def _install_fake_speechbrain(monkeypatch, *, estimated_sources=None):
    calls = {}

    class FakeSeparator:
        @classmethod
        def from_hparams(cls, source, savedir, run_opts):
            calls["source"] = source
            calls["savedir"] = savedir
            calls["run_opts"] = run_opts

            class _Instance:
                def separate_file(self, audio_path):
                    calls["audio_path"] = audio_path
                    return estimated_sources

            return _Instance()

    separation_module = types.ModuleType("speechbrain.inference.separation")
    separation_module.SepformerSeparation = FakeSeparator
    inference_module = types.ModuleType("speechbrain.inference")
    inference_module.separation = separation_module
    speechbrain_module = types.ModuleType("speechbrain")
    speechbrain_module.inference = inference_module

    monkeypatch.setitem(sys.modules, "speechbrain", speechbrain_module)
    monkeypatch.setitem(sys.modules, "speechbrain.inference", inference_module)
    monkeypatch.setitem(sys.modules, "speechbrain.inference.separation", separation_module)
    return calls


def test_speaker_separation_manager_resolves_models_for_two_and_three_speakers(tmp_path: Path):
    manager = SpeakerSeparationManager(user_config={}, models_dir=tmp_path)

    model_2 = manager.get_model(2)
    model_3 = manager.get_model(3)

    assert model_2.repo_id == "speechbrain/sepformer-whamr16k"
    assert model_2.sample_rate == 16000
    assert model_3.repo_id == "speechbrain/sepformer-libri3mix"
    assert model_3.sample_rate == 8000


def test_speaker_separation_manager_uses_local_model_path_when_available(tmp_path: Path, monkeypatch):
    local_dir = tmp_path / "sepformer-whamr16k"
    local_dir.mkdir(parents=True)
    for filename in ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"):
        (local_dir / filename).write_text("ok", encoding="utf-8")
    calls = _install_fake_speechbrain(monkeypatch, estimated_sources=np.zeros((1, 8, 2), dtype=np.float32))
    monkeypatch.setattr(
        "modules.core_components.ai_models.speaker_separation_manager.run_pre_load_hooks",
        lambda: None,
    )

    manager = SpeakerSeparationManager(user_config={}, models_dir=tmp_path)
    manager.get_separator(2)

    assert calls["source"] == str(local_dir)
    assert calls["savedir"] == str(local_dir)


def test_speaker_separation_manager_raises_clear_offline_error_when_model_missing(tmp_path: Path, monkeypatch):
    _install_fake_speechbrain(monkeypatch, estimated_sources=np.zeros((1, 8, 3), dtype=np.float32))
    monkeypatch.setattr(
        "modules.core_components.ai_models.speaker_separation_manager.run_pre_load_hooks",
        lambda: None,
    )

    manager = SpeakerSeparationManager(user_config={"offline_mode": True}, models_dir=tmp_path)

    with pytest.raises(RuntimeError, match="Offline mode enabled"):
        manager.get_separator(3)


def test_speaker_separation_manager_normalizes_tracks_from_sepformer_output(tmp_path: Path, monkeypatch):
    audio_path = tmp_path / "mixed.wav"
    sf.write(str(audio_path), np.zeros(1600, dtype=np.float32), 16000)
    local_dir = tmp_path / "sepformer-whamr16k"
    local_dir.mkdir(parents=True)
    for filename in ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"):
        (local_dir / filename).write_text("ok", encoding="utf-8")

    estimated_sources = np.stack(
        [
            np.linspace(-0.2, 0.2, 12, dtype=np.float32),
            np.linspace(0.3, -0.3, 12, dtype=np.float32),
        ],
        axis=1,
    )[np.newaxis, :, :]
    calls = _install_fake_speechbrain(monkeypatch, estimated_sources=estimated_sources)
    monkeypatch.setattr(
        "modules.core_components.ai_models.speaker_separation_manager.run_pre_load_hooks",
        lambda: None,
    )

    manager = SpeakerSeparationManager(user_config={}, models_dir=tmp_path)
    tracks = manager.separate_file(str(audio_path), 2)

    assert calls["audio_path"] == str(audio_path)
    assert len(tracks) == 2
    assert tracks[0].speaker_index == 1
    assert tracks[0].sample_rate == 16000
    assert tracks[0].audio_data.shape == (12,)
    assert tracks[1].audio_data.shape == (12,)


def test_speaker_separation_manager_download_model_uses_snapshot_download_and_required_files(tmp_path, monkeypatch):
    local_dir = tmp_path / "sepformer-whamr16k"
    recorded = {}

    def fake_snapshot_download(repo_id, local_dir, **kwargs):
        recorded["repo_id"] = repo_id
        recorded["local_dir"] = local_dir
        recorded["kwargs"] = kwargs
        target = Path(local_dir)
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt"):
            (target / filename).write_text("ok", encoding="utf-8")
        return str(target)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    manager = SpeakerSeparationManager(user_config={}, models_dir=tmp_path)
    info = manager.download_model(2)

    assert recorded["repo_id"] == "speechbrain/sepformer-whamr16k"
    assert recorded["local_dir"] == str(local_dir)
    assert info["model_dir"] == str(local_dir)
