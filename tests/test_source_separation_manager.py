from pathlib import Path
import json

import numpy as np
import pytest
import soundfile as sf

from modules.core_components.ai_models.source_separation_manager import (
    SourceSeparationManager,
    SourceSeparationModel,
    SourceSeparationResult,
    synthesize_backing_track,
)
from modules.core_components.tools.singing_enhancements import (
    build_source_separation_studio_route_update,
    build_source_separation_ui_payload,
    deserialize_source_separation_result,
    save_source_separation_outputs,
)


def _write_wav(path: Path, data: np.ndarray, sample_rate: int = 44_100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate)
    return path


def test_source_separation_catalog_normalization_filters_music_models(tmp_path: Path, monkeypatch):
    manager = SourceSeparationManager(user_config={}, models_dir=tmp_path)
    payload = [
        {
            "model_filename": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            "friendly_name": "Roformer Large",
            "architecture": "MDXC",
            "output_stems": ["Vocals", "Instrumental"],
        },
        {
            "model_filename": "htdemucs_6s.yaml",
            "friendly_name": "Demucs 6 Stem",
            "architecture": "Demucs",
            "output_stems": ["Vocals", "Drums", "Bass", "Other", "Guitar", "Piano"],
        },
        {
            "model_filename": "echo_clean.onnx",
            "friendly_name": "De-Echo",
            "architecture": "VR",
            "output_stems": ["Vocals"],
        },
    ]

    monkeypatch.setattr(manager, "_run_cli", lambda args, timeout_s=900: json.dumps(payload))

    models = manager.refresh_catalog(strict=True)

    filenames = {model.model_filename for model in models}
    assert "htdemucs_6s.yaml" in filenames
    assert "model_bs_roformer_ep_317_sdr_12.9755.ckpt" in filenames
    assert "echo_clean.onnx" not in filenames
    assert "Demucs" in manager.get_architecture_choices(models=models)
    assert "MDXC" in manager.get_architecture_choices(models=models)

    filtered = manager.filter_models(stem_filter="6 Stem", models=models)
    assert [model.model_filename for model in filtered] == ["htdemucs_6s.yaml"]
    assert manager.catalog_cache_path.exists()


def test_source_separation_offline_mode_accepts_downloaded_model_and_rejects_missing_one(tmp_path: Path):
    manager = SourceSeparationManager(user_config={"offline_mode": True}, models_dir=tmp_path)
    model = SourceSeparationModel(
        model_filename="htdemucs_ft.yaml",
        display_name="Demucs FT",
        architecture="Demucs",
        stems=("Vocals", "Drums", "Bass", "Other"),
    )
    manager._write_catalog_cache([model])

    model_dir = manager.get_model_dir(model.model_filename)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "htdemucs_ft.yaml").write_text("weights", encoding="utf-8")

    assert manager.ensure_model_available(model.model_filename) == model_dir

    missing = SourceSeparationModel(
        model_filename="missing.ckpt",
        display_name="Missing",
        architecture="MDXC",
        stems=("Vocals", "Instrumental"),
    )
    manager._write_catalog_cache([model, missing])

    with pytest.raises(RuntimeError, match="Offline mode enabled"):
        manager.ensure_model_available(missing.model_filename)


def test_source_separation_default_model_bundle_prefers_curated_upstream_models(tmp_path: Path):
    manager = SourceSeparationManager(user_config={}, models_dir=tmp_path)
    curated = SourceSeparationModel(
        model_filename="model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        display_name="Upstream Default Roformer",
        architecture="MDXC",
        stems=("Vocals", "Instrumental"),
    )
    demucs = SourceSeparationModel(
        model_filename="htdemucs_6s.yaml",
        display_name="Demucs 6 Stem",
        architecture="Demucs",
        stems=("Vocals", "Drums", "Bass", "Other", "Guitar", "Piano"),
    )
    fallback = SourceSeparationModel(
        model_filename="fallback.onnx",
        display_name="Fallback",
        architecture="MDX",
        stems=("Vocals", "Instrumental"),
    )

    manager._write_catalog_cache([fallback, demucs, curated])
    models = manager.list_models(refresh=False)

    assert manager.get_default_model_filename(models=models) == curated.model_filename
    assert [model.model_filename for model in manager.get_default_models(models=models)] == [
        "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "UVR_MDXNET_KARA_2.onnx",
        "htdemucs_ft.yaml",
        "htdemucs_6s.yaml",
    ]


def test_source_separation_builtin_models_are_available_without_live_catalog(tmp_path: Path, monkeypatch):
    manager = SourceSeparationManager(user_config={}, models_dir=tmp_path)
    monkeypatch.setattr(manager, "_run_cli", lambda args, timeout_s=900: "{}")

    models = manager.list_models(refresh=False)

    filenames = {model.model_filename for model in models}
    assert "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt" in filenames
    assert "UVR_MDXNET_KARA_2.onnx" in filenames
    assert "htdemucs_ft.yaml" in filenames


def test_synthesize_backing_track_aligns_length_and_channels(tmp_path: Path):
    drums = np.sin(np.linspace(0, np.pi * 4, 2_000, dtype=np.float32))
    bass = np.stack(
        [
            0.5 * np.sin(np.linspace(0, np.pi * 6, 3_000, dtype=np.float32)),
            0.5 * np.cos(np.linspace(0, np.pi * 6, 3_000, dtype=np.float32)),
        ],
        axis=1,
    )

    drums_path = _write_wav(tmp_path / "drums.wav", drums)
    bass_path = _write_wav(tmp_path / "bass.wav", bass)

    output_path = Path(
        synthesize_backing_track(
            {
                "Drums": str(drums_path),
                "Bass": str(bass_path),
            },
            tmp_path / "backing.wav",
        )
    )

    mixed, sample_rate = sf.read(str(output_path))
    assert sample_rate == 44_100
    assert mixed.shape[0] == 3_000
    assert mixed.ndim == 2
    assert mixed.shape[1] == 2


def test_source_separation_collects_outputs_even_if_backend_writes_to_cwd(tmp_path: Path, monkeypatch):
    manager = SourceSeparationManager(user_config={}, models_dir=tmp_path)
    target_dir = tmp_path / "target"
    target_dir.mkdir(parents=True, exist_ok=True)
    stray_dir = tmp_path / "cwd"
    stray_dir.mkdir(parents=True, exist_ok=True)

    vocals_path = _write_wav(stray_dir / "song__vocals.wav", np.zeros(256, dtype=np.float32))
    backing_path = _write_wav(stray_dir / "song__instrumental.wav", np.zeros(256, dtype=np.float32))

    monkeypatch.chdir(stray_dir)

    collected = manager._collect_separated_stem_paths(
        target_dir=target_dir,
        custom_output_names={
            "Vocals": "song__vocals",
            "Instrumental": "song__instrumental",
        },
    )

    assert collected["Vocals"] == str(target_dir / vocals_path.name)
    assert collected["Instrumental"] == str(target_dir / backing_path.name)
    assert (target_dir / vocals_path.name).exists()
    assert (target_dir / backing_path.name).exists()


def test_source_separation_save_bundle_and_ui_helpers(tmp_path: Path):
    vocals_path = _write_wav(tmp_path / "vocals.wav", np.zeros(1_000, dtype=np.float32))
    backing_path = _write_wav(tmp_path / "backing.wav", np.zeros((1_000, 2), dtype=np.float32))
    drums_path = _write_wav(tmp_path / "drums.wav", np.zeros((1_000, 2), dtype=np.float32))

    model = SourceSeparationModel(
        model_filename="htdemucs_6s.yaml",
        display_name="Demucs 6 Stem",
        architecture="Demucs",
        stems=("Vocals", "Drums", "Bass", "Other", "Guitar", "Piano"),
    )
    result = SourceSeparationResult(
        model=model,
        input_path="song.wav",
        output_dir=str(tmp_path),
        stem_paths={
            "Vocals": str(vocals_path),
            "Backing": str(backing_path),
            "Drums": str(drums_path),
        },
        raw_stem_paths={
            "Vocals": str(vocals_path),
            "Drums": str(drums_path),
        },
        suggested_names={
            "Vocals": "singing_sep_vocals_song",
            "Backing": "singing_sep_backing_song",
            "All": "singing_sep_song",
        },
        metadata_text="Type: Singing Source Separation",
        metadata_by_stem={
            "Vocals": "Type: Singing Source Separation\nOutput Stem: Vocals",
            "Backing": "Type: Singing Source Separation\nOutput Stem: Backing",
            "Drums": "Type: Singing Source Separation\nOutput Stem: Drums",
        },
        backing_is_synthesized=False,
        status_message="done",
    )

    saved = save_source_separation_outputs(result, tmp_path / "out", "bundle_take")
    saved_names = sorted(path.name for path in saved)
    assert saved_names == [
        "bundle_take_backing.wav",
        "bundle_take_drums.wav",
        "bundle_take_vocals.wav",
    ]

    single = save_source_separation_outputs(
        result,
        tmp_path / "single",
        "lead_only",
        stem_name="Vocals",
    )
    assert single[0].name == "lead_only.wav"

    payload = build_source_separation_ui_payload(result, status_message="done")
    assert payload["vocals_audio"] == str(vocals_path)
    assert payload["backing_audio"] == str(backing_path)
    assert payload["detailed_updates"]["Drums"]["visible"] is True
    assert payload["detailed_updates"]["Bass"]["visible"] is False

    restored = deserialize_source_separation_result(payload["result_json"])
    assert restored is not None
    assert restored.stem_paths["Vocals"] == str(vocals_path)

    route_update = build_source_separation_studio_route_update(str(vocals_path), "Vocals")
    assert route_update[0]["value"] == str(vocals_path)
    assert route_update[1]["selected"] == "singing_studio"
