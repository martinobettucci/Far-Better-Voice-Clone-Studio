from pathlib import Path
import shutil

import numpy as np
import pytest
import soundfile as sf

from modules.core_components.tools.singing_audio_pipeline import (
    AutotuneConfig,
    KeyEstimate,
    MixConfig,
    SegmentSelection,
    SingingEffectConfig,
    TARGET_SAMPLE_RATE,
    build_ffmpeg_effect_filter_chain,
    extract_audio_segment,
    mix_vocal_with_backing,
    process_singing_audio,
    quantize_midi_to_scale,
    reinsert_processed_segment,
    resolve_autotune_target,
)


def test_extract_and_reinsert_segment_preserves_length():
    sample_rate = 1000
    original = np.linspace(-1.0, 1.0, sample_rate, dtype=np.float32)
    selection = SegmentSelection(start_sec=0.2, end_sec=0.5, enabled=True)

    segment, start_idx, end_idx, normalized = extract_audio_segment(original, sample_rate, selection)
    processed = np.full_like(segment, 0.25)
    combined = reinsert_processed_segment(
        original,
        processed,
        start_idx,
        end_idx,
        crossfade_samples=0,
    )

    assert len(segment) == 300
    assert normalized.enabled is True
    assert len(combined) == len(original)
    np.testing.assert_allclose(combined[:start_idx], original[:start_idx])
    np.testing.assert_allclose(combined[end_idx:], original[end_idx:])
    np.testing.assert_allclose(combined[start_idx:end_idx], processed)


def test_quantize_midi_to_scale_uses_manual_key():
    quantized = quantize_midi_to_scale(70.4, "C", "major")
    assert quantized == pytest.approx(71.0)


def test_resolve_autotune_target_falls_back_to_chromatic(monkeypatch):
    def fake_estimate(_audio_path, *, source_label, sample_rate=TARGET_SAMPLE_RATE, max_duration_sec=90.0):
        return KeyEstimate(tonic="D", scale="minor", confidence=0.02, source=source_label)

    monkeypatch.setattr(
        "modules.core_components.tools.singing_audio_pipeline.estimate_key_from_audio",
        fake_estimate,
    )

    result = resolve_autotune_target(
        AutotuneConfig(mode="auto", strength=1.0),
        vocal_reference_path="lead.wav",
        backing_reference_path="backing.wav",
    )

    assert result.resolved_mode == "chromatic_fallback"
    assert result.scale == "chromatic"
    assert result.message.startswith("Auto key detection confidence was low")


def test_build_ffmpeg_effect_chain_contains_expected_order():
    chain = build_ffmpeg_effect_filter_chain(
        SingingEffectConfig(
            pitch_shift_semitones=2.0,
            highpass_hz=120.0,
            lowpass_hz=9000.0,
            compression_amount=0.5,
            echo_amount=0.25,
            reverb_amount=0.0,
        ),
        MixConfig(vocal_gain_db=3.0, backing_gain_db=-6.0),
        preserve_formants=True,
    )

    parts = chain.split(",")
    assert parts[0].startswith("rubberband=")
    assert parts[1] == "highpass=f=120"
    assert parts[2] == "lowpass=f=9000"
    assert parts[3].startswith("acompressor=")
    assert parts[4].startswith("aecho=")
    assert parts[5] == "volume=3.00dB"


def test_mix_vocal_with_backing_aligns_length_and_channels(tmp_path: Path):
    vocal = np.sin(np.linspace(0, np.pi * 6, int(TARGET_SAMPLE_RATE * 0.20), dtype=np.float32))
    backing_length = int(TARGET_SAMPLE_RATE * 0.35)
    backing_left = 0.2 * np.sin(np.linspace(0, np.pi * 12, backing_length, dtype=np.float32))
    backing_right = 0.2 * np.cos(np.linspace(0, np.pi * 12, backing_length, dtype=np.float32))
    backing = np.stack([backing_left, backing_right], axis=1)

    vocal_path = tmp_path / "vocal.wav"
    backing_path = tmp_path / "backing.wav"
    sf.write(str(vocal_path), vocal, TARGET_SAMPLE_RATE)
    sf.write(str(backing_path), backing, TARGET_SAMPLE_RATE)

    output_path = mix_vocal_with_backing(
        vocal_path,
        backing_path,
        workspace_dir=tmp_path,
        mix=MixConfig(vocal_gain_db=6.0, backing_gain_db=-3.0),
    )

    mixed, sample_rate = sf.read(output_path)
    assert sample_rate == TARGET_SAMPLE_RATE
    assert mixed.shape[0] == backing_length
    assert mixed.ndim == 2
    assert mixed.shape[1] == 2


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("sox") is None,
    reason="ffmpeg and sox are required for end-to-end singing pipeline coverage",
)
def test_process_singing_audio_end_to_end(tmp_path: Path):
    duration_sec = 0.6
    total_samples = int(TARGET_SAMPLE_RATE * duration_sec)
    time = np.arange(total_samples, dtype=np.float32) / TARGET_SAMPLE_RATE

    vocal = 0.35 * np.sin(2 * np.pi * 466.16 * time).astype(np.float32)
    backing_left = 0.15 * np.sin(2 * np.pi * 261.63 * time).astype(np.float32)
    backing_right = 0.12 * np.sin(2 * np.pi * 329.63 * time).astype(np.float32)
    backing = np.stack([backing_left, backing_right], axis=1)

    vocal_path = tmp_path / "lead.wav"
    backing_path = tmp_path / "backing.wav"
    sf.write(str(vocal_path), vocal, TARGET_SAMPLE_RATE)
    sf.write(str(backing_path), backing, TARGET_SAMPLE_RATE)

    result = process_singing_audio(
        vocal_path=vocal_path,
        backing_path=backing_path,
        selection=SegmentSelection(start_sec=0.1, end_sec=0.45, enabled=True),
        autotune=AutotuneConfig(
            mode="manual",
            strength=1.0,
            tonic="C",
            scale="major",
            preserve_formants=True,
        ),
        effects=SingingEffectConfig(
            pitch_shift_semitones=0.0,
            highpass_hz=80.0,
            lowpass_hz=0.0,
            compression_amount=0.15,
            echo_amount=0.05,
            reverb_amount=0.05,
        ),
        mix=MixConfig(vocal_gain_db=0.0, backing_gain_db=-4.0),
        temp_dir=tmp_path,
    )

    assert Path(result.vocal_output_path).exists()
    assert result.mix_output_path is not None
    assert Path(result.mix_output_path).exists()
    assert result.selection.enabled is True

    vocal_out, vocal_sr = sf.read(result.vocal_output_path)
    mix_out, mix_sr = sf.read(result.mix_output_path)

    assert vocal_sr == TARGET_SAMPLE_RATE
    assert mix_sr == TARGET_SAMPLE_RATE
    assert vocal_out.shape[0] == total_samples
    assert mix_out.shape[0] == total_samples
    assert mix_out.ndim == 2
    assert mix_out.shape[1] == 2
