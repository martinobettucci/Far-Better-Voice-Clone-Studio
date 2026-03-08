from pathlib import Path

import soundfile as sf

from modules.core_components.tools.output_audio_pipeline import (
    OutputAudioPipelineConfig,
    apply_generation_output_pipeline,
)


def _write_wav(path: Path) -> str:
    sf.write(str(path), [0.0, 0.1, -0.1, 0.0], 24000)
    return str(path)


def test_pipeline_order_is_denoise_remove_silences_normalize_mono(tmp_path: Path):
    src = _write_wav(tmp_path / "input.wav")
    calls = []

    def denoise(path: str):
        calls.append(("denoise", Path(path).name))
        return _write_wav(tmp_path / "denoise.wav"), "ok"

    def remove_silences(path: str):
        calls.append(("remove_silences", Path(path).name))
        return _write_wav(tmp_path / "nosilence.wav"), "ok"

    def normalize(path: str):
        calls.append(("normalize", Path(path).name))
        return _write_wav(tmp_path / "normalize.wav"), "ok"

    def mono(path: str):
        calls.append(("mono", Path(path).name))
        return _write_wav(tmp_path / "mono.wav"), "ok"

    out, status = apply_generation_output_pipeline(
        src,
        OutputAudioPipelineConfig(
            enable_denoise=True,
            enable_remove_silences=True,
            enable_normalize=True,
            enable_mono=True,
        ),
        deepfilter_available=True,
        denoise_step=denoise,
        remove_silences_step=remove_silences,
        normalize_step=normalize,
        mono_step=mono,
    )

    assert out == str(tmp_path / "mono.wav")
    assert calls == [
        ("denoise", "input.wav"),
        ("remove_silences", "denoise.wav"),
        ("normalize", "nosilence.wav"),
        ("mono", "normalize.wav"),
    ]
    assert status == "Pipeline applied to output audio: Denoise -> Remove Silences -> Normalize -> Mono"


def test_pipeline_no_steps_enabled_is_passthrough(tmp_path: Path):
    src = _write_wav(tmp_path / "input.wav")
    calls = []

    def never_called(_path: str):
        calls.append("called")
        return None, "should not run"

    out, status = apply_generation_output_pipeline(
        {"path": src},
        OutputAudioPipelineConfig(),
        deepfilter_available=True,
        denoise_step=never_called,
        remove_silences_step=never_called,
        normalize_step=never_called,
        mono_step=never_called,
    )

    assert out == src
    assert status == "No pipeline steps enabled. Using current output."
    assert calls == []


def test_pipeline_handles_denoise_unavailable(tmp_path: Path):
    src = _write_wav(tmp_path / "input.wav")
    calls = []

    def never_called(_path: str):
        calls.append("called")
        return None, "should not run"

    out, status = apply_generation_output_pipeline(
        src,
        OutputAudioPipelineConfig(enable_denoise=True),
        deepfilter_available=False,
        denoise_step=never_called,
        remove_silences_step=never_called,
        normalize_step=never_called,
        mono_step=never_called,
    )

    assert out == src
    assert status == "Denoise unavailable in this environment"
    assert calls == []


def test_pipeline_failure_short_circuits_after_first_error(tmp_path: Path):
    src = _write_wav(tmp_path / "input.wav")
    calls = []

    def denoise(path: str):
        calls.append(("denoise", Path(path).name))
        return _write_wav(tmp_path / "denoise.wav"), "ok"

    def remove_silences(path: str):
        calls.append(("remove_silences", Path(path).name))
        return _write_wav(tmp_path / "nosilence.wav"), "ok"

    def normalize(path: str):
        calls.append(("normalize", Path(path).name))
        return None, "Normalize failed"

    def mono(path: str):
        calls.append(("mono", Path(path).name))
        return _write_wav(tmp_path / "mono.wav"), "ok"

    out, status = apply_generation_output_pipeline(
        src,
        OutputAudioPipelineConfig(
            enable_denoise=True,
            enable_remove_silences=True,
            enable_normalize=True,
            enable_mono=True,
        ),
        deepfilter_available=True,
        denoise_step=denoise,
        remove_silences_step=remove_silences,
        normalize_step=normalize,
        mono_step=mono,
    )

    assert out == src
    assert status == "Normalize failed"
    assert calls == [
        ("denoise", "input.wav"),
        ("remove_silences", "denoise.wav"),
        ("normalize", "nosilence.wav"),
    ]


def test_pipeline_treats_warning_message_as_failure(tmp_path: Path):
    src = _write_wav(tmp_path / "input.wav")
    calls = []

    def denoise(path: str):
        calls.append(("denoise", Path(path).name))
        return _write_wav(tmp_path / "denoise.wav"), "⚠ Error cleaning audio"

    out, status = apply_generation_output_pipeline(
        src,
        OutputAudioPipelineConfig(enable_denoise=True, enable_normalize=True),
        deepfilter_available=True,
        denoise_step=denoise,
        remove_silences_step=lambda path: (_write_wav(tmp_path / "nosilence.wav"), "ok"),
        normalize_step=lambda path: (_write_wav(tmp_path / "normalize.wav"), "ok"),
        mono_step=lambda path: (_write_wav(tmp_path / "mono.wav"), "ok"),
    )

    assert out == src
    assert status == "⚠ Error cleaning audio"
    assert calls == [("denoise", "input.wav")]


def test_pipeline_invalid_or_missing_audio_input():
    out, status = apply_generation_output_pipeline(
        "",
        OutputAudioPipelineConfig(enable_normalize=True),
        deepfilter_available=True,
        denoise_step=lambda p: (p, "ok"),
        remove_silences_step=lambda p: (p, "ok"),
        normalize_step=lambda p: (p, "ok"),
        mono_step=lambda p: (p, "ok"),
    )

    assert out is None
    assert status == "No generated audio to process."
