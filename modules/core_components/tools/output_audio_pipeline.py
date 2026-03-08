"""Shared manual output-audio pipeline for generation tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


AudioStep = Callable[[str], tuple[str | None, str | None]]


@dataclass(frozen=True)
class OutputAudioPipelineConfig:
    """Manual output pipeline toggles applied in fixed order."""

    enable_denoise: bool = False
    enable_remove_silences: bool = False
    enable_normalize: bool = False
    enable_mono: bool = False


def _as_audio_path(audio_value: Any) -> str | None:
    if isinstance(audio_value, str) and audio_value.strip():
        return audio_value
    if isinstance(audio_value, dict):
        maybe_path = audio_value.get("path") or audio_value.get("name")
        if isinstance(maybe_path, str) and maybe_path.strip():
            return maybe_path
    return None


def _run_step(
    step_name: str,
    step_fn: AudioStep,
    current_audio: str,
) -> tuple[str | None, str | None]:
    try:
        next_audio, step_msg = step_fn(current_audio)
    except Exception as exc:
        return None, f"{step_name} failed: {str(exc)}"
    if not next_audio:
        return None, step_msg or f"{step_name} failed"
    if _message_indicates_failure(step_msg):
        return None, str(step_msg)
    return next_audio, None


def _message_indicates_failure(step_msg: str | None) -> bool:
    if step_msg is None:
        return False
    text = str(step_msg).strip().lower()
    if not text:
        return False
    return text.startswith("⚠") or text.startswith("error") or "failed" in text


def apply_generation_output_pipeline(
    audio_value: Any,
    pipeline: OutputAudioPipelineConfig,
    *,
    deepfilter_available: bool,
    denoise_step: AudioStep,
    remove_silences_step: AudioStep,
    normalize_step: AudioStep,
    mono_step: AudioStep,
) -> tuple[str | None, str]:
    """Apply output pipeline in fixed order: denoise -> remove silences -> normalize -> mono."""
    audio_path = _as_audio_path(audio_value)
    if not audio_path:
        return None, "No generated audio to process."

    if not Path(audio_path).exists():
        return None, "Generated audio file not found."

    if not (
        pipeline.enable_denoise
        or pipeline.enable_remove_silences
        or pipeline.enable_normalize
        or pipeline.enable_mono
    ):
        return audio_path, "No pipeline steps enabled. Using current output."

    current_audio = audio_path
    applied_steps: list[str] = []

    if pipeline.enable_denoise:
        if not deepfilter_available:
            return current_audio, "Denoise unavailable in this environment"
        current_audio, error = _run_step("Denoise", denoise_step, current_audio)
        if error:
            return audio_path, error
        applied_steps.append("Denoise")

    if pipeline.enable_remove_silences:
        current_audio, error = _run_step("Remove silences", remove_silences_step, current_audio)
        if error:
            return audio_path, error
        applied_steps.append("Remove Silences")

    if pipeline.enable_normalize:
        current_audio, error = _run_step("Normalize", normalize_step, current_audio)
        if error:
            return audio_path, error
        applied_steps.append("Normalize")

    if pipeline.enable_mono:
        current_audio, error = _run_step("Mono conversion", mono_step, current_audio)
        if error:
            return audio_path, error
        applied_steps.append("Mono")

    return current_audio, f"Pipeline applied to output audio: {' -> '.join(applied_steps)}"
