"""Singing-focused audio processing backend."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
from typing import Callable
import uuid

import librosa
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 44_100
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_TEMPLATE = np.asarray(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
MINOR_TEMPLATE = np.asarray(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)
MAJOR_INTERVALS = {0, 2, 4, 5, 7, 9, 11}
MINOR_INTERVALS = {0, 2, 3, 5, 7, 8, 10}
AUTO_KEY_CONFIDENCE_THRESHOLD = 0.08
DEFAULT_CROSSFADE_MS = 20.0


@dataclass(frozen=True)
class SegmentSelection:
    start_sec: float = 0.0
    end_sec: float = 0.0
    enabled: bool = False


@dataclass(frozen=True)
class AutotuneConfig:
    mode: str = "off"
    strength: float = 1.0
    tonic: str | None = None
    scale: str = "chromatic"
    preserve_formants: bool = True


@dataclass(frozen=True)
class SingingEffectConfig:
    pitch_shift_semitones: float = 0.0
    highpass_hz: float = 0.0
    lowpass_hz: float = 0.0
    compression_amount: float = 0.0
    echo_amount: float = 0.0
    reverb_amount: float = 0.0


@dataclass(frozen=True)
class MixConfig:
    vocal_gain_db: float = 0.0
    backing_gain_db: float = 0.0


@dataclass(frozen=True)
class KeyEstimate:
    tonic: str
    scale: str
    confidence: float
    source: str


@dataclass(frozen=True)
class ResolvedAutotuneResult:
    requested_mode: str
    resolved_mode: str
    tonic: str | None
    scale: str
    confidence: float
    source: str
    message: str
    applied_regions: int = 0


@dataclass(frozen=True)
class SingingProcessResult:
    vocal_output_path: str
    mix_output_path: str | None
    selection: SegmentSelection
    autotune_result: ResolvedAutotuneResult
    status_message: str


ProgressCallback = Callable[[float, str], None] | None


def clamp_segment_selection(
    selection: SegmentSelection,
    total_samples: int,
    sample_rate: int,
) -> SegmentSelection:
    """Clamp selection bounds to audio duration and normalize invalid inputs."""
    if total_samples <= 0 or sample_rate <= 0:
        return SegmentSelection(enabled=False)

    duration = total_samples / float(sample_rate)
    if not selection.enabled:
        return SegmentSelection(start_sec=0.0, end_sec=duration, enabled=False)

    start_sec = max(0.0, float(selection.start_sec or 0.0))
    end_sec = max(0.0, float(selection.end_sec or 0.0))
    start_sec = min(start_sec, duration)
    end_sec = min(end_sec, duration)
    if end_sec <= start_sec:
        return SegmentSelection(start_sec=0.0, end_sec=duration, enabled=False)
    return SegmentSelection(start_sec=start_sec, end_sec=end_sec, enabled=True)


def extract_audio_segment(
    audio: np.ndarray,
    sample_rate: int,
    selection: SegmentSelection,
) -> tuple[np.ndarray, int, int, SegmentSelection]:
    """Extract a segment from audio using clamped bounds."""
    normalized = clamp_segment_selection(selection, len(audio), sample_rate)
    start_idx = int(round(normalized.start_sec * sample_rate))
    end_idx = int(round(normalized.end_sec * sample_rate))
    if end_idx <= start_idx:
        start_idx = 0
        end_idx = len(audio)
        normalized = SegmentSelection(
            start_sec=0.0,
            end_sec=len(audio) / float(sample_rate),
            enabled=False,
        )
    return audio[start_idx:end_idx].copy(), start_idx, end_idx, normalized


def reinsert_processed_segment(
    original_audio: np.ndarray,
    processed_segment: np.ndarray,
    start_idx: int,
    end_idx: int,
    *,
    crossfade_samples: int = 0,
) -> np.ndarray:
    """Replace a segment while smoothing boundaries with short crossfades."""
    base = np.asarray(original_audio, dtype=np.float32).copy()
    if base.ndim != 1:
        raise ValueError("reinsert_processed_segment expects mono 1D audio.")

    start_idx = max(0, min(int(start_idx), len(base)))
    end_idx = max(start_idx, min(int(end_idx), len(base)))
    region_len = end_idx - start_idx
    if region_len <= 0:
        return base

    segment = _fit_to_length(np.asarray(processed_segment, dtype=np.float32), region_len)
    if segment.ndim != 1:
        raise ValueError("processed_segment must be mono 1D audio.")

    fade = int(max(0, min(crossfade_samples, region_len // 2)))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=True, dtype=np.float32)
        segment[:fade] = (base[start_idx : start_idx + fade] * (1.0 - ramp)) + (segment[:fade] * ramp)
        ramp_out = np.linspace(1.0, 0.0, fade, endpoint=True, dtype=np.float32)
        segment[-fade:] = (segment[-fade:] * ramp_out) + (
            base[end_idx - fade : end_idx] * (1.0 - ramp_out)
        )

    base[start_idx:end_idx] = segment
    return base


def quantize_midi_to_scale(midi_note: float, tonic: str | None, scale: str) -> float:
    """Quantize a MIDI note to the nearest note in the target scale."""
    if not math.isfinite(midi_note):
        return midi_note

    normalized_scale = (scale or "chromatic").strip().lower()
    if normalized_scale == "chromatic":
        return float(round(midi_note))

    tonic_name = normalize_tonic_name(tonic)
    if tonic_name is None:
        tonic_name = "C"
    tonic_idx = NOTE_NAMES.index(tonic_name)
    intervals = _scale_intervals(normalized_scale)

    base_octave = int(math.floor(midi_note / 12.0))
    candidates: list[float] = []
    for octave in range(base_octave - 2, base_octave + 3):
        octave_base = octave * 12
        for interval in intervals:
            candidates.append(float(octave_base + tonic_idx + interval))

    return min(candidates, key=lambda candidate: abs(candidate - midi_note))


def estimate_key_from_audio(
    audio_path: str | Path,
    *,
    source_label: str,
    sample_rate: int = TARGET_SAMPLE_RATE,
    max_duration_sec: float = 90.0,
) -> KeyEstimate:
    """Estimate tonic and mode from audio using chroma correlation."""
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if audio.size == 0:
        return KeyEstimate(tonic="C", scale="chromatic", confidence=0.0, source=source_label)

    max_samples = int(max_duration_sec * sample_rate)
    if max_samples > 0:
        audio = audio[:max_samples]

    chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
    chroma_mean = np.asarray(chroma.mean(axis=1), dtype=np.float32)
    if not np.isfinite(chroma_mean).all() or np.allclose(chroma_mean, 0.0):
        return KeyEstimate(tonic="C", scale="chromatic", confidence=0.0, source=source_label)

    scores: list[tuple[float, str, str]] = []
    for shift, note_name in enumerate(NOTE_NAMES):
        major_score = _safe_corrcoef(chroma_mean, np.roll(MAJOR_TEMPLATE, shift))
        minor_score = _safe_corrcoef(chroma_mean, np.roll(MINOR_TEMPLATE, shift))
        scores.append((major_score, note_name, "major"))
        scores.append((minor_score, note_name, "minor"))

    scores.sort(key=lambda item: item[0], reverse=True)
    best_score, tonic, scale = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1.0
    confidence = max(0.0, float(best_score - second_score))
    return KeyEstimate(tonic=tonic, scale=scale, confidence=confidence, source=source_label)


def resolve_autotune_target(
    config: AutotuneConfig,
    *,
    vocal_reference_path: str | Path,
    backing_reference_path: str | Path | None = None,
) -> ResolvedAutotuneResult:
    """Resolve the effective autotune target, including auto-key fallback."""
    requested_mode = normalize_mode(config.mode)
    strength = clamp_unit_interval(config.strength)
    if requested_mode == "off" or strength <= 0.0:
        return ResolvedAutotuneResult(
            requested_mode=requested_mode,
            resolved_mode="off",
            tonic=None,
            scale="chromatic",
            confidence=1.0,
            source="disabled",
            message="Autotune disabled.",
        )

    if requested_mode == "manual":
        tonic = normalize_tonic_name(config.tonic) or "C"
        scale = normalize_scale_name(config.scale)
        return ResolvedAutotuneResult(
            requested_mode=requested_mode,
            resolved_mode="manual",
            tonic=tonic,
            scale=scale,
            confidence=1.0,
            source="manual",
            message=f"Manual autotune locked to {tonic} {scale}.",
        )

    estimates: list[KeyEstimate] = []
    if backing_reference_path:
        estimates.append(
            estimate_key_from_audio(
                backing_reference_path,
                source_label="backing",
            )
        )
    estimates.append(
        estimate_key_from_audio(
            vocal_reference_path,
            source_label="vocal",
        )
    )
    estimates.sort(key=lambda item: item.confidence, reverse=True)
    best = estimates[0]
    if best.confidence >= AUTO_KEY_CONFIDENCE_THRESHOLD:
        return ResolvedAutotuneResult(
            requested_mode=requested_mode,
            resolved_mode="auto",
            tonic=best.tonic,
            scale=best.scale,
            confidence=best.confidence,
            source=best.source,
            message=f"Auto key detection resolved to {best.tonic} {best.scale} from {best.source}.",
        )

    return ResolvedAutotuneResult(
        requested_mode=requested_mode,
        resolved_mode="chromatic_fallback",
        tonic=best.tonic,
        scale="chromatic",
        confidence=best.confidence,
        source=best.source,
        message=(
            "Auto key detection confidence was low; falling back to chromatic correction."
        ),
    )


def build_rubberband_filter(pitch_scale: float, preserve_formants: bool = True) -> str:
    """Build a rubberband filter string suitable for pitch shifts."""
    return (
        "rubberband="
        f"pitch={float(pitch_scale):.10f}:"
        "tempo=1.0:"
        "transients=mixed:"
        "phase=laminar:"
        "window=standard:"
        f"formant={'preserved' if preserve_formants else 'shifted'}:"
        "pitchq=quality"
    )


def build_ffmpeg_effect_filter_chain(
    effects: SingingEffectConfig,
    mix: MixConfig,
    *,
    preserve_formants: bool = True,
) -> str:
    """Build the ordered FFmpeg audio filter chain for the singing effects stage."""
    filters: list[str] = []

    if abs(float(effects.pitch_shift_semitones)) >= 0.01:
        pitch_scale = 2.0 ** (float(effects.pitch_shift_semitones) / 12.0)
        filters.append(build_rubberband_filter(pitch_scale, preserve_formants=preserve_formants))

    if float(effects.highpass_hz) > 0.0:
        filters.append(f"highpass=f={int(round(float(effects.highpass_hz)))}")

    if float(effects.lowpass_hz) > 0.0:
        filters.append(f"lowpass=f={int(round(float(effects.lowpass_hz)))}")

    compression = clamp_unit_interval(effects.compression_amount)
    if compression > 0.0:
        threshold = max(0.05, 0.5 - (compression * 0.35))
        ratio = 1.0 + (compression * 7.0)
        makeup = 1.0 + (compression * 1.5)
        filters.append(
            "acompressor="
            f"threshold={threshold:.3f}:ratio={ratio:.3f}:attack=20:release=250:makeup={makeup:.3f}"
        )

    echo = clamp_unit_interval(effects.echo_amount)
    if echo > 0.0:
        delay1 = int(round(60 + (echo * 120)))
        delay2 = int(round(delay1 * 2))
        decay1 = 0.12 + (echo * 0.20)
        decay2 = 0.08 + (echo * 0.16)
        filters.append(
            f"aecho=0.8:0.88:{delay1}|{delay2}:{decay1:.3f}|{decay2:.3f}"
        )

    if abs(float(mix.vocal_gain_db)) >= 0.01:
        filters.append(f"volume={float(mix.vocal_gain_db):.2f}dB")

    return ",".join(filters)


def process_singing_audio(
    *,
    vocal_path: str | Path,
    selection: SegmentSelection,
    autotune: AutotuneConfig,
    effects: SingingEffectConfig,
    mix: MixConfig,
    temp_dir: str | Path,
    backing_path: str | Path | None = None,
    progress_callback: ProgressCallback = None,
) -> SingingProcessResult:
    """Process a singing vocal track and optionally produce a mix with backing audio."""
    workspace_dir = Path(temp_dir) / f"singing_enhancements_{uuid.uuid4().hex[:10]}"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    _progress(progress_callback, 0.05, "Preparing vocal input...")
    prepared_vocal = convert_audio_to_wav(
        vocal_path,
        workspace_dir=workspace_dir,
        channels=1,
        prefix="vocal_input",
    )
    prepared_backing = None
    if backing_path:
        _progress(progress_callback, 0.12, "Preparing backing track...")
        prepared_backing = convert_audio_to_wav(
            backing_path,
            workspace_dir=workspace_dir,
            channels=2,
            prefix="backing_input",
        )

    vocal_audio, vocal_sr = _load_mono_audio(prepared_vocal)
    segment_audio, start_idx, end_idx, normalized_selection = extract_audio_segment(
        vocal_audio,
        vocal_sr,
        selection,
    )
    if segment_audio.size == 0:
        raise ValueError("Selected segment is empty.")

    _progress(progress_callback, 0.24, "Resolving autotune target...")
    autotune_result, autotuned_segment = apply_autotune_to_segment(
        segment_audio,
        sample_rate=vocal_sr,
        config=autotune,
        workspace_dir=workspace_dir,
        vocal_reference_path=prepared_vocal,
        backing_reference_path=prepared_backing,
        progress_callback=progress_callback,
    )

    _progress(progress_callback, 0.64, "Applying singing effects...")
    autotuned_segment_path = _write_wav(
        workspace_dir / "segment_autotuned.wav",
        autotuned_segment,
        vocal_sr,
    )
    processed_segment_path = apply_effects_chain(
        autotuned_segment_path,
        workspace_dir=workspace_dir,
        effects=effects,
        mix=mix,
        preserve_formants=autotune.preserve_formants,
    )
    processed_segment_audio, _ = _load_mono_audio(processed_segment_path)

    _progress(progress_callback, 0.78, "Reinserting processed segment...")
    crossfade_samples = int(round((DEFAULT_CROSSFADE_MS / 1000.0) * vocal_sr))
    processed_vocal_audio = reinsert_processed_segment(
        vocal_audio,
        processed_segment_audio,
        start_idx,
        end_idx,
        crossfade_samples=crossfade_samples,
    )
    processed_vocal_path = _write_wav(
        workspace_dir / "singing_processed_vocal.wav",
        processed_vocal_audio,
        vocal_sr,
    )

    mix_output_path: str | None = None
    if prepared_backing:
        _progress(progress_callback, 0.9, "Mixing vocal with backing track...")
        mix_output_path = mix_vocal_with_backing(
            processed_vocal_path,
            prepared_backing,
            workspace_dir=workspace_dir,
            mix=mix,
        )

    status_parts = [
        autotune_result.message,
        f"Processed selection: {normalized_selection.start_sec:.2f}s to {normalized_selection.end_sec:.2f}s.",
    ]
    if mix_output_path:
        status_parts.append("Generated both processed vocal and final mix.")
    else:
        status_parts.append("Generated processed vocal only.")

    _progress(progress_callback, 1.0, "Done.")
    return SingingProcessResult(
        vocal_output_path=str(processed_vocal_path),
        mix_output_path=str(mix_output_path) if mix_output_path else None,
        selection=normalized_selection,
        autotune_result=autotune_result,
        status_message=" ".join(status_parts),
    )


def apply_autotune_to_segment(
    audio_segment: np.ndarray,
    *,
    sample_rate: int,
    config: AutotuneConfig,
    workspace_dir: str | Path,
    vocal_reference_path: str | Path,
    backing_reference_path: str | Path | None = None,
    progress_callback: ProgressCallback = None,
) -> tuple[ResolvedAutotuneResult, np.ndarray]:
    """Apply note-region pitch correction to a vocal segment."""
    normalized_audio = np.asarray(audio_segment, dtype=np.float32)
    segment_duration_sec = len(normalized_audio) / float(sample_rate) if sample_rate > 0 else 0.0
    resolved = resolve_autotune_target(
        config,
        vocal_reference_path=vocal_reference_path,
        backing_reference_path=backing_reference_path,
    )
    if resolved.resolved_mode == "off":
        return resolved, normalized_audio.copy()

    hop_length = 256
    frame_length = 2048
    _progress(
        progress_callback,
        0.27,
        f"Analyzing pitch contour ({segment_duration_sec:.1f}s selection)...",
    )
    print(
        f"[Singing Enhancements] Pitch analysis started for {segment_duration_sec:.2f}s selection "
        f"(mode={resolved.resolved_mode}, tonic={resolved.tonic or 'N/A'}, scale={resolved.scale})",
        flush=True,
    )
    f0, voiced_flag, _ = librosa.pyin(
        normalized_audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    _progress(progress_callback, 0.5, "Grouping stable notes...")
    midi = librosa.hz_to_midi(f0)
    voiced_mask = np.isfinite(midi) & np.asarray(voiced_flag, dtype=bool)
    if not voiced_mask.any():
        print("[Singing Enhancements] Pitch analysis finished: no voiced notes detected.", flush=True)
        return (
            ResolvedAutotuneResult(
                requested_mode=resolved.requested_mode,
                resolved_mode=resolved.resolved_mode,
                tonic=resolved.tonic,
                scale=resolved.scale,
                confidence=resolved.confidence,
                source=resolved.source,
                message=f"{resolved.message} No voiced notes detected in the selected segment.",
                applied_regions=0,
            ),
            normalized_audio.copy(),
        )

    corrected = normalized_audio.copy()
    regions = detect_stable_voiced_regions(
        midi,
        voiced_mask,
        hop_length=hop_length,
        frame_length=frame_length,
        sample_rate=sample_rate,
    )
    print(
        f"[Singing Enhancements] Pitch analysis finished: {len(regions)} stable voiced regions detected.",
        flush=True,
    )
    applied_regions = 0
    for region_index, (start_idx, end_idx, region_midi) in enumerate(regions, start=1):
        target_midi = quantize_midi_to_scale(region_midi, resolved.tonic, resolved.scale)
        semitone_shift = (target_midi - region_midi) * clamp_unit_interval(config.strength)
        if not math.isfinite(semitone_shift) or abs(semitone_shift) < 0.05:
            continue

        region_audio = corrected[start_idx:end_idx]
        if region_audio.size < int(sample_rate * 0.06):
            continue

        _progress(
            progress_callback,
            min(0.24 + (region_index / max(len(regions), 1)) * 0.30, 0.62),
            f"Autotuning region {region_index}/{len(regions)}...",
        )
        try:
            shifted = pitch_shift_audio_array(
                region_audio,
                sample_rate=sample_rate,
                semitone_shift=semitone_shift,
                workspace_dir=workspace_dir,
                preserve_formants=config.preserve_formants,
            )
        except RuntimeError:
            continue

        corrected = reinsert_processed_segment(
            corrected,
            shifted,
            start_idx,
            end_idx,
            crossfade_samples=min(256, max(32, (end_idx - start_idx) // 8)),
        )
        applied_regions += 1

    if applied_regions <= 0:
        message = f"{resolved.message} No stable voiced regions required correction."
    else:
        message = f"{resolved.message} Corrected {applied_regions} voiced regions."

    return (
        ResolvedAutotuneResult(
            requested_mode=resolved.requested_mode,
            resolved_mode=resolved.resolved_mode,
            tonic=resolved.tonic,
            scale=resolved.scale,
            confidence=resolved.confidence,
            source=resolved.source,
            message=message,
            applied_regions=applied_regions,
        ),
        corrected,
    )


def detect_stable_voiced_regions(
    midi_values: np.ndarray,
    voiced_mask: np.ndarray,
    *,
    hop_length: int,
    frame_length: int,
    sample_rate: int,
    max_jump_semitones: float = 0.75,
    min_duration_sec: float = 0.08,
) -> list[tuple[int, int, float]]:
    """Group contiguous voiced frames with limited pitch movement."""
    midi_values = np.asarray(midi_values, dtype=np.float32)
    voiced_mask = np.asarray(voiced_mask, dtype=bool)
    regions: list[tuple[int, int, float]] = []

    start_frame: int | None = None
    last_frame: int | None = None
    collected: list[float] = []

    def flush_region():
        nonlocal start_frame, last_frame, collected
        if start_frame is None or last_frame is None or not collected:
            start_frame = None
            last_frame = None
            collected = []
            return

        start_sample = int(start_frame * hop_length)
        end_sample = int(last_frame * hop_length + frame_length)
        duration = (end_sample - start_sample) / float(sample_rate)
        if duration >= min_duration_sec:
            regions.append((start_sample, end_sample, float(np.median(collected))))

        start_frame = None
        last_frame = None
        collected = []

    for frame_idx, is_voiced in enumerate(voiced_mask):
        if not is_voiced or not math.isfinite(float(midi_values[frame_idx])):
            flush_region()
            continue

        current_midi = float(midi_values[frame_idx])
        if start_frame is None:
            start_frame = frame_idx
            last_frame = frame_idx
            collected = [current_midi]
            continue

        assert last_frame is not None
        previous_midi = collected[-1]
        contiguous = frame_idx == (last_frame + 1)
        stable = abs(current_midi - previous_midi) <= max_jump_semitones
        if contiguous and stable:
            collected.append(current_midi)
            last_frame = frame_idx
            continue

        flush_region()
        start_frame = frame_idx
        last_frame = frame_idx
        collected = [current_midi]

    flush_region()
    return regions


def pitch_shift_audio_array(
    audio: np.ndarray,
    *,
    sample_rate: int,
    semitone_shift: float,
    workspace_dir: str | Path,
    preserve_formants: bool = True,
) -> np.ndarray:
    """Pitch shift a mono array using FFmpeg rubberband."""
    if abs(float(semitone_shift)) < 0.01:
        return np.asarray(audio, dtype=np.float32).copy()

    workspace = Path(workspace_dir)
    input_path = _write_wav(
        workspace / f"pitch_in_{uuid.uuid4().hex[:8]}.wav",
        np.asarray(audio, dtype=np.float32),
        sample_rate,
    )
    output_path = workspace / f"pitch_out_{uuid.uuid4().hex[:8]}.wav"
    pitch_scale = 2.0 ** (float(semitone_shift) / 12.0)
    filter_chain = build_rubberband_filter(pitch_scale, preserve_formants=preserve_formants)
    _run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-af",
            filter_chain,
            str(output_path),
        ],
        step_name="Pitch shift",
    )
    shifted, _ = _load_mono_audio(output_path)
    return _fit_to_length(shifted, len(audio))


def apply_effects_chain(
    input_path: str | Path,
    *,
    workspace_dir: str | Path,
    effects: SingingEffectConfig,
    mix: MixConfig,
    preserve_formants: bool = True,
) -> Path:
    """Apply the ordered singing effects chain to a mono vocal file."""
    workspace = Path(workspace_dir)
    current_path = Path(input_path)
    filter_chain = build_ffmpeg_effect_filter_chain(
        effects,
        mix,
        preserve_formants=preserve_formants,
    )

    if filter_chain:
        ffmpeg_out = workspace / "segment_fx_ffmpeg.wav"
        _run_command(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(current_path),
                "-af",
                filter_chain,
                str(ffmpeg_out),
            ],
            step_name="Effects chain",
        )
        current_path = ffmpeg_out

    reverb = clamp_unit_interval(effects.reverb_amount)
    if reverb > 0.0:
        sox_out = workspace / "segment_fx_reverb.wav"
        reverberance = int(round(20 + (reverb * 80)))
        room_scale = int(round(25 + (reverb * 60)))
        wet_gain = int(round(-4 + (reverb * 4)))
        _run_command(
            [
                "sox",
                str(current_path),
                str(sox_out),
                "reverb",
                str(reverberance),
                "50",
                str(room_scale),
                "100",
                "0",
                str(wet_gain),
            ],
            step_name="Reverb",
        )
        current_path = sox_out

    return current_path


def mix_vocal_with_backing(
    vocal_path: str | Path,
    backing_path: str | Path,
    *,
    workspace_dir: str | Path,
    mix: MixConfig,
) -> str:
    """Mix processed vocal with backing audio while aligning sample rate, length, and channels."""
    vocal_audio, _ = _load_audio(vocal_path, mono=False)
    backing_audio, _ = _load_audio(backing_path, mono=False)

    vocal_audio = _ensure_2d(vocal_audio)
    backing_audio = _ensure_2d(backing_audio)

    output_channels = max(vocal_audio.shape[0], backing_audio.shape[0])
    vocal_audio = _match_channels(vocal_audio, output_channels)
    backing_audio = _match_channels(backing_audio, output_channels)

    target_length = max(vocal_audio.shape[1], backing_audio.shape[1])
    vocal_audio = _pad_audio(vocal_audio, target_length)
    backing_audio = _pad_audio(backing_audio, target_length)

    backing_gain = db_to_linear(mix.backing_gain_db)
    mixed = vocal_audio + (backing_audio * backing_gain)
    mixed = np.clip(mixed, -1.0, 1.0)

    output_path = Path(workspace_dir) / "singing_final_mix.wav"
    sf.write(str(output_path), mixed.T, TARGET_SAMPLE_RATE)
    return str(output_path)


def convert_audio_to_wav(
    source_path: str | Path,
    *,
    workspace_dir: str | Path,
    channels: int,
    prefix: str,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> str:
    """Convert any supported input audio into a deterministic WAV format."""
    output_path = Path(workspace_dir) / f"{prefix}_{uuid.uuid4().hex[:8]}.wav"
    _run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            str(output_path),
        ],
        step_name=f"Convert {prefix}",
    )
    return str(output_path)


def normalize_mode(mode: str | None) -> str:
    text = (mode or "off").strip().lower()
    if text in {"off", "auto", "manual"}:
        return text
    return "off"


def normalize_tonic_name(tonic: str | None) -> str | None:
    if tonic is None:
        return None
    raw = tonic.strip().upper().replace("B#", "C").replace("E#", "F")
    raw = raw.replace("DB", "C#").replace("EB", "D#").replace("GB", "F#")
    raw = raw.replace("AB", "G#").replace("BB", "A#")
    if raw in NOTE_NAMES:
        return raw
    return None


def normalize_scale_name(scale: str | None) -> str:
    text = (scale or "chromatic").strip().lower()
    if text in {"major", "minor", "chromatic"}:
        return text
    return "chromatic"


def clamp_unit_interval(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def db_to_linear(db_value: float) -> float:
    return float(10.0 ** (float(db_value) / 20.0))


def _run_command(cmd: list[str], *, step_name: str) -> None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        binary = cmd[0]
        raise FileNotFoundError(f"{binary} not found while running {step_name.lower()}.") from exc

    if result.returncode == 0:
        return

    message = (result.stderr or result.stdout or "Unknown error").strip()
    message = message.splitlines()[-1][:240] if message else "Unknown error"
    raise RuntimeError(f"{step_name} failed: {message}")


def _load_audio(audio_path: str | Path, *, mono: bool) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(audio_path), always_2d=False)
    audio = np.asarray(data, dtype=np.float32)
    if mono and audio.ndim > 1:
        audio = audio.mean(axis=1)
    elif not mono and audio.ndim == 1:
        audio = audio[np.newaxis, :]
    elif not mono and audio.ndim == 2:
        audio = audio.T
    return audio, int(sample_rate)


def _load_mono_audio(audio_path: str | Path) -> tuple[np.ndarray, int]:
    return _load_audio(audio_path, mono=True)


def _write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> str:
    data = np.asarray(audio, dtype=np.float32)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        sf.write(str(output_path), data.T, sample_rate)
    else:
        sf.write(str(output_path), data, sample_rate)
    return str(output_path)


def _fit_to_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(data) == target_length:
        return data
    if len(data) > target_length:
        return data[:target_length]
    return np.pad(data, (0, target_length - len(data)))


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return -1.0
    value = float(np.corrcoef(a, b)[0, 1])
    if math.isnan(value):
        return -1.0
    return value


def _scale_intervals(scale: str) -> list[int]:
    if scale == "major":
        return sorted(MAJOR_INTERVALS)
    if scale == "minor":
        return sorted(MINOR_INTERVALS)
    return list(range(12))


def _ensure_2d(audio: np.ndarray) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        return data[np.newaxis, :]
    if data.ndim == 2:
        return data
    raise ValueError("Audio array must be 1D or 2D.")


def _match_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    if audio.shape[0] == channels:
        return audio
    if audio.shape[0] == 1 and channels > 1:
        return np.repeat(audio, channels, axis=0)
    if channels == 1:
        return audio.mean(axis=0, keepdims=True)
    return audio[:channels, :]


def _pad_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    if audio.shape[1] >= target_length:
        return audio[:, :target_length]
    pad_width = target_length - audio.shape[1]
    return np.pad(audio, ((0, 0), (0, pad_width)))


def _progress(callback: ProgressCallback, value: float, desc: str) -> None:
    if callback is not None:
        callback(value, desc=desc)


__all__ = [
    "TARGET_SAMPLE_RATE",
    "NOTE_NAMES",
    "SegmentSelection",
    "AutotuneConfig",
    "SingingEffectConfig",
    "MixConfig",
    "KeyEstimate",
    "ResolvedAutotuneResult",
    "SingingProcessResult",
    "clamp_segment_selection",
    "extract_audio_segment",
    "reinsert_processed_segment",
    "quantize_midi_to_scale",
    "estimate_key_from_audio",
    "resolve_autotune_target",
    "build_rubberband_filter",
    "build_ffmpeg_effect_filter_chain",
    "detect_stable_voiced_regions",
    "apply_autotune_to_segment",
    "apply_effects_chain",
    "mix_vocal_with_backing",
    "convert_audio_to_wav",
    "process_singing_audio",
]
