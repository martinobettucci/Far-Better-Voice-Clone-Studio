"""Reusable processing helpers for the unified Library Manager workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .tenant_storage import sanitize_filename


LANGUAGE_CODE_MAP = {
    "English": "en",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "German": "de",
    "French": "fr",
    "Russian": "ru",
    "Portuguese": "pt",
    "Spanish": "es",
    "Italian": "it",
}


def parse_asr_model(model_str: str) -> tuple[str, str | None]:
    """Parse unified ASR dropdown value into (engine, size)."""
    if " - " in model_str:
        engine, size = model_str.rsplit(" - ", 1)
        return engine, (size if size != "Default" else None)
    return model_str, None


def language_to_code(language: str | None) -> str | None:
    if not language or language == "Auto-detect":
        return None
    return LANGUAGE_CODE_MAP.get(language)


def clean_transcription_for_engine(engine: str, text: str) -> str:
    """Normalize ASR output text for downstream sample/dataset saving."""
    cleaned = (text or "").strip()
    if engine == "VibeVoice ASR":
        cleaned = re.sub(r"\[.*?\]\s*:", "", cleaned)
        cleaned = re.sub(r"\[.*?\]", "", cleaned)
        cleaned = " ".join(cleaned.split())
    return cleaned


def estimate_pcm16_wav_bytes(sample_count: int, channels: int = 1) -> int:
    """Rough output size estimate for PCM16 WAV payload + header."""
    sample_count = max(int(sample_count), 0)
    channels = max(int(channels), 1)
    # 16-bit PCM = 2 bytes per channel/sample + typical 44-byte WAV header.
    return (sample_count * channels * 2) + 44


@dataclass(frozen=True)
class WordTimestampLike:
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class ProcessingSourceContext:
    """Processing source metadata shown in Processing Studio summary."""

    source_type: str
    source_identifier: str
    original_audio_path: str
    source_dataset_folder: str = ""


@dataclass(frozen=True)
class ProcessingPipelineConfig:
    """Pipeline toggles applied in deterministic order from original source."""

    enable_denoise: bool = False
    enable_normalize: bool = False
    enable_mono: bool = False


@dataclass(frozen=True)
class SpeakerSeparatedSampleImport:
    """Planned sample import for one separated speaker track."""

    sample_name: str
    speaker_index: int
    sample_rate: int
    estimated_size_bytes: int
    metadata: dict


def build_routed_processing_context(audio_path: str) -> ProcessingSourceContext:
    """Build processing context for routed/generated audio."""
    resolved_path = str(audio_path or "").strip()
    name = Path(resolved_path).name if resolved_path else "routed_audio.wav"
    return ProcessingSourceContext(
        source_type="External",
        source_identifier=name,
        original_audio_path=resolved_path,
        source_dataset_folder="",
    )


def resolve_processing_save_destination(action_kind: str, source_kind: str) -> str:
    """Resolve whether a processing save action targets samples or datasets."""
    source_kind = (source_kind or "").lower()
    if source_kind == "sample":
        return "sample" if action_kind == "primary" else "dataset"
    if source_kind == "dataset":
        return "dataset" if action_kind == "primary" else "sample"
    return "sample" if action_kind == "primary" else "dataset"


def build_speaker_separation_sample_name(base_name: str, speaker_index: int) -> str:
    """Build deterministic sample names for imported speaker tracks."""
    safe_base = sanitize_filename(base_name or "sample", keep_extension=False)
    return f"{safe_base}_speaker_{int(speaker_index)}"


def build_speaker_separation_sample_metadata(
    sample_name: str,
    *,
    source_identifier: str,
    expected_speakers: int,
    speaker_index: int,
    backend: str = "SpeechBrain SepFormer",
) -> dict:
    """Build metadata for an imported separated speaker sample."""
    final_name = sanitize_filename(sample_name or "sample", keep_extension=False)
    return {
        "Name": final_name,
        "name": final_name,
        "Type": "Sample",
        "Text": "",
        "text": "",
        "Source": str(source_identifier or "").strip(),
        "SeparationBackend": str(backend or "SpeechBrain SepFormer").strip(),
        "ExpectedSpeakers": int(expected_speakers),
        "SpeakerIndex": int(speaker_index),
    }


def build_speaker_separation_import_plan(
    base_name: str,
    separated_tracks: Iterable[object],
    *,
    source_identifier: str,
    expected_speakers: int,
    backend: str = "SpeechBrain SepFormer",
) -> list[SpeakerSeparatedSampleImport]:
    """Build deterministic sample-import plans for separated speaker tracks."""
    plans: list[SpeakerSeparatedSampleImport] = []
    for default_index, track in enumerate(separated_tracks or [], start=1):
        speaker_index = int(getattr(track, "speaker_index", default_index))
        sample_rate = int(getattr(track, "sample_rate", 0) or 0)
        audio_data = np.asarray(getattr(track, "audio_data", []), dtype=np.float32).reshape(-1)
        sample_name = build_speaker_separation_sample_name(base_name, speaker_index)
        metadata = build_speaker_separation_sample_metadata(
            sample_name,
            source_identifier=source_identifier,
            expected_speakers=expected_speakers,
            speaker_index=speaker_index,
            backend=backend,
        )
        plans.append(
            SpeakerSeparatedSampleImport(
                sample_name=sample_name,
                speaker_index=speaker_index,
                sample_rate=sample_rate,
                estimated_size_bytes=estimate_pcm16_wav_bytes(len(audio_data), channels=1),
                metadata=metadata,
            )
        )
    return plans


def split_into_segments(
    full_text: str,
    word_timestamps: Iterable[WordTimestampLike],
    min_duration: float = 4.0,
    max_duration: float = 20.0,
    silence_trim: float = 1.0,
    discard_under: float = 1.0,
) -> list[tuple[float, float, str]]:
    """Split transcript into clip segments using word timestamps and silence cuts."""
    words = list(word_timestamps or [])
    if not words or not full_text:
        return []

    text_tokens = full_text.split()
    sample = words[: min(30, len(words))]
    has_punct_in_words = any(re.search(r"[.!?,]", (w.text or "")) for w in sample)

    if has_punct_in_words or len(text_tokens) != len(words):
        def get_word_text(i: int) -> str:
            return (words[i].text or "").strip()
    else:
        def get_word_text(i: int) -> str:
            return text_tokens[i]

    sentence_ranges: list[tuple[int, int, str]] = []
    sent_start = 0
    for i, _w in enumerate(words):
        is_last = i == len(words) - 1
        word_text = get_word_text(i)
        ends_sentence = bool(re.search(r"[.!?][\"')]?$", word_text))
        if ends_sentence or is_last:
            sent_words = [get_word_text(j) for j in range(sent_start, i + 1)]
            sent_text = " ".join(sent_words)
            sentence_ranges.append((sent_start, i, sent_text))
            sent_start = i + 1

    if not sentence_ranges:
        return []

    silence_cuts = set()
    if silence_trim > 0:
        for i in range(len(words) - 1):
            gap = words[i + 1].start_time - words[i].end_time
            if gap > silence_trim:
                silence_cuts.add(i)

    segments: list[tuple[float, float, str]] = []
    group_texts: list[str] = []
    group_word_start = sentence_ranges[0][0]

    for si, (s_start, s_end, s_text) in enumerate(sentence_ranges):
        is_last_sentence = si == len(sentence_ranges) - 1
        cuts_in_sentence = sorted(c for c in silence_cuts if s_start <= c < s_end)

        if cuts_in_sentence:
            if group_texts:
                grp_start = words[group_word_start].start_time
                grp_end = words[s_start - 1].end_time if s_start > 0 else grp_start
                combined = " ".join(group_texts)
                if combined.strip():
                    segments.append((grp_start, grp_end, combined.strip()))
                group_texts = []

            chunk_boundaries = [s_start] + [c + 1 for c in cuts_in_sentence] + [s_end + 1]
            for cb_i in range(len(chunk_boundaries) - 1):
                chunk_w_start = chunk_boundaries[cb_i]
                chunk_w_end = chunk_boundaries[cb_i + 1] - 1
                if chunk_w_end < chunk_w_start:
                    continue
                chunk_word_texts = [get_word_text(w) for w in range(chunk_w_start, chunk_w_end + 1)]
                chunk_text = " ".join(chunk_word_texts)
                t_start = words[chunk_w_start].start_time
                t_end = words[chunk_w_end].end_time
                if chunk_text.strip():
                    segments.append((t_start, t_end, chunk_text.strip()))

            if not is_last_sentence:
                group_word_start = sentence_ranges[si + 1][0]
            continue

        if group_texts and s_start > 0:
            prev_word_idx = s_start - 1
            if prev_word_idx in silence_cuts:
                grp_start = words[group_word_start].start_time
                grp_end = words[prev_word_idx].end_time
                combined = " ".join(group_texts)
                if combined.strip():
                    segments.append((grp_start, grp_end, combined.strip()))
                group_texts = []
                group_word_start = s_start

        group_texts.append(s_text)
        group_end_idx = s_end
        grp_start_time = words[group_word_start].start_time
        grp_end_time = words[min(group_end_idx, len(words) - 1)].end_time
        grp_duration = grp_end_time - grp_start_time

        next_crosses_silence = False
        if not is_last_sentence:
            next_s_start = sentence_ranges[si + 1][0]
            for w in range(group_end_idx, next_s_start):
                if w in silence_cuts:
                    next_crosses_silence = True
                    break

        if grp_duration >= min_duration or is_last_sentence or next_crosses_silence:
            combined = " ".join(group_texts)
            if combined.strip():
                segments.append((grp_start_time, grp_end_time, combined.strip()))
            group_texts = []
            if not is_last_sentence:
                group_word_start = sentence_ranges[si + 1][0]

    if max_duration and max_duration > 0:
        final_segments: list[tuple[float, float, str]] = []
        for seg_start, seg_end, seg_text in segments:
            seg_duration = seg_end - seg_start
            if seg_duration <= max_duration:
                final_segments.append((seg_start, seg_end, seg_text))
                continue

            seg_word_indices = [
                i for i, w in enumerate(words)
                if w.start_time >= seg_start - 0.01 and w.end_time <= seg_end + 0.01
            ]
            if not seg_word_indices:
                final_segments.append((seg_start, seg_end, seg_text))
                continue

            comma_indices = [wi for wi in seg_word_indices if get_word_text(wi).endswith(",")]
            if not comma_indices:
                final_segments.append((seg_start, seg_end, seg_text))
                continue

            sub_start_idx = seg_word_indices[0]
            for ci, comma_wi in enumerate(comma_indices):
                is_last_comma = ci == len(comma_indices) - 1
                sub_start_time = words[sub_start_idx].start_time
                sub_end_time = words[comma_wi].end_time
                sub_dur = sub_end_time - sub_start_time
                should_cut = sub_dur >= min_duration and (sub_dur >= max_duration or is_last_comma)
                if should_cut:
                    sub_text = " ".join(get_word_text(j) for j in range(sub_start_idx, comma_wi + 1))
                    if sub_text.strip():
                        final_segments.append((sub_start_time, sub_end_time, sub_text.strip()))
                    sub_start_idx = comma_wi + 1

            last_word_idx = seg_word_indices[-1]
            if sub_start_idx <= last_word_idx:
                sub_text = " ".join(get_word_text(j) for j in range(sub_start_idx, last_word_idx + 1))
                sub_start_time = words[sub_start_idx].start_time
                sub_end_time = words[last_word_idx].end_time
                if sub_text.strip():
                    final_segments.append((sub_start_time, sub_end_time, sub_text.strip()))
        segments = final_segments

    pad = 0.15
    trimmed: list[tuple[float, float, str]] = []
    for seg_start, seg_end, seg_text in segments:
        seg_words = [
            w for w in words
            if w.start_time >= seg_start - 0.05 and w.end_time <= seg_end + 0.05
        ]
        if seg_words:
            first_start = seg_words[0].start_time
            last_end = seg_words[-1].end_time
            if first_start - seg_start > silence_trim:
                seg_start = max(0.0, first_start - pad)
            if seg_end - last_end > silence_trim:
                seg_end = last_end + pad
        trimmed.append((seg_start, seg_end, seg_text))
    segments = trimmed

    if discard_under and discard_under > 0:
        segments = [s for s in segments if (s[1] - s[0]) >= discard_under]

    return segments
