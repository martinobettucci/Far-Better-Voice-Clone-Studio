"""
Audio utility functions for sample preparation.

Robust audio/video handling with proper error handling and fallbacks.
"""

import os
import re
import platform
import time
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime


def check_audio_format(audio_path):
    """Check if audio is 24kHz, 16-bit, mono.

    Returns:
        Tuple of (is_correct, info_or_none)
    """
    try:
        info = sf.info(audio_path)
        is_correct = (info.samplerate == 24000 and
                      info.channels == 1 and
                      info.subtype == 'PCM_16')
        return is_correct, info
    except Exception:
        return False, None


def is_video_file(filepath):
    """Check if file is a video based on extension."""
    if not filepath:
        return False
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg'}
    return Path(filepath).suffix.lower() in video_extensions


def is_audio_file(filepath):
    """Check if file is an audio file based on extension."""
    if not filepath:
        return False
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus'}
    return Path(filepath).suffix.lower() in audio_extensions


def extract_audio_from_video(video_path, temp_dir):
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to video file
        temp_dir: Directory to save extracted audio

    Returns:
        str: Path to extracted audio file, or None if failed
    """
    try:
        import subprocess
        import shutil

        video_input = Path(video_path)

        # Use a deterministic output name based on the video filename
        # so re-dragging the same video reuses the cached extraction
        stem = re.sub(r'[^\w\s.-]', '', video_input.stem).strip() or 'video'
        audio_output = Path(temp_dir) / f"{stem}.wav"

        # If we already have a cached extraction, reuse it
        if audio_output.exists():
            return str(audio_output), "Reused cached audio from video"

        # Copy input video to project temp if it's outside the project
        # (Gradio uploads go to system temp which can have permission issues for ffmpeg)
        project_root = Path(temp_dir).parent
        try:
            video_input.resolve().relative_to(project_root.resolve())
            local_video = video_input  # Already in project directory
        except ValueError:
            # Video is outside project (e.g. Gradio temp) — copy it locally
            local_name = f"input_video_{stem}{video_input.suffix}"
            local_video = Path(temp_dir) / local_name
            try:
                shutil.copy2(str(video_input), str(local_video))
            except Exception:
                local_video = video_input  # Fall back to original path

        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',  # Suppress banner/config output
            '-nostdin',
            '-i', str(local_video),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '24000',  # 24kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            str(audio_output)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Clean up the local video copy if we made one
        if local_video != video_input and local_video.exists():
            try:
                local_video.unlink()
            except Exception:
                pass

        if result.returncode == 0 and audio_output.exists():
            return str(audio_output), "Extracted audio from video"
        else:
            err_msg = result.stderr.strip()[:200] if result.stderr else "Unknown error"
            print(f"ffmpeg error: {err_msg}")
            return None, "⚠ Failed to extract audio from video"

    except FileNotFoundError:
        print("[ERROR] ffmpeg not found. Please install ffmpeg to extract audio from video.")
        return None, "⚠ ffmpeg not found"
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None, "⚠ Error extracting audio"


def get_audio_duration(audio_path):
    """Get duration of audio file in seconds."""
    try:
        audio_data, sr = sf.read(audio_path)
        return len(audio_data) / sr
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        return 0.0


def format_time(seconds):
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds < 0:
        return "0:00"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def normalize_audio(audio_file, temp_dir):
    """
    Normalize audio levels.

    Args:
        audio_file: Path to audio file
        temp_dir: Directory to save normalized audio

    Returns:
        tuple: Path to normalized audio file, or original path if failed and status message
    """
    if audio_file is None:
        return None, "⚠ No audio file provided"

    if not os.path.exists(audio_file):
        return None, "⚠ Audio file not found"

    try:
        data, sr = sf.read(audio_file)

        # Normalize to -1 to 1 range with conservative headroom
        max_val = np.max(np.abs(data))
        if max_val > 0:
            normalized = data / max_val * 0.85  # Leave 15% headroom to prevent clipping in TTS
        else:
            normalized = data

        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"normalized_{timestamp}.wav"
        temp_path = Path(temp_dir) / filename

        try:
            sf.write(str(temp_path), normalized, sr)
        except (PermissionError, OSError) as e:
            # Fallback to system temp
            try:
                print(f"[WARN] Could not write to {temp_path} ({e}). Falling back to system temp.")
                temp_path = Path(tempfile.gettempdir()) / filename
                sf.write(str(temp_path), normalized, sr)
            except Exception as fbe:
                print(f"Fallback save failed: {fbe}")
                raise RuntimeError(
                    f"Failed to save normalized audio to both {temp_dir} and system temp. "
                    f"Primary error: {e}; fallback error: {fbe}"
                ) from fbe

        # Force file flush on Windows to prevent connection reset errors
        if platform.system() == "Windows":
            time.sleep(0.1)  # Small delay to ensure file is fully written

        return str(temp_path), "Normalized audio"

    except Exception as e:
        print(f"Error normalizing audio: {e}")
        return audio_file, "⚠ Error normalizing audio"


def remove_silences(audio_file, temp_dir, top_db=40.0, keep_silence_ms=60, min_segment_ms=80):
    """
    Remove long silent spans from audio while keeping short pauses between retained regions.

    Args:
        audio_file: Path to audio file
        temp_dir: Directory to save processed audio
        top_db: Silence threshold relative to peak level
        keep_silence_ms: Short gap to retain between stitched segments
        min_segment_ms: Ignore tiny non-silent detections shorter than this

    Returns:
        tuple: Path to processed audio file, or original path if nothing changed and status message
    """
    if audio_file is None:
        return None, "⚠ No audio file provided"

    if not os.path.exists(audio_file):
        return None, "⚠ Audio file not found"

    try:
        import librosa

        data, sr = sf.read(audio_file)
        sample_count = data.shape[0] if getattr(data, "ndim", 0) else 0
        if sample_count == 0:
            return audio_file, "No audio samples found"

        guide = data.astype(np.float32, copy=False)
        if guide.ndim > 1:
            guide = np.mean(guide, axis=1)

        peak = float(np.max(np.abs(guide))) if guide.size else 0.0
        if peak <= 1e-6:
            return audio_file, "No non-silent audio detected"

        intervals = librosa.effects.split(
            guide,
            top_db=float(top_db),
            ref=peak,
            frame_length=2048,
            hop_length=512,
        )
        if len(intervals) == 0:
            return audio_file, "No non-silent audio detected"

        keep_padding = max(int(sr * (float(keep_silence_ms) / 1000.0)), 0)
        min_segment_samples = max(int(sr * (float(min_segment_ms) / 1000.0)), 1)
        merged_intervals: list[list[int]] = []

        for raw_start, raw_end in intervals.tolist():
            start = max(int(raw_start) - keep_padding, 0)
            end = min(int(raw_end) + keep_padding, sample_count)
            if end - start < min_segment_samples:
                continue
            if merged_intervals and start <= merged_intervals[-1][1]:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], end)
            else:
                merged_intervals.append([start, end])

        if not merged_intervals:
            return audio_file, "No non-silent audio detected"
        if len(merged_intervals) == 1 and merged_intervals[0] == [0, sample_count]:
            return audio_file, "No silences removed"

        gap_samples = max(int(sr * (float(keep_silence_ms) / 1000.0)), 0)
        if data.ndim > 1:
            gap = np.zeros((gap_samples, data.shape[1]), dtype=data.dtype)
        else:
            gap = np.zeros(gap_samples, dtype=data.dtype)

        stitched_parts = []
        for index, (start, end) in enumerate(merged_intervals):
            if index > 0 and gap_samples > 0:
                stitched_parts.append(gap)
            stitched_parts.append(data[start:end])

        trimmed = np.concatenate(stitched_parts, axis=0)
        if trimmed.shape[0] >= sample_count:
            return audio_file, "No silences removed"

        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"nosilence_{timestamp}.wav"
        temp_path = Path(temp_dir) / filename

        try:
            sf.write(str(temp_path), trimmed, sr)
        except (PermissionError, OSError) as e:
            print(f"[WARN] Could not write to {temp_path} ({e}). Falling back to system temp.")
            temp_path = Path(tempfile.gettempdir()) / filename
            sf.write(str(temp_path), trimmed, sr)

        if platform.system() == "Windows":
            time.sleep(0.1)

        return str(temp_path), "Removed silences"

    except Exception as e:
        print(f"Error removing silences: {e}")
        return audio_file, "⚠ Error removing silences"


def convert_to_mono(audio_file, temp_dir):
    """
    Convert stereo audio to mono.

    Args:
        audio_file: Path to audio file
        temp_dir: Directory to save mono audio

    Returns:
        tuple: Path to mono audio file, or original path if already mono or failed and status message
    """
    if audio_file is None:
        return None, "⚠ No audio file provided"

    if not os.path.exists(audio_file):
        return None, "⚠ Audio file not found"

    try:
        data, sr = sf.read(audio_file)

        # Check if stereo, if mono return original
        if len(data.shape) > 1 and data.shape[1] > 1:
            mono = np.mean(data, axis=1)

            timestamp = datetime.now().strftime('%H%M%S')
            filename = f"mono_{timestamp}.wav"
            temp_path = Path(temp_dir) / filename

            try:
                sf.write(str(temp_path), mono, sr)
            except (PermissionError, OSError) as e:
                # Fallback to system temp
                print(f"[WARN] Could not write to {temp_path} ({e}). Falling back to system temp.")
                temp_path = Path(tempfile.gettempdir()) / filename
                sf.write(str(temp_path), mono, sr)

            # Force file flush on Windows
            if platform.system() == "Windows":
                time.sleep(0.1)

            return str(temp_path), "Converted to mono"
        else:
            return audio_file, "Already mono"

    except Exception as e:
        print(f"Error converting to mono: {e}")
        return audio_file, "⚠ Error converting to mono"


def clean_audio(audio_file, temp_dir, get_deepfilter_model_func, progress_callback=None):
    """
    Clean audio using DeepFilterNet.

    Args:
        audio_file: Path to audio file
        temp_dir: Directory to save cleaned audio
        get_deepfilter_model_func: Function that returns (model, state, params) tuple
        progress_callback: Optional progress callback function

    Returns:
        tuple: Path to cleaned audio file, or original path if failed and status message
    """
    if audio_file is None:
        return None, "⚠ No audio file provided"

    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at path: {audio_file}")
        return None, "⚠ Audio file not found"

    try:
        if progress_callback:
            progress_callback(0.1, desc="Loading Audio Cleaner...")

        df_model, df_state, df_params = get_deepfilter_model_func()

        # Get sample rate from params or use default
        target_sr = df_params.sr if df_params is not None and hasattr(df_params, 'sr') else 48000

        if progress_callback:
            progress_callback(0.3, desc="Processing audio...")

        # Import DeepFilterNet functions
        from df.enhance import enhance
        from df.io import load_audio as df_load_audio, save_audio
        import torch

        # Load audio using DeepFilterNet's loader
        audio, _ = df_load_audio(audio_file, sr=target_sr)

        # Ensure tensor is contiguous (video-extracted audio can be non-contiguous)
        if hasattr(audio, 'is_contiguous') and not audio.is_contiguous():
            audio = audio.contiguous()

        # Run enhancement with cuDNN disabled to avoid CUDNN_STATUS_NOT_SUPPORTED
        # errors from non-contiguous intermediate tensors inside DeepFilterNet
        cudnn_was_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            enhanced_audio = enhance(df_model, df_state=df_state, audio=audio)
        finally:
            torch.backends.cudnn.enabled = cudnn_was_enabled

        # Save output
        timestamp = datetime.now().strftime("%H%M%S")
        output_filename = f"cleaned_{timestamp}.wav"
        output_path = Path(temp_dir) / output_filename

        # Robust save with fallback for permission/system errors
        try:
            save_audio(str(output_path), enhanced_audio, target_sr)
        except (PermissionError, OSError, RuntimeError) as e:
            msg = str(e)
            if "Permission denied" in msg or "System error" in msg:
                print(f"[WARN] Could not write to {output_path} ({msg}). Falling back to system temp.")
                output_path = Path(tempfile.gettempdir()) / output_filename
                save_audio(str(output_path), enhanced_audio, target_sr)
            else:
                raise e

        if progress_callback:
            progress_callback(1.0, desc="Done!")

        return str(output_path), "Cleaned with DeepFilterNet"

    except Exception as e:
        print(f"Error cleaning audio: {e}")
        return audio_file, "⚠ Error cleaning audio"


def save_audio_as_sample(audio_file, transcription, sample_name, samples_dir):
    """
    Save audio and transcription as a new sample.

    Args:
        audio_file: Path to audio file
        transcription: Transcription text
        sample_name: Name for the sample
        samples_dir: Directory to save sample

    Returns:
        tuple: (status_message, success_bool)
    """
    import re
    import json

    if not audio_file:
        return "[ERROR] No audio file to save.", False

    if not transcription or transcription.startswith("[ERROR]"):
        return "[ERROR] Please provide a transcription first.", False

    if not sample_name or not sample_name.strip():
        return "[ERROR] Please enter a sample name.", False

    # Clean sample name
    clean_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in sample_name).strip()
    clean_name = clean_name.replace(" ", "_")

    if not clean_name:
        return "[ERROR] Invalid sample name.", False

    try:
        # Read audio file
        audio_data, sr = sf.read(audio_file)

        # Clean transcription: remove ALL text in square brackets [...]
        # This removes [Speaker X], [human sounds], [lyrics], etc.
        cleaned_transcription = re.sub(r'\[.*?\]\s*', '', transcription)
        cleaned_transcription = cleaned_transcription.strip()

        # Save wav file
        wav_path = Path(samples_dir) / f"{clean_name}.wav"
        sf.write(str(wav_path), audio_data, sr)

        # Save .json metadata
        meta = {
            "Type": "Sample",
            "Text": cleaned_transcription if cleaned_transcription else ""
        }
        json_path = Path(samples_dir) / f"{clean_name}.json"
        json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return f"Sample saved as '{clean_name}'", True

    except Exception as e:
        return f"[ERROR] Error saving sample: {str(e)}", False
