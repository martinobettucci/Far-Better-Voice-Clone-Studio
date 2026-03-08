"""Speaker separation model management using SpeechBrain SepFormer models."""

from __future__ import annotations

from dataclasses import dataclass
import gc
from pathlib import Path

import numpy as np

from .model_utils import empty_device_cache, run_pre_load_hooks


@dataclass(frozen=True)
class SpeakerSeparationModel:
    """Normalized metadata for a SpeechBrain speaker-separation model."""

    expected_speakers: int
    display_name: str
    repo_id: str
    sample_rate: int

    @property
    def local_folder_name(self) -> str:
        return self.repo_id.split("/")[-1]

    @property
    def required_files(self) -> tuple[str, ...]:
        return ("hyperparams.yaml", "encoder.ckpt", "decoder.ckpt", "masknet.ckpt")


@dataclass(frozen=True)
class SeparatedTrack:
    """One separated speaker track returned by the manager."""

    speaker_index: int
    audio_data: np.ndarray
    sample_rate: int


_SPEAKER_SEPARATION_MODELS = (
    SpeakerSeparationModel(
        expected_speakers=2,
        display_name="SpeechBrain SepFormer - 2 Speakers (16kHz)",
        repo_id="speechbrain/sepformer-whamr16k",
        sample_rate=16_000,
    ),
    SpeakerSeparationModel(
        expected_speakers=3,
        display_name="SpeechBrain SepFormer - 3 Speakers (8kHz)",
        repo_id="speechbrain/sepformer-libri3mix",
        sample_rate=8_000,
    ),
)


def _speechbrain_runtime_device() -> str:
    """SpeechBrain separation is only enabled on CUDA or CPU for now."""
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _normalize_estimated_sources(raw_sources, expected_speakers: int) -> list[np.ndarray]:
    """Normalize SpeechBrain outputs to one mono float32 array per speaker."""
    if hasattr(raw_sources, "detach"):
        data = raw_sources.detach().cpu().float().numpy()
    else:
        data = np.asarray(raw_sources, dtype=np.float32)

    data = np.squeeze(data)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim != 2:
        raise RuntimeError(
            f"Unexpected speaker-separation output shape: {tuple(data.shape)}"
        )

    if data.shape[0] == expected_speakers and data.shape[1] >= expected_speakers:
        speakers_first = data
    elif data.shape[1] == expected_speakers and data.shape[0] >= expected_speakers:
        speakers_first = data.T
    else:
        raise RuntimeError(
            "Speaker-separation output did not match the requested speaker count "
            f"({expected_speakers}). Received shape {tuple(data.shape)}."
        )

    tracks: list[np.ndarray] = []
    for idx in range(expected_speakers):
        track = np.asarray(speakers_first[idx], dtype=np.float32).reshape(-1)
        track = np.nan_to_num(track, nan=0.0, posinf=0.0, neginf=0.0)
        track = np.clip(track, -1.0, 1.0)
        tracks.append(np.ascontiguousarray(track))
    return tracks


class SpeakerSeparationManager:
    """Manage SpeechBrain SepFormer speaker-separation models."""

    def __init__(self, user_config=None, models_dir=None):
        self.user_config = user_config or {}
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self._separator = None
        self._last_model_key: int | None = None

    def _configured_models_dir(self) -> Path:
        from .model_utils import get_configured_models_dir

        if self.models_dir is not None:
            return Path(self.models_dir)
        return get_configured_models_dir()

    def list_models(self) -> list[SpeakerSeparationModel]:
        return list(_SPEAKER_SEPARATION_MODELS)

    def get_model(self, expected_speakers: int) -> SpeakerSeparationModel:
        try:
            expected = int(expected_speakers)
        except Exception as exc:
            raise ValueError(f"Invalid speaker count: {expected_speakers}") from exc

        for model in _SPEAKER_SEPARATION_MODELS:
            if model.expected_speakers == expected:
                return model
        raise ValueError(f"Unsupported speaker count: {expected_speakers}")

    def get_model_dir(self, expected_speakers: int) -> Path:
        model = self.get_model(expected_speakers)
        return self._configured_models_dir() / model.local_folder_name

    def _has_required_files(self, model: SpeakerSeparationModel, model_dir: Path) -> bool:
        return all((model_dir / filename).exists() for filename in model.required_files)

    def ensure_model_available(self, expected_speakers: int) -> Path:
        model = self.get_model(expected_speakers)
        model_dir = self.get_model_dir(model.expected_speakers)
        if self._has_required_files(model, model_dir):
            return model_dir

        offline_mode = bool(self.user_config.get("offline_mode", False))
        if offline_mode:
            missing = ", ".join(model.required_files)
            raise RuntimeError(
                "❌ Offline mode enabled and speaker-separation model is missing locally: "
                f"{model.display_name}\n"
                f"Expected files in {model_dir}: {missing}\n"
                "Download it in Settings -> Model Downloading or disable Offline Mode."
            )

        return Path(self.download_model(model.expected_speakers)["model_dir"])

    def download_model(self, expected_speakers: int) -> dict:
        model = self.get_model(expected_speakers)
        model_dir = self.get_model_dir(model.expected_speakers)
        model_dir.mkdir(parents=True, exist_ok=True)

        if self._has_required_files(model, model_dir):
            return {
                "display_name": model.display_name,
                "repo_id": model.repo_id,
                "model_dir": str(model_dir),
                "message": f"Model already exists at {model_dir}",
            }

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required for SpeechBrain model downloads."
            ) from exc

        try:
            snapshot_download(
                repo_id=model.repo_id,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except TypeError:
            # Newer huggingface_hub versions removed local_dir_use_symlinks.
            snapshot_download(
                repo_id=model.repo_id,
                local_dir=str(model_dir),
                resume_download=True,
            )

        if not self._has_required_files(model, model_dir):
            missing = ", ".join(
                filename for filename in model.required_files if not (model_dir / filename).exists()
            )
            raise RuntimeError(
                "Model files not found after download. "
                f"Missing required SpeechBrain files: {missing}"
            )

        return {
            "display_name": model.display_name,
            "repo_id": model.repo_id,
            "model_dir": str(model_dir),
            "message": f"Successfully downloaded to {model_dir}",
        }

    def unload_all(self):
        if self._separator is not None:
            del self._separator
            self._separator = None
            self._last_model_key = None
            gc.collect()
            empty_device_cache()

    def get_separator(self, expected_speakers: int):
        model = self.get_model(expected_speakers)
        if self._separator is not None and self._last_model_key == model.expected_speakers:
            return self._separator

        if self._separator is not None and self._last_model_key != model.expected_speakers:
            self.unload_all()

        try:
            from speechbrain.inference.separation import SepformerSeparation
        except ImportError as exc:
            raise ImportError(
                "SpeechBrain is not installed. Install with: pip install speechbrain"
            ) from exc

        run_pre_load_hooks()
        savedir = self.ensure_model_available(model.expected_speakers)
        self._separator = SepformerSeparation.from_hparams(
            source=str(savedir),
            savedir=str(savedir),
            run_opts={"device": _speechbrain_runtime_device()},
        )
        self._last_model_key = model.expected_speakers
        return self._separator

    def separate_file(self, audio_path: str, expected_speakers: int) -> list[SeparatedTrack]:
        audio_path = str(audio_path or "").strip()
        if not audio_path:
            raise ValueError("No audio file was provided for speaker separation.")
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self.get_model(expected_speakers)
        separator = self.get_separator(model.expected_speakers)
        estimated_sources = separator.separate_file(audio_path)
        tracks = _normalize_estimated_sources(estimated_sources, model.expected_speakers)
        return [
            SeparatedTrack(
                speaker_index=index,
                audio_data=track,
                sample_rate=model.sample_rate,
            )
            for index, track in enumerate(tracks, start=1)
        ]


_speaker_separation_manager: SpeakerSeparationManager | None = None


def get_speaker_separation_manager(user_config=None, models_dir=None) -> SpeakerSeparationManager:
    """Get or create the shared speaker-separation manager."""
    global _speaker_separation_manager
    if _speaker_separation_manager is None:
        _speaker_separation_manager = SpeakerSeparationManager(
            user_config=user_config,
            models_dir=models_dir,
        )
    else:
        if user_config is not None:
            _speaker_separation_manager.user_config = user_config
        if models_dir is not None:
            _speaker_separation_manager.models_dir = Path(models_dir)
    return _speaker_separation_manager
