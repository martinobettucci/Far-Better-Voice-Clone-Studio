"""Source separation model management using the audio-separator CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

from .model_utils import get_configured_models_dir


CANONICAL_STEMS = (
    "Vocals",
    "Instrumental",
    "Drums",
    "Bass",
    "Other",
    "Guitar",
    "Piano",
)
DETAILED_STEMS = ("Drums", "Bass", "Other", "Guitar", "Piano")
STEM_FILTER_CHOICES = (
    "All",
    "2 Stem",
    "4 Stem",
    "6 Stem",
    "Detailed Stems",
)

_STEM_SYNONYMS = {
    "vocal": "Vocals",
    "vocals": "Vocals",
    "instrumental": "Instrumental",
    "drums": "Drums",
    "bass": "Bass",
    "other": "Other",
    "guitar": "Guitar",
    "piano": "Piano",
}

_MODEL_EXTENSIONS = {
    ".onnx",
    ".ckpt",
    ".pth",
    ".pt",
    ".yaml",
    ".yml",
    ".json",
    ".bin",
    ".safetensors",
}

# Recommended music-separation defaults based on the upstream audio-separator
# docs/examples and validated against the current CLI catalog.
DEFAULT_SOURCE_SEPARATION_MODEL_FILENAMES = (
    "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "UVR_MDXNET_KARA_2.onnx",
    "htdemucs_ft.yaml",
    "htdemucs_6s.yaml",
)

_BUILTIN_SOURCE_SEPARATION_MODEL_DATA = (
    {
        "model_filename": "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        "display_name": "Mel-Band Roformer Default (2 Stem)",
        "architecture": "MDXC",
        "stems": ["Vocals", "Instrumental"],
        "description": "Built-in default music separation model.",
    },
    {
        "model_filename": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "display_name": "BS-Roformer Large (2 Stem)",
        "architecture": "MDXC",
        "stems": ["Vocals", "Instrumental"],
        "description": "Built-in high-quality vocal/instrumental separation model.",
    },
    {
        "model_filename": "UVR_MDXNET_Main.onnx",
        "display_name": "UVR MDX Main (2 Stem)",
        "architecture": "MDX",
        "stems": ["Vocals", "Instrumental"],
        "description": "Built-in MDX general-purpose vocal/instrumental model.",
    },
    {
        "model_filename": "UVR_MDXNET_KARA_2.onnx",
        "display_name": "UVR MDX Karaoke 2 (2 Stem)",
        "architecture": "MDX",
        "stems": ["Vocals", "Instrumental"],
        "description": "Built-in karaoke-oriented instrumental/vocal separation model.",
    },
    {
        "model_filename": "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        "display_name": "Mel-Roformer Karaoke Aufr33 Viperx (2 Stem)",
        "architecture": "Roformer",
        "stems": ["Vocals", "Instrumental"],
        "description": "Built-in Roformer karaoke separation model.",
    },
    {
        "model_filename": "htdemucs_ft.yaml",
        "display_name": "HTDemucs FT (4 Stem)",
        "architecture": "Demucs",
        "stems": ["Vocals", "Drums", "Bass", "Other"],
        "description": "Built-in Demucs 4-stem model.",
    },
    {
        "model_filename": "htdemucs_6s.yaml",
        "display_name": "HTDemucs 6 Stem",
        "architecture": "Demucs",
        "stems": ["Vocals", "Drums", "Bass", "Other", "Guitar", "Piano"],
        "description": "Built-in Demucs 6-stem model with guitar/piano outputs.",
    },
)


@dataclass(frozen=True)
class SourceSeparationModel:
    """Normalized source-separation model metadata."""

    model_filename: str
    display_name: str
    architecture: str
    stems: tuple[str, ...]
    description: str = ""

    @property
    def stem_count(self) -> int:
        return len(self.stems)

    @property
    def supports_instrumental(self) -> bool:
        return "Instrumental" in self.stems

    @property
    def has_detailed_stems(self) -> bool:
        return any(stem in self.stems for stem in DETAILED_STEMS)

    @property
    def is_music_workflow_model(self) -> bool:
        if "Vocals" not in self.stems:
            return False
        return self.supports_instrumental or self.has_detailed_stems

    def to_cache_dict(self) -> dict:
        data = asdict(self)
        data["stems"] = list(self.stems)
        return data

    @classmethod
    def from_cache_dict(cls, data: dict) -> "SourceSeparationModel":
        return cls(
            model_filename=str(data.get("model_filename") or "").strip(),
            display_name=str(data.get("display_name") or data.get("model_filename") or "").strip(),
            architecture=str(data.get("architecture") or "Unknown").strip() or "Unknown",
            stems=tuple(_normalize_stem_names(data.get("stems"))),
            description=str(data.get("description") or "").strip(),
        )


@dataclass
class SourceSeparationResult:
    """Normalized source-separation outputs for Singing integration."""

    model: SourceSeparationModel
    input_path: str
    output_dir: str
    stem_paths: dict[str, str]
    raw_stem_paths: dict[str, str]
    suggested_names: dict[str, str]
    metadata_text: str
    metadata_by_stem: dict[str, str]
    backing_is_synthesized: bool
    status_message: str

    def to_dict(self) -> dict:
        return {
            "model": self.model.to_cache_dict(),
            "input_path": self.input_path,
            "output_dir": self.output_dir,
            "stem_paths": dict(self.stem_paths),
            "raw_stem_paths": dict(self.raw_stem_paths),
            "suggested_names": dict(self.suggested_names),
            "metadata_text": self.metadata_text,
            "metadata_by_stem": dict(self.metadata_by_stem),
            "backing_is_synthesized": bool(self.backing_is_synthesized),
            "status_message": self.status_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceSeparationResult":
        model_data = data.get("model") or {}
        return cls(
            model=SourceSeparationModel.from_cache_dict(model_data),
            input_path=str(data.get("input_path") or ""),
            output_dir=str(data.get("output_dir") or ""),
            stem_paths={str(k): str(v) for k, v in dict(data.get("stem_paths") or {}).items() if v},
            raw_stem_paths={str(k): str(v) for k, v in dict(data.get("raw_stem_paths") or {}).items() if v},
            suggested_names={str(k): str(v) for k, v in dict(data.get("suggested_names") or {}).items() if v},
            metadata_text=str(data.get("metadata_text") or ""),
            metadata_by_stem={str(k): str(v) for k, v in dict(data.get("metadata_by_stem") or {}).items() if v},
            backing_is_synthesized=bool(data.get("backing_is_synthesized", False)),
            status_message=str(data.get("status_message") or ""),
        )


def _get_source_separation_usage_label(model: SourceSeparationModel) -> str:
    haystack = f"{model.display_name} {model.model_filename}".lower()
    if model.stem_count >= 6:
        return "Detailed band stems incl. guitar/piano"
    if model.stem_count >= 4:
        return "Detailed band stems"
    if "karaoke" in haystack or "_kara" in haystack or "inst_" in haystack or "inst-" in haystack:
        return "Karaoke / cleaner instrumental"
    return "Vocals + backing"


def _get_source_separation_behavior_note(model: SourceSeparationModel) -> str:
    if model.supports_instrumental:
        return "Returns vocals plus instrumental directly."
    if model.has_detailed_stems:
        return "Returns detailed stems; the app rebuilds a backing track from non-vocal stems."
    return "Music separation model."


def _normalize_stem_names(value) -> list[str]:
    """Normalize raw stem metadata into canonical stem names."""
    if value is None:
        return []

    raw_items: list[str] = []
    if isinstance(value, dict):
        raw_items.extend(str(k) for k in value.keys())
    elif isinstance(value, (list, tuple, set)):
        raw_items.extend(str(item) for item in value)
    else:
        raw_items.extend(re.split(r"[,/|]+", str(value)))

    stems: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item or "").strip()
        if not text:
            continue
        text = re.sub(r"\([^)]*\)", "", text)
        text = text.replace("*", "")
        lowered = re.sub(r"[^a-z]+", "", text.lower())
        canonical = _STEM_SYNONYMS.get(lowered)
        if not canonical:
            continue
        if canonical not in seen:
            seen.add(canonical)
            stems.append(canonical)

    ordered = [stem for stem in CANONICAL_STEMS if stem in seen]
    return ordered


def _safe_json_load(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def synthesize_backing_track(stem_paths: dict[str, str], output_path: Path) -> str:
    """Build a backing track by summing all detailed non-vocal stems."""
    backing_stems = [Path(stem_paths[stem]) for stem in DETAILED_STEMS if stem in stem_paths]
    if not backing_stems:
        raise ValueError("No detailed stems were available to synthesize a backing track.")

    mixed = None
    sample_rate = None
    max_frames = 0
    max_channels = 1
    loaded_tracks: list[np.ndarray] = []

    for stem_path in backing_stems:
        audio, current_sr = sf.read(str(stem_path), always_2d=True)
        if sample_rate is None:
            sample_rate = current_sr
        elif current_sr != sample_rate:
            raise ValueError(
                f"Cannot synthesize backing with mismatched sample rates: {sample_rate} vs {current_sr}"
            )
        audio = np.asarray(audio, dtype=np.float32)
        loaded_tracks.append(audio)
        max_frames = max(max_frames, audio.shape[0])
        max_channels = max(max_channels, audio.shape[1])

    for audio in loaded_tracks:
        current = audio
        if current.shape[1] < max_channels:
            current = np.repeat(current, max_channels, axis=1)
        if current.shape[0] < max_frames:
            padding = np.zeros((max_frames - current.shape[0], current.shape[1]), dtype=np.float32)
            current = np.concatenate([current, padding], axis=0)
        mixed = current if mixed is None else mixed + current

    if mixed is None or sample_rate is None:
        raise ValueError("No audio data was available to synthesize a backing track.")

    mixed = np.clip(mixed, -1.0, 1.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mixed, sample_rate)
    return str(output_path)


class SourceSeparationManager:
    """Manage audio-separator catalog, downloads, and offline execution."""

    def __init__(self, user_config=None, models_dir=None):
        self.user_config = user_config or {}
        self.models_dir = Path(models_dir) if models_dir is not None else get_configured_models_dir()
        self._update_paths()

    def _update_paths(self):
        self.base_dir = self.models_dir / "source_separation"
        self.downloads_dir = self.base_dir / "models"
        self.catalog_cache_path = self.base_dir / "catalog_cache.json"
        self.download_manifest_path = self.base_dir / "download_manifest.json"

    def _ensure_dirs(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.downloads_dir.mkdir(parents=True, exist_ok=True)

    def _is_offline_mode(self) -> bool:
        return bool(self.user_config.get("offline_mode", False))

    def _audio_separator_executable(self) -> str:
        executable = shutil.which("audio-separator")
        if executable:
            return executable
        raise RuntimeError(
            "audio-separator is not installed or not in PATH.\n"
            "Install dependencies for Voice Clone Studio and retry."
        )

    def _run_cli(self, args: list[str], *, timeout_s: int = 900) -> str:
        command = [self._audio_separator_executable(), *args]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "audio-separator executable was not found. Install the package first."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"audio-separator timed out after {timeout_s} seconds.") from exc

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            detail = stderr or stdout or "No error details were returned."
            detail_line = detail.splitlines()[-1]
            raise RuntimeError(f"audio-separator failed: {detail_line}")

        return result.stdout or ""

    def _read_catalog_cache(self) -> list[SourceSeparationModel]:
        data = _safe_json_load(self.catalog_cache_path, {})
        items = data.get("models") if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        models: list[SourceSeparationModel] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            model = SourceSeparationModel.from_cache_dict(item)
            if model.model_filename and model.is_music_workflow_model:
                models.append(model)
        return models

    def _write_catalog_cache(self, models: list[SourceSeparationModel]):
        self._ensure_dirs()
        payload = {
            "updated_at": _timestamp_utc(),
            "models": [model.to_cache_dict() for model in models],
        }
        self.catalog_cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_download_manifest(self) -> dict:
        data = _safe_json_load(self.download_manifest_path, {})
        if not isinstance(data, dict):
            return {"models": {}}
        data.setdefault("models", {})
        if not isinstance(data["models"], dict):
            data["models"] = {}
        return data

    def _save_download_manifest(self, manifest: dict):
        self._ensure_dirs()
        manifest.setdefault("models", {})
        manifest["updated_at"] = _timestamp_utc()
        self.download_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _builtin_models(self) -> list[SourceSeparationModel]:
        return [
            SourceSeparationModel.from_cache_dict(item)
            for item in _BUILTIN_SOURCE_SEPARATION_MODEL_DATA
        ]

    def _merge_model_lists(self, *collections: Iterable[SourceSeparationModel]) -> list[SourceSeparationModel]:
        merged: list[SourceSeparationModel] = []
        seen: set[str] = set()
        for collection in collections:
            for model in collection:
                if not model.model_filename or model.model_filename in seen:
                    continue
                seen.add(model.model_filename)
                merged.append(model)
        return merged

    def _normalize_catalog_entry(self, entry: dict, fallback_name: str | None = None) -> SourceSeparationModel | None:
        filename = ""
        for key in ("filename", "model_filename", "file", "path", "download_filename", "name"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                filename = value.strip()
                break
        if not filename and fallback_name:
            filename = str(fallback_name).strip()
        if not filename:
            return None

        description = ""
        for key in ("description", "title", "friendly_name", "display_name", "name"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                description = value.strip()
                break

        architecture = ""
        for key in ("architecture", "arch", "type", "model_type"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                architecture = value.strip()
                break
        if not architecture:
            lowered = filename.lower()
            if "demucs" in lowered:
                architecture = "Demucs"
            elif "roformer" in lowered or "mdxc" in lowered:
                architecture = "MDXC"
            elif "mdx" in lowered:
                architecture = "MDX"
            elif "vr" in lowered:
                architecture = "VR"
            else:
                architecture = "Unknown"

        stems = _normalize_stem_names(
            entry.get("stems")
            or entry.get("output_stems")
            or entry.get("stem_names")
            or entry.get("outputs")
        )
        if not stems:
            for key in ("target_stem", "target", "stem"):
                candidate = _normalize_stem_names(entry.get(key))
                if candidate:
                    stems = candidate
                    break

        display_name = description or filename
        model = SourceSeparationModel(
            model_filename=filename,
            display_name=display_name,
            architecture=architecture,
            stems=tuple(stems),
            description=description,
        )
        return model if model.is_music_workflow_model else None

    def _normalize_catalog_payload(self, payload) -> list[SourceSeparationModel]:
        entries: list[tuple[str | None, dict]] = []
        if isinstance(payload, dict):
            candidate_items = payload.get("models") or payload.get("data")
            if isinstance(candidate_items, list):
                for item in candidate_items:
                    if isinstance(item, dict):
                        entries.append((None, item))
            else:
                for key, value in payload.items():
                    if isinstance(value, dict):
                        entries.append((str(key), value))
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    entries.append((None, item))

        normalized: list[SourceSeparationModel] = []
        seen: set[str] = set()
        for fallback_name, entry in entries:
            model = self._normalize_catalog_entry(entry, fallback_name=fallback_name)
            if not model or model.model_filename in seen:
                continue
            seen.add(model.model_filename)
            normalized.append(model)

        return sorted(
            normalized,
            key=lambda model: (
                model.architecture.lower(),
                -model.stem_count,
                model.display_name.lower(),
                model.model_filename.lower(),
            ),
        )

    def refresh_catalog(self, *, strict: bool = False) -> list[SourceSeparationModel]:
        cached_models = self._read_catalog_cache()
        builtin_models = self._builtin_models()
        try:
            payload = json.loads(self._run_cli(["-l", "--list_format=json"], timeout_s=180))
            models = self._normalize_catalog_payload(payload)
            merged_models = self._merge_model_lists(builtin_models, models, cached_models)
            if not merged_models:
                raise RuntimeError("audio-separator returned an empty music separation catalog.")
            self._write_catalog_cache(merged_models)
            return merged_models
        except Exception:
            merged_models = self._merge_model_lists(builtin_models, cached_models)
            if merged_models:
                return merged_models
            if cached_models and not strict:
                return cached_models
            raise

    def list_models(self, *, refresh: bool = False) -> list[SourceSeparationModel]:
        cached_models = self._read_catalog_cache()
        builtin_models = self._builtin_models()
        if not refresh:
            merged_models = self._merge_model_lists(builtin_models, cached_models)
            if merged_models:
                return merged_models
        return self.refresh_catalog(strict=refresh)

    def get_model(self, model_filename: str) -> SourceSeparationModel | None:
        target = str(model_filename or "").strip()
        if not target:
            return None
        for model in self.list_models():
            if model.model_filename == target:
                return model
        return None

    def get_default_model_filename(self, models: Iterable[SourceSeparationModel] | None = None) -> str | None:
        source = list(models) if models is not None else self.list_models()
        by_filename = {model.model_filename: model for model in source}
        for model_filename in DEFAULT_SOURCE_SEPARATION_MODEL_FILENAMES:
            if model_filename in by_filename:
                return model_filename
        return source[0].model_filename if source else None

    def get_default_models(self, models: Iterable[SourceSeparationModel] | None = None) -> list[SourceSeparationModel]:
        source = list(models) if models is not None else self.list_models()
        by_filename = {model.model_filename: model for model in source}
        selected: list[SourceSeparationModel] = []
        for model_filename in DEFAULT_SOURCE_SEPARATION_MODEL_FILENAMES:
            model = by_filename.get(model_filename)
            if model is not None:
                selected.append(model)
        return selected

    def get_architecture_choices(self, models: Iterable[SourceSeparationModel] | None = None) -> list[str]:
        source = list(models) if models is not None else self.list_models()
        choices = ["All"]
        for architecture in sorted({model.architecture for model in source if model.architecture}):
            choices.append(architecture)
        return choices

    def filter_models(
        self,
        *,
        search_text: str = "",
        architecture: str = "All",
        stem_filter: str = "All",
        models: Iterable[SourceSeparationModel] | None = None,
    ) -> list[SourceSeparationModel]:
        query = str(search_text or "").strip().lower()
        arch = str(architecture or "All").strip()
        stem_mode = str(stem_filter or "All").strip()
        source = list(models) if models is not None else self.list_models()

        filtered: list[SourceSeparationModel] = []
        for model in source:
            if arch not in {"", "All"} and model.architecture != arch:
                continue
            if query:
                haystack = " ".join(
                    [
                        model.display_name,
                        model.model_filename,
                        model.architecture,
                        " ".join(model.stems),
                        model.description,
                    ]
                ).lower()
                if query not in haystack:
                    continue
            if stem_mode == "2 Stem" and model.stem_count != 2:
                continue
            if stem_mode == "4 Stem" and model.stem_count != 4:
                continue
            if stem_mode == "6 Stem" and model.stem_count != 6:
                continue
            if stem_mode == "Detailed Stems" and not model.has_detailed_stems:
                continue
            filtered.append(model)
        return filtered

    def get_dropdown_choices(
        self,
        *,
        search_text: str = "",
        architecture: str = "All",
        stem_filter: str = "All",
        models: Iterable[SourceSeparationModel] | None = None,
    ) -> list[tuple[str, str]]:
        choices = []
        for model in self.filter_models(
            search_text=search_text,
            architecture=architecture,
            stem_filter=stem_filter,
            models=models,
        ):
            stem_text = ", ".join(model.stems)
            usage = _get_source_separation_usage_label(model)
            label = f"{model.display_name} | {usage} | {stem_text}"
            choices.append((label, model.model_filename))
        return choices

    def _model_folder_name(self, model_filename: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_filename or "")).strip("._-")
        if not safe:
            safe = "model"
        digest = hashlib.sha1(str(model_filename).encode("utf-8")).hexdigest()[:8]
        return f"{safe}__{digest}"

    def get_model_dir(self, model_filename: str) -> Path:
        return self.downloads_dir / self._model_folder_name(model_filename)

    def _scan_model_files(self, model_dir: Path) -> list[str]:
        if not model_dir.exists():
            return []
        files: list[str] = []
        for path in sorted(model_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() in _MODEL_EXTENSIONS:
                files.append(str(path.relative_to(model_dir)))
        return files

    def _record_local_model_files(self, model: SourceSeparationModel, files: list[str]):
        manifest = self._load_download_manifest()
        folder_name = self._model_folder_name(model.model_filename)
        manifest["models"][model.model_filename] = {
            "display_name": model.display_name,
            "architecture": model.architecture,
            "stems": list(model.stems),
            "folder_name": folder_name,
            "files": sorted(files),
            "updated_at": _timestamp_utc(),
        }
        self._save_download_manifest(manifest)

    def is_model_downloaded(self, model_filename: str) -> bool:
        model_dir = self.get_model_dir(model_filename)
        files = self._scan_model_files(model_dir)
        return bool(files)

    def ensure_model_available(self, model_filename: str, progress_callback=None) -> Path:
        model = self.get_model(model_filename)
        if model is None:
            raise RuntimeError(f"Unknown source-separation model: {model_filename}")

        model_dir = self.get_model_dir(model.model_filename)
        files = self._scan_model_files(model_dir)
        if files:
            self._record_local_model_files(model, files)
            return model_dir

        if self._is_offline_mode():
            raise RuntimeError(
                "Offline mode enabled and source-separation model is missing locally: "
                f"{model.model_filename}\n"
                "Use Settings > Model Downloading > Download them all, or disable Offline Mode."
            )

        self.download_model(model.model_filename, progress_callback=progress_callback)
        files = self._scan_model_files(model_dir)
        if not files:
            raise RuntimeError(
                f"Source-separation download finished but no files were found in {model_dir}"
            )
        return model_dir

    def download_model(self, model_filename: str, progress_callback=None) -> dict:
        model = self.get_model(model_filename)
        if model is None:
            raise RuntimeError(f"Unknown source-separation model: {model_filename}")

        self._ensure_dirs()
        model_dir = self.get_model_dir(model.model_filename)
        model_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback(0.1, desc=f"Downloading {model.display_name}...")

        self._run_cli(
            [
                "--model_filename",
                model.model_filename,
                "--model_file_dir",
                str(model_dir),
                "--download_model_only",
            ],
            timeout_s=3600,
        )

        files = self._scan_model_files(model_dir)
        if not files:
            raise RuntimeError(
                f"audio-separator reported success but no model files were found in {model_dir}"
            )

        self._record_local_model_files(model, files)

        if progress_callback:
            progress_callback(1.0, desc=f"{model.display_name} ready.")

        return {
            "model_filename": model.model_filename,
            "display_name": model.display_name,
            "model_dir": str(model_dir),
            "files": files,
        }

    def describe_model(self, model_filename: str) -> str:
        model = self.get_model(model_filename)
        if model is None:
            return "Select a source-separation model."
        downloaded = "Yes" if self.is_model_downloaded(model.model_filename) else "No"
        usage = _get_source_separation_usage_label(model)
        behavior = _get_source_separation_behavior_note(model)
        return "\n".join(
            [
                f"Model: {model.display_name}",
                f"Best For: {usage}",
                f"Filename: {model.model_filename}",
                f"Architecture: {model.architecture}",
                f"Outputs: {', '.join(model.stems)}",
                f"Behavior: {behavior}",
                f"Downloaded: {downloaded}",
            ]
        )

    def separate_audio(
        self,
        *,
        input_path: str | Path,
        model_filename: str,
        output_dir: str | Path,
        use_autocast: bool = False,
        normalization: float = 0.9,
        invert_spect: bool = False,
        progress_callback=None,
    ) -> SourceSeparationResult:
        model = self.get_model(model_filename)
        if model is None:
            raise RuntimeError(f"Unknown source-separation model: {model_filename}")

        input_audio = Path(str(input_path))
        if not input_audio.exists():
            raise FileNotFoundError(f"Input audio not found: {input_audio}")

        model_dir = self.ensure_model_available(model.model_filename, progress_callback=progress_callback)

        target_dir = Path(str(output_dir))
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_input_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", input_audio.stem).strip("_") or "audio"
        custom_output_names = {
            stem: f"{safe_input_stem}__{stem.lower().replace(' ', '_')}"
            for stem in CANONICAL_STEMS
        }

        if progress_callback:
            progress_callback(0.2, desc=f"Separating with {model.display_name}...")

        args = [
            str(input_audio),
            "--model_filename",
            model.model_filename,
            "--model_file_dir",
            str(model_dir),
            "--output_dir",
            str(target_dir),
            "--output_format",
            "wav",
            "--sample_rate",
            "44100",
            "--normalization",
            f"{float(normalization):.4f}",
            "--use_soundfile",
            "--custom_output_names",
            json.dumps(custom_output_names),
        ]
        if invert_spect:
            args.append("--invert_spect")
        if use_autocast:
            args.append("--use_autocast")

        self._run_cli(args, timeout_s=7200)

        raw_stem_paths: dict[str, str] = {}
        for stem, custom_name in custom_output_names.items():
            candidate = target_dir / f"{custom_name}.wav"
            if candidate.exists():
                raw_stem_paths[stem] = str(candidate)

        if "Vocals" not in raw_stem_paths:
            raise RuntimeError(
                f"Source separation completed but no vocals stem was produced for {input_audio.name}"
            )

        backing_is_synthesized = False
        backing_path = raw_stem_paths.get("Instrumental")
        if not backing_path:
            backing_is_synthesized = True
            backing_path = synthesize_backing_track(
                raw_stem_paths,
                target_dir / f"{safe_input_stem}__backing.wav",
            )

        stem_paths: dict[str, str] = {
            "Vocals": raw_stem_paths["Vocals"],
            "Backing": backing_path,
        }
        for stem in DETAILED_STEMS:
            if stem in raw_stem_paths:
                stem_paths[stem] = raw_stem_paths[stem]

        suggested_names = {
            "Vocals": f"singing_sep_vocals_{safe_input_stem}",
            "Backing": f"singing_sep_backing_{safe_input_stem}",
            "All": f"singing_sep_{safe_input_stem}",
        }
        for stem in DETAILED_STEMS:
            if stem in stem_paths:
                suggested_names[stem] = f"singing_sep_{stem.lower()}_{safe_input_stem}"

        metadata_text = "\n".join(
            [
                f"Generated: {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "Type: Singing Source Separation",
                f"Source Audio: {input_audio.name}",
                f"Model: {model.display_name}",
                f"Model Filename: {model.model_filename}",
                f"Architecture: {model.architecture}",
                f"Model Stems: {', '.join(model.stems)}",
                f"Normalization: {float(normalization):.4f}",
                f"Invert Spect: {bool(invert_spect)}",
                f"Use Autocast: {bool(use_autocast)}",
                f"Backing Synthesized: {bool(backing_is_synthesized)}",
                f"Available Outputs: {', '.join(stem_paths.keys())}",
            ]
        )
        metadata_by_stem = {
            stem: f"{metadata_text}\nOutput Stem: {stem}\n"
            for stem in stem_paths.keys()
        }

        detailed = [stem for stem in DETAILED_STEMS if stem in stem_paths]
        status_parts = [
            f"Separated {input_audio.name} using {model.display_name}.",
            "Outputs: Vocals + Backing",
        ]
        if detailed:
            status_parts.append(f"Detailed stems: {', '.join(detailed)}")
        if backing_is_synthesized:
            status_parts.append("Backing was synthesized from detailed stems.")
        status_message = " ".join(status_parts)

        if progress_callback:
            progress_callback(1.0, desc="Source separation complete.")

        return SourceSeparationResult(
            model=model,
            input_path=str(input_audio),
            output_dir=str(target_dir),
            stem_paths=stem_paths,
            raw_stem_paths=raw_stem_paths,
            suggested_names=suggested_names,
            metadata_text=metadata_text,
            metadata_by_stem=metadata_by_stem,
            backing_is_synthesized=backing_is_synthesized,
            status_message=status_message,
        )


_source_separation_manager: SourceSeparationManager | None = None


def get_source_separation_manager(user_config=None, models_dir=None) -> SourceSeparationManager:
    """Get or create the singleton source-separation manager."""
    global _source_separation_manager
    if _source_separation_manager is None:
        _source_separation_manager = SourceSeparationManager(user_config=user_config, models_dir=models_dir)
    else:
        if user_config is not None:
            _source_separation_manager.user_config = user_config
        if models_dir is not None:
            _source_separation_manager.models_dir = Path(models_dir)
            _source_separation_manager._update_paths()
    return _source_separation_manager
