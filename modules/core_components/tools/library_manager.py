"""Unified tenant-scoped Library Manager for samples, datasets, and processing."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
from gradio_filelister import FileLister

from modules.core_components.ai_models.asr_manager import get_asr_manager
from modules.core_components.ai_models import get_speaker_separation_manager
from modules.core_components.constants import resolve_preferred_asr_engine_and_model
from modules.core_components.constants import coerce_choice_value
from modules.core_components.library_processing import (
    ProcessingPipelineConfig,
    ProcessingSourceContext,
    build_speaker_separation_import_plan,
    build_speaker_separation_sample_metadata,
    WordTimestampLike,
    build_routed_processing_context,
    clean_transcription_for_engine,
    estimate_pcm16_wav_bytes,
    language_to_code,
    parse_asr_model,
    resolve_processing_save_destination,
    split_into_segments,
)
from modules.core_components.tenant_storage import (
    AUDIO_EXTENSIONS,
    collision_safe_path,
    is_allowed_media,
    sanitize_filename,
)
from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.runtime import MemoryAdmissionError
from modules.core_components.ui_components.audio_routing import audio_route_parse_payload


class LibraryManagerTool(Tool):
    """Tenant-aware library and processing manager."""

    config = ToolConfig(
        name="Library Manager",
        module_name="tool_library_manager",
        description="Manage tenant samples/datasets and process audio end-to-end",
        enabled=True,
        category="preparation",
    )

    @classmethod
    def create_tool(cls, shared_state):
        components = {}

        _user_config = shared_state.get("_user_config", {})
        asr_engines = shared_state.get("ASR_ENGINES", {})
        languages = shared_state.get("LANGUAGES", ["Auto"])
        qwen_available = shared_state.get("QWEN3_ASR_AVAILABLE", False)
        whisper_available = shared_state.get("WHISPER_AVAILABLE", False)
        deepfilter_available = shared_state.get("DEEPFILTER_AVAILABLE", False)
        get_sample_choices = shared_state["get_sample_choices"]
        get_dataset_folders = shared_state["get_dataset_folders"]

        asr_settings = _user_config.get("enabled_asr_engines", {})
        visible_asr_options: list[str] = []
        for engine_key, engine_info in asr_engines.items():
            if not asr_settings.get(engine_key, engine_info.get("default_enabled", True)):
                continue
            if engine_key == "Qwen3 ASR" and not qwen_available:
                continue
            if engine_key == "Whisper" and not whisper_available:
                continue
            visible_asr_options.extend(engine_info["choices"])
        if not visible_asr_options:
            visible_asr_options = ["VibeVoice ASR - Default"]

        preferred_engine, preferred_model = resolve_preferred_asr_engine_and_model(_user_config)
        default_asr = coerce_choice_value(_user_config.get("transcribe_model", preferred_model))
        default_asr = str(default_asr) if default_asr is not None else preferred_model
        if preferred_engine in asr_engines and default_asr not in asr_engines[preferred_engine].get("choices", []):
            choices = asr_engines[preferred_engine].get("choices", [])
            default_asr = choices[-1] if choices else default_asr
        if default_asr not in visible_asr_options:
            default_asr = visible_asr_options[0]
        show_lang = any(k in default_asr for k in ("Qwen3 ASR", "Whisper"))

        default_expected_speakers = str(_user_config.get("speaker_separation_expected_speakers", "2"))
        speaker_note = (
            "3-speaker separation uses the upstream SpeechBrain 8 kHz model. "
            "Imported samples are lower fidelity than the 2-speaker path."
            if default_expected_speakers == "3"
            else "Uses the 2-speaker SpeechBrain model at 16 kHz."
        )

        with gr.TabItem("Library Manager", id="tab_library_manager") as library_tab:
            components["library_tab"] = library_tab

            with gr.Row():
                components["library_tenant_md"] = gr.Markdown("Tenant: -")
                components["library_usage_md"] = gr.Markdown("Tenant storage: -")
            components["library_status"] = gr.Textbox(label="Status", interactive=False, max_lines=3)

            with gr.Tabs(selected="library_samples") as library_sections:
                components["library_sections"] = library_sections

                with gr.TabItem("Samples", id="library_samples"):
                    gr.Markdown("Load and maintain canonical sample WAV files. Transcript edits autosave.")
                    with gr.Row():
                        with gr.Column(scale=1):
                            components["samples_lister"] = FileLister(
                                value=[],
                                height=260,
                                show_footer=False,
                                interactive=True,
                            )
                            components["samples_upload"] = gr.File(
                                label="Upload Sample Audio/Video",
                                file_count="multiple",
                                type="filepath",
                                file_types=["audio", "video"],
                            )
                            with gr.Row():
                                components["upload_samples_btn"] = gr.Button("Upload", variant="primary")
                                components["refresh_samples_btn"] = gr.Button("Refresh", size="sm")
                            with gr.Row():
                                components["open_sample_processing_btn"] = gr.Button("Process", size="sm")
                                components["clear_sample_cache_btn"] = gr.Button("Clear Cache", size="sm")
                                components["delete_samples_btn"] = gr.Button("Delete Selected", size="sm", variant="stop")
                            components["sample_action_status"] = gr.Markdown(" ")

                        with gr.Column(scale=1):
                            components["sample_preview"] = gr.Audio(label="Preview", type="filepath")
                            components["sample_transcript"] = gr.Textbox(label="Transcript", lines=6)
                            components["sample_transcript_status"] = gr.Markdown("Saved")
                            components["sample_info"] = gr.Markdown(" ")
                            components["legacy_sample_warning"] = gr.Markdown(visible=False)
                            components["convert_legacy_samples_btn"] = gr.Button(
                                "Convert Legacy Sample Audio to WAV",
                                size="sm",
                                visible=False,
                            )

                with gr.TabItem("Datasets", id="library_datasets"):
                    gr.Markdown("Manage dataset folders/files. Transcript edits autosave. Auto-split runs from selected dataset files.")
                    with gr.Row():
                        with gr.Column(scale=1):
                            components["dataset_folder_dropdown"] = gr.Dropdown(
                                choices=["(Select Dataset)"] + get_dataset_folders(),
                                value="(Select Dataset)",
                                label="Dataset Folder",
                                interactive=True,
                            )
                            with gr.Row():
                                components["new_dataset_name"] = gr.Textbox(
                                    label="New Dataset Folder",
                                    placeholder="e.g. narrator_a",
                                )
                                components["create_dataset_btn"] = gr.Button("Create", size="sm")
                                components["delete_dataset_btn"] = gr.Button("Delete Folder", size="sm", variant="stop")

                            components["dataset_lister"] = FileLister(
                                value=[],
                                height=260,
                                show_footer=False,
                                interactive=True,
                            )
                            components["dataset_upload"] = gr.File(
                                label="Upload Dataset Audio/Video",
                                file_count="multiple",
                                type="filepath",
                                file_types=["audio", "video"],
                            )
                            with gr.Row():
                                components["upload_dataset_btn"] = gr.Button("Upload", variant="primary")
                                components["refresh_dataset_btn"] = gr.Button("Refresh", size="sm")
                            with gr.Row():
                                components["open_dataset_processing_btn"] = gr.Button("Process", size="sm")
                                components["delete_dataset_files_btn"] = gr.Button("Delete Selected", size="sm", variant="stop")
                            components["dataset_action_status"] = gr.Markdown(" ")

                        with gr.Column(scale=1):
                            components["dataset_preview"] = gr.Audio(label="Preview", type="filepath")
                            components["dataset_transcript"] = gr.Textbox(label="Transcript", lines=6)
                            components["dataset_transcript_status"] = gr.Markdown("Saved")
                            with gr.Row():
                                components["batch_replace_existing"] = gr.Checkbox(
                                    label="Replace existing transcripts",
                                    value=False,
                                )
                            components["dataset_transcribe_model"] = gr.Dropdown(
                                choices=visible_asr_options,
                                value=default_asr,
                                label="Transcription Engine",
                            )
                            components["dataset_language"] = gr.Dropdown(
                                choices=["Auto-detect"] + languages[1:],
                                value=_user_config.get("whisper_language", "Auto-detect"),
                                label="Language",
                                visible=show_lang,
                            )
                            components["batch_transcribe_btn"] = gr.Button("Batch Transcribe", variant="secondary")

                            with gr.Accordion("Auto-Split Selected File(s)", open=False):
                                components["dataset_split_target_folder"] = gr.Dropdown(
                                    choices=["(Select Dataset)"] + get_dataset_folders(),
                                    value="(Select Dataset)",
                                    label="Target Dataset Folder",
                                )
                                components["dataset_split_prefix"] = gr.Textbox(
                                    label="Clip Prefix (optional)",
                                    placeholder="Leave empty to use source filename stem",
                                )
                                components["dataset_split_transcribe_model"] = gr.Dropdown(
                                    choices=visible_asr_options,
                                    value=default_asr,
                                    label="Transcription Engine",
                                )
                                components["dataset_split_language"] = gr.Dropdown(
                                    choices=["Auto-detect"] + languages[1:],
                                    value=_user_config.get("whisper_language", "Auto-detect"),
                                    label="Language",
                                    visible=show_lang,
                                )
                                with gr.Row():
                                    components["split_min"] = gr.Slider(
                                        minimum=1,
                                        maximum=15,
                                        value=_user_config.get("split_min", 5),
                                        step=0.5,
                                        label="Min clip (s)",
                                    )
                                    components["split_max"] = gr.Slider(
                                        minimum=10,
                                        maximum=60,
                                        value=_user_config.get("split_max", 20),
                                        step=1,
                                        label="Max clip (s)",
                                    )
                                with gr.Row():
                                    components["silence_trim"] = gr.Slider(
                                        minimum=0,
                                        maximum=10,
                                        value=_user_config.get("silence_trim", 1),
                                        step=0.5,
                                        label="Silence trim threshold (s)",
                                    )
                                    components["discard_under"] = gr.Slider(
                                        minimum=0,
                                        maximum=5,
                                        value=_user_config.get("discard_under", 1),
                                        step=0.5,
                                        label="Discard clips under (s)",
                                    )
                                components["dataset_auto_split_btn"] = gr.Button("Auto-Split Selected File(s)", variant="secondary")

                with gr.TabItem("Processing Studio", id="library_processing"):
                    gr.Markdown("Context-driven pipeline. Source is loaded from Samples or Datasets. Pipeline always recomputes from the original file.")
                    with gr.Row():
                        with gr.Column(scale=1):
                            components["proc_source_type"] = gr.Textbox(label="Source Type", interactive=False)
                            components["proc_source_identifier"] = gr.Textbox(label="Source", interactive=False)
                            components["proc_original_file"] = gr.Textbox(label="Original File", interactive=False)
                            components["proc_status"] = gr.Textbox(label="Process Status", interactive=False, lines=4)

                        with gr.Column(scale=2):
                            components["proc_audio_editor"] = gr.Audio(
                                label="Audio Editor",
                                type="filepath",
                                interactive=True,
                                visible=False,
                                elem_id="library-proc-audio-editor",
                            )
                            with gr.Row():
                                components["proc_enable_denoise"] = gr.Checkbox(
                                    label="Enable Denoise",
                                    value=False,
                                    visible=deepfilter_available,
                                )
                                components["proc_enable_normalize"] = gr.Checkbox(
                                    label="Enable Normalize",
                                    value=False,
                                )
                                components["proc_enable_mono"] = gr.Checkbox(
                                    label="Enable Mono",
                                    value=False,
                                )
                            with gr.Row():
                                components["proc_apply_pipeline_btn"] = gr.Button("Apply Pipeline", variant="secondary", size="sm")
                                components["proc_reset_btn"] = gr.Button("Reset to Original", size="sm")
                                components["proc_transcribe_btn"] = gr.Button("Transcribe", variant="primary", size="sm")

                            components["proc_transcribe_model"] = gr.Dropdown(
                                choices=visible_asr_options,
                                value=default_asr,
                                label="Transcription Engine",
                            )
                            components["proc_language"] = gr.Dropdown(
                                choices=["Auto-detect"] + languages[1:],
                                value=_user_config.get("whisper_language", "Auto-detect"),
                                label="Language",
                                visible=show_lang,
                            )
                            components["proc_transcript"] = gr.Textbox(
                                label="Transcript (draft, saved with output)",
                                lines=6,
                                placeholder="Transcription appears here, or edit manually.",
                            )

                            with gr.Group():
                                gr.Markdown("### Speaker Separation")
                                with gr.Row():
                                    components["proc_expected_speakers"] = gr.Dropdown(
                                        choices=["2", "3"],
                                        value=default_expected_speakers,
                                        label="Expected Speakers",
                                    )
                                    components["proc_separate_speakers_btn"] = gr.Button(
                                        "Separate Speakers",
                                        variant="secondary",
                                        size="sm",
                                    )
                                components["proc_speaker_separation_note"] = gr.Markdown(
                                    speaker_note
                                )

                            with gr.Group():
                                gr.Markdown("### Save")
                                components["proc_save_dataset_folder"] = gr.Dropdown(
                                    choices=["(Select Dataset)"] + get_dataset_folders(),
                                    value="(Select Dataset)",
                                    label="Dataset destination folder",
                                )
                                with gr.Row():
                                    components["proc_save_primary_btn"] = gr.Button("Save (Primary)", variant="primary")
                                    components["proc_save_secondary_btn"] = gr.Button("Save (Secondary)", variant="secondary")

            components["lm_existing_names_json"] = gr.Textbox(visible=False)
            components["lm_default_name"] = gr.Textbox(visible=False)
            components["proc_original_audio_path"] = gr.Textbox(visible=False)
            components["proc_source_kind"] = gr.Textbox(visible=False)
            components["proc_source_dataset_folder_state"] = gr.Textbox(visible=False)
            components["dataset_split_prefix_auto_state"] = gr.State("")

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        tenant_service = shared_state["tenant_service"]
        get_tenant_paths = shared_state["get_tenant_paths"]
        get_sample_choices = shared_state["get_sample_choices"]
        get_dataset_folders = shared_state["get_dataset_folders"]
        get_dataset_files = shared_state["get_dataset_files"]
        load_sample_details = shared_state["load_sample_details"]
        get_prompt_cache_path = shared_state["get_prompt_cache_path"]
        save_preference = shared_state["save_preference"]
        format_usage = shared_state["format_tenant_usage_meter"]
        get_usage_summary = shared_state["get_tenant_usage_summary"]
        get_tenant_samples_dir = shared_state["get_tenant_samples_dir"]
        get_tenant_datasets_dir = shared_state["get_tenant_datasets_dir"]
        is_audio_file = shared_state["is_audio_file"]
        is_video_file = shared_state["is_video_file"]
        extract_audio_from_video = shared_state["extract_audio_from_video"]
        normalize_audio = shared_state["normalize_audio"]
        convert_to_mono = shared_state["convert_to_mono"]
        clean_audio = shared_state["clean_audio"]
        show_confirmation_modal_js = shared_state["show_confirmation_modal_js"]
        show_input_modal_js = shared_state["show_input_modal_js"]
        confirm_trigger = shared_state["confirm_trigger"]
        input_trigger = shared_state["input_trigger"]
        _user_config = shared_state["_user_config"]
        play_completion_beep = shared_state.get("play_completion_beep")
        deepfilter_available = bool(shared_state.get("DEEPFILTER_AVAILABLE", False))
        run_heavy_job = shared_state.get("run_heavy_job")
        audio_route_trigger = shared_state.get("audio_route_trigger")

        asr_manager = get_asr_manager()
        speaker_separation_manager = get_speaker_separation_manager(user_config=_user_config)

        def _parse_modal_value(payload: str, prefix: str) -> tuple[str | None, str | None]:
            if not payload or not payload.startswith(prefix):
                return None, None
            body = payload[len(prefix):]
            value, sep, tail = body.rpartition("_")
            if not sep or not tail.isdigit():
                return None, None
            mode = None
            name = value
            if "::" in value:
                maybe_mode, maybe_name = value.split("::", 1)
                if maybe_mode in {"new", "replace"}:
                    mode = maybe_mode
                    name = maybe_name
            return name, mode

        def _is_confirm(payload: str, prefix: str) -> bool:
            return bool(payload and payload.startswith(prefix) and "_confirm_" in payload)

        def _tenant_banner(request: gr.Request):
            try:
                paths = get_tenant_paths(request=request, strict=True)
                return f"Tenant: `{paths.tenant_id}`"
            except Exception as e:
                return f"Tenant: error ({str(e)})"

        def _usage_banner(request: gr.Request):
            try:
                summary = get_usage_summary(request=request, strict=True)
                if not summary:
                    return "Tenant storage: -"
                return format_usage(summary)
            except Exception as e:
                return f"Tenant storage: error ({str(e)})"

        def _clean_name(value: str) -> str:
            return sanitize_filename(value, keep_extension=False)

        def _default_name_from_path(path: str | None) -> str:
            if not path:
                return ""
            try:
                return _clean_name(Path(path).stem)
            except Exception:
                return ""

        def _legacy_sample_audio_files(request: gr.Request) -> list[Path]:
            sample_dir = get_tenant_samples_dir(request=request, strict=True)
            legacy = []
            for p in sample_dir.iterdir() if sample_dir.exists() else []:
                if not p.is_file():
                    continue
                suffix = p.suffix.lower()
                if suffix in AUDIO_EXTENSIONS and suffix != ".wav":
                    legacy.append(p)
            return sorted(legacy, key=lambda p: p.name.lower())

        def _legacy_sample_warning_updates(request: gr.Request):
            legacy = _legacy_sample_audio_files(request)
            if not legacy:
                return gr.update(visible=False, value=""), gr.update(visible=False)
            names = ", ".join(p.name for p in legacy[:5])
            if len(legacy) > 5:
                names += f" (+{len(legacy) - 5} more)"
            text = (
                "⚠ Legacy non-WAV sample files detected. Convert to canonical WAV for consistent transcript/cache behavior.\\n\\n"
                f"Found: `{names}`"
            )
            return gr.update(visible=True, value=text), gr.update(visible=True)

        def _load_audio_from_path_or_video(path: str | None, request: gr.Request):
            if not path:
                return None, "No file selected"
            if is_video_file(path):
                extracted, msg = extract_audio_from_video(path, request)
                if not extracted:
                    return None, msg
                return extracted, f"Loaded video source -> extracted audio ({Path(extracted).name})"
            if is_audio_file(path):
                return path, f"Loaded audio source ({Path(path).name})"
            return None, "Unsupported source format"

        def _refresh_dataset_files(folder: str, request: gr.Request):
            if not folder or folder == "(Select Dataset)":
                return []
            return get_dataset_files(folder, request=request, strict=True)

        def _dataset_folder_updates(selected_folder: str | None, request: gr.Request):
            folders = ["(Select Dataset)"] + get_dataset_folders(request=request, strict=True)
            selected = selected_folder if selected_folder in folders else "(Select Dataset)"
            return (
                gr.update(choices=folders, value=selected),
                gr.update(choices=folders, value=selected),
                gr.update(choices=folders, value=selected),
            )

        def _list_existing_sample_stems(request: gr.Request):
            sample_dir = get_tenant_samples_dir(request=request, strict=True)
            return sorted([p.stem for p in sample_dir.glob("*.wav")])

        def _list_existing_dataset_stems(folder: str, request: gr.Request):
            if not folder or folder == "(Select Dataset)":
                return []
            ds_dir = get_tenant_datasets_dir(request=request, strict=True) / folder
            if not ds_dir.exists():
                return []
            return sorted([p.stem for p in ds_dir.glob("*.wav")] + [p.stem for p in ds_dir.glob("*.mp3")])

        def _prepare_uploads(files, request: gr.Request):
            prepared: list[tuple[Path, str, int]] = []
            for src in files or []:
                src_path = Path(src)
                if not src_path.exists() or not src_path.is_file():
                    return None, f"Upload source not found: {src_path}"

                if is_video_file(str(src_path)):
                    extracted, emsg = extract_audio_from_video(str(src_path), request)
                    if not extracted:
                        return None, f"Failed extracting {src_path.name}: {emsg}"
                    media_path = Path(extracted)
                    dest_name = sanitize_filename(media_path.stem, keep_extension=False) + ".wav"
                else:
                    if not is_allowed_media(src_path.name):
                        return None, f"Unsupported media format: {src_path.name}"
                    media_path = src_path
                    dest_name = sanitize_filename(src_path.name, keep_extension=True)

                size = media_path.stat().st_size
                prepared.append((media_path, dest_name, size))
            return prepared, "ok"

        def _prepare_sample_uploads(files, request: gr.Request):
            prepared: list[tuple[np.ndarray, int, str, int]] = []
            for src in files or []:
                src_path = Path(src)
                if not src_path.exists() or not src_path.is_file():
                    return None, f"Upload source not found: {src_path}"

                if is_video_file(str(src_path)):
                    extracted, emsg = extract_audio_from_video(str(src_path), request)
                    if not extracted:
                        return None, f"Failed extracting {src_path.name}: {emsg}"
                    media_path = Path(extracted)
                else:
                    if not is_allowed_media(src_path.name):
                        return None, f"Unsupported media format: {src_path.name}"
                    media_path = src_path

                try:
                    audio_data, sr = sf.read(str(media_path), dtype="float32")
                except Exception as e:
                    return None, f"Failed reading {media_path.name}: {str(e)}"

                channels = int(audio_data.shape[1]) if len(audio_data.shape) > 1 else 1
                est_size = estimate_pcm16_wav_bytes(len(audio_data), channels=channels)
                dest_name = sanitize_filename(src_path.stem, keep_extension=False) + ".wav"
                prepared.append((audio_data, sr, dest_name, est_size))
            return prepared, "ok"

        def _validate_prepared_uploads(paths, prepared, label: str):
            if not prepared:
                return False, "No files selected"

            for _path, name, size in prepared:
                if size > tenant_service.file_limit_bytes:
                    return False, (
                        f"{label} file '{name}' exceeds per-file limit of "
                        f"{tenant_service.tenant_file_limit_mb} MB"
                    )

            incoming = sum(size for _path, _name, size in prepared)
            used = tenant_service.get_media_usage_bytes(paths)
            if used + incoming > tenant_service.media_quota_bytes:
                return False, (
                    f"{label} would exceed tenant media quota ({tenant_service.tenant_media_quota_gb} GB). "
                    f"Current usage: {used / (1024**3):.2f} GB"
                )
            return True, "ok"

        def _validate_generated_sample_uploads(paths, prepared, label: str):
            if not prepared:
                return False, "No files selected"

            for _audio, _sr, name, est_size in prepared:
                if est_size > tenant_service.file_limit_bytes:
                    return False, (
                        f"{label} file '{name}' exceeds per-file limit of "
                        f"{tenant_service.tenant_file_limit_mb} MB"
                    )

            incoming = sum(est_size for _audio, _sr, _name, est_size in prepared)
            used = tenant_service.get_media_usage_bytes(paths)
            if used + incoming > tenant_service.media_quota_bytes:
                return False, (
                    f"{label} would exceed tenant media quota ({tenant_service.tenant_media_quota_gb} GB). "
                    f"Current usage: {used / (1024**3):.2f} GB"
                )
            return True, "ok"

        def _clear_sample_cache(sample_name: str, request: gr.Request):
            sample_name = _clean_name(sample_name)
            if not sample_name:
                return "Invalid sample name"
            sample_dir = get_tenant_samples_dir(request=request, strict=True)
            cleared: list[str] = []
            for model_size in ("0.6B", "1.7B"):
                cache_path = get_prompt_cache_path(sample_name, model_size, request=request, strict=True)
                if cache_path.exists():
                    cache_path.unlink()
                    cleared.append(model_size)
            lux_path = sample_dir / f"{sample_name}_luxtts.pt"
            if lux_path.exists():
                lux_path.unlink()
                cleared.append("LuxTTS")
            if not cleared:
                return "No cache files found"
            return f"Cache cleared for {sample_name}: {', '.join(cleared)}"

        def _run_transcribe(audio_file: str, transcribe_model: str, language: str):
            if not audio_file:
                return "", "Load an audio file first"
            engine, size = parse_asr_model(transcribe_model)
            options = {}

            if engine == "Qwen3 ASR":
                model = asr_manager.get_qwen3_asr(size=size)
                if language and language != "Auto-detect":
                    options["language"] = language
                result = model.transcribe(audio_file, **options)
            elif engine == "VibeVoice ASR":
                model = asr_manager.get_vibevoice_asr()
                result = model.transcribe(audio_file)
            else:
                if not asr_manager.whisper_available:
                    return "", "Whisper not available"
                model = asr_manager.get_whisper(size=size or "Medium")
                lang_code = language_to_code(language)
                if lang_code:
                    options["language"] = lang_code
                result = model.transcribe(audio_file, **options)

            text = clean_transcription_for_engine(engine, (result.get("text") or ""))
            return text, f"Transcribed with {engine}"

        def _save_sample_meta(
            sample_dir: Path,
            sample_name: str,
            transcript: str,
            extra_fields: dict | None = None,
        ):
            meta_path = sample_dir / f"{sample_name}.json"
            existing = {}
            if meta_path.exists():
                try:
                    existing = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
            data = dict(existing)
            data["Name"] = sample_name
            data["name"] = sample_name
            data["Type"] = data.get("Type", "Sample")
            data["Text"] = (transcript or "").strip()
            data["text"] = data["Text"]
            if extra_fields:
                data.update(extra_fields)
                data["Name"] = sample_name
                data["name"] = sample_name
                data["Type"] = "Sample"
                data["Text"] = (transcript or "").strip()
                data["text"] = data["Text"]
            meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        def _speaker_separation_note(expected_speakers):
            if str(expected_speakers) == "3":
                return (
                    "3-speaker separation uses the upstream SpeechBrain 8 kHz model. "
                    "Imported samples are lower fidelity than the 2-speaker path."
                )
            return "Uses the 2-speaker SpeechBrain model at 16 kHz."

        def _save_sample_from_processing(
            audio_path: str,
            transcript: str,
            raw_name: str,
            save_mode: str,
            request: gr.Request,
        ):
            if not audio_path:
                return "No audio loaded"

            sample_name = _clean_name(raw_name)
            if not sample_name:
                return "Invalid sample name"

            paths = get_tenant_paths(request=request, strict=True)
            sample_dir = paths.samples_dir
            sample_dir.mkdir(parents=True, exist_ok=True)

            if save_mode == "replace":
                dst_wav = sample_dir / f"{sample_name}.wav"
                if not dst_wav.exists():
                    return f"Replace failed: sample '{sample_name}.wav' does not exist"
                final_name = sample_name
            else:
                desired = sample_dir / f"{sample_name}.wav"
                dst_wav = desired if not desired.exists() else collision_safe_path(sample_dir, f"{sample_name}.wav")
                final_name = dst_wav.stem

            try:
                audio_data, sr = sf.read(audio_path, dtype="float32")
            except Exception as e:
                return f"Failed to read audio: {str(e)}"

            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            estimated_size = estimate_pcm16_wav_bytes(len(audio_data), channels=1)
            reclaimed = dst_wav.stat().st_size if dst_wav.exists() else 0
            ok, msg = tenant_service.validate_generated_sizes(
                paths,
                [estimated_size],
                label="Sample save",
                reclaimed_bytes=reclaimed,
            )
            if not ok:
                return msg

            sf.write(str(dst_wav), audio_data, sr, subtype="PCM_16")
            _save_sample_meta(sample_dir, final_name, transcript)
            _clear_sample_cache(final_name, request=request)

            mode_label = "Replace existing" if save_mode == "replace" else "Save as new"
            return f"Saved to Samples ({mode_label}): {final_name}.wav"

        def _save_dataset_clip_from_processing(
            audio_path: str,
            transcript: str,
            dataset_folder: str,
            raw_name: str,
            save_mode: str,
            request: gr.Request,
        ):
            if not audio_path:
                return "No audio loaded"
            if not dataset_folder or dataset_folder == "(Select Dataset)":
                return "Select a target dataset folder"

            clip_name = _clean_name(raw_name)
            if not clip_name:
                return "Invalid clip name"

            paths = get_tenant_paths(request=request, strict=True)
            target_dir = paths.datasets_dir / _clean_name(dataset_folder)
            target_dir.mkdir(parents=True, exist_ok=True)

            if save_mode == "replace":
                dst_wav = target_dir / f"{clip_name}.wav"
                if not dst_wav.exists():
                    return f"Replace failed: dataset clip '{clip_name}.wav' does not exist in '{dataset_folder}'"
                final_name = clip_name
            else:
                desired = target_dir / f"{clip_name}.wav"
                dst_wav = desired if not desired.exists() else collision_safe_path(target_dir, f"{clip_name}.wav")
                final_name = dst_wav.stem

            try:
                audio_data, sr = sf.read(audio_path, dtype="float32")
            except Exception as e:
                return f"Failed to read audio: {str(e)}"

            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            estimated_size = estimate_pcm16_wav_bytes(len(audio_data), channels=1)
            reclaimed = dst_wav.stat().st_size if dst_wav.exists() else 0
            ok, msg = tenant_service.validate_generated_sizes(
                paths,
                [estimated_size],
                label="Dataset clip save",
                reclaimed_bytes=reclaimed,
            )
            if not ok:
                return msg

            sf.write(str(dst_wav), audio_data, sr, subtype="PCM_16")
            (dst_wav.with_suffix(".txt")).write_text((transcript or "").strip(), encoding="utf-8")
            mode_label = "Replace existing" if save_mode == "replace" else "Save as new"
            return f"Saved to Dataset ({mode_label}): {dataset_folder}/{final_name}.wav"

        def _import_separated_tracks_to_samples(
            separated_tracks,
            *,
            base_name: str,
            source_identifier: str,
            expected_speakers: int,
            request: gr.Request,
        ):
            tracks = list(separated_tracks or [])
            if not tracks:
                return [], "Speaker separation did not return any tracks."

            paths = get_tenant_paths(request=request, strict=True)
            sample_dir = paths.samples_dir
            sample_dir.mkdir(parents=True, exist_ok=True)

            plans = build_speaker_separation_import_plan(
                base_name,
                tracks,
                source_identifier=source_identifier,
                expected_speakers=int(expected_speakers),
            )
            ok, msg = tenant_service.validate_generated_sizes(
                paths,
                [plan.estimated_size_bytes for plan in plans],
                label="Speaker separation import",
            )
            if not ok:
                return [], msg

            created_names: list[str] = []
            for plan, track in zip(plans, tracks):
                desired = sample_dir / f"{plan.sample_name}.wav"
                dst_wav = desired if not desired.exists() else collision_safe_path(sample_dir, f"{plan.sample_name}.wav")
                final_name = dst_wav.stem

                audio_data = np.asarray(getattr(track, "audio_data", []), dtype=np.float32).reshape(-1)
                audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
                audio_data = np.clip(audio_data, -1.0, 1.0)
                sample_rate = int(getattr(track, "sample_rate", plan.sample_rate) or plan.sample_rate or 0)
                if sample_rate <= 0:
                    return created_names, f"Invalid sample rate for separated speaker {plan.speaker_index}"

                sf.write(str(dst_wav), audio_data, sample_rate, subtype="PCM_16")
                extra_fields = build_speaker_separation_sample_metadata(
                    final_name,
                    source_identifier=source_identifier,
                    expected_speakers=int(expected_speakers),
                    speaker_index=plan.speaker_index,
                )
                _save_sample_meta(sample_dir, final_name, "", extra_fields=extra_fields)
                _clear_sample_cache(final_name, request=request)
                created_names.append(final_name)

            return created_names, (
                f"Imported {len(created_names)} separated speaker samples into Samples: "
                + ", ".join(f"{name}.wav" for name in created_names)
            )

        def _auto_split_to_dataset(
            audio_path: str,
            dataset_folder: str,
            clip_prefix: str,
            transcribe_model: str,
            language: str,
            split_min: float,
            split_max: float,
            silence_trim: float,
            discard_under: float,
            request: gr.Request,
            progress=gr.Progress(),
        ):
            if not audio_path:
                return "Load an audio file first"
            if not dataset_folder or dataset_folder == "(Select Dataset)":
                return "Select a target dataset folder"
            prefix = _clean_name(clip_prefix)
            if not prefix:
                return "Invalid clip prefix"

            info = sf.info(audio_path)
            duration = info.duration
            engine, asr_size = parse_asr_model(transcribe_model)
            if engine == "Qwen3 ASR" and duration > 300 and not _user_config.get("bypass_split_limit", False):
                return (
                    "Audio exceeds the recommended 5 minute limit for Qwen3 ASR auto-split. "
                    "Enable extended split in Settings or use Whisper."
                )

            word_timestamps = []
            full_text = ""
            progress(0.05, desc="Transcribing source")

            if engine == "Whisper":
                model = asr_manager.get_whisper(size=asr_size or "Medium")
                options = {"word_timestamps": True}
                code = language_to_code(language)
                if code:
                    options["language"] = code
                result = model.transcribe(audio_path, **options)
                full_text = (result.get("text") or "").strip()
                for seg in result.get("segments", []):
                    for w in seg.get("words", []):
                        word_timestamps.append(
                            WordTimestampLike(
                                text=w.get("word", "").strip(),
                                start_time=float(w.get("start", 0.0)),
                                end_time=float(w.get("end", 0.0)),
                            )
                        )
            else:
                model = asr_manager.get_qwen3_asr(size=asr_size or "Large")
                lang_option = language if language and language != "Auto-detect" else None
                asr_result = model.transcribe(audio_path, language=lang_option)
                full_text = (asr_result.get("text") or "").strip()
                try:
                    aligner = asr_manager.get_qwen3_forced_aligner()
                    align_results = aligner.align(
                        audio=audio_path,
                        text=full_text,
                        language=lang_option or asr_result.get("language"),
                    )
                    raw_ts = align_results[0] if align_results else []
                    for item in raw_ts:
                        word_timestamps.append(
                            WordTimestampLike(
                                text=getattr(item, "text", ""),
                                start_time=float(getattr(item, "start_time", 0.0)),
                                end_time=float(getattr(item, "end_time", 0.0)),
                            )
                        )
                finally:
                    asr_manager.unload_forced_aligner()

            if not full_text or not word_timestamps:
                return "Could not generate aligned word timestamps for auto-split"

            progress(0.45, desc="Building split segments")
            segments = split_into_segments(
                full_text=full_text,
                word_timestamps=word_timestamps,
                min_duration=float(split_min),
                max_duration=float(split_max),
                silence_trim=float(silence_trim),
                discard_under=float(discard_under),
            )
            if not segments:
                return "No valid segments produced by auto-split"

            data, sr = sf.read(audio_path, dtype="float32")
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            clip_sizes = []
            sample_ranges = []
            for seg_start, seg_end, _text in segments:
                start_sample = max(0, int(seg_start * sr))
                end_sample = min(len(data), int(seg_end * sr))
                if end_sample <= start_sample:
                    continue
                sample_ranges.append((start_sample, end_sample))
                clip_sizes.append(estimate_pcm16_wav_bytes(end_sample - start_sample, channels=1))
            if not clip_sizes:
                return "No valid sample ranges produced by auto-split"

            paths = get_tenant_paths(request=request, strict=True)
            ok, msg = tenant_service.validate_generated_sizes(paths, clip_sizes, label="Auto-split output")
            if not ok:
                return msg

            target_dir = paths.datasets_dir / _clean_name(dataset_folder)
            target_dir.mkdir(parents=True, exist_ok=True)

            existing_numbers = []
            for f in target_dir.glob(f"{prefix}_*.wav"):
                suffix = f.stem[len(prefix) + 1:]
                if suffix.isdigit():
                    existing_numbers.append(int(suffix))
            next_number = (max(existing_numbers) + 1) if existing_numbers else 1

            progress(0.70, desc="Saving split clips")
            saved = 0
            for idx, ((start_sample, end_sample), (_, _, seg_text)) in enumerate(zip(sample_ranges, segments), start=1):
                clip_name = f"{prefix}_{next_number + idx - 1:03d}"
                clip_data = data[start_sample:end_sample]
                clip_path = target_dir / f"{clip_name}.wav"
                sf.write(str(clip_path), clip_data, sr, subtype="PCM_16")
                clip_path.with_suffix(".txt").write_text(seg_text.strip(), encoding="utf-8")
                saved += 1
                progress(0.70 + (0.30 * (idx / max(len(sample_ranges), 1))), desc=f"Saved {idx}/{len(sample_ranges)}")

            if play_completion_beep:
                play_completion_beep()
            return f"Auto-split complete: saved {saved} clips into '{dataset_folder}'"

        def _prepare_processing_context(
            context: ProcessingSourceContext,
            transcript: str,
            default_name: str,
            dataset_target: str,
            primary_label: str,
            secondary_label: str,
            status: str,
        ):
            return (
                gr.update(selected="library_processing"),
                gr.update(visible=True, value=context.original_audio_path),
                gr.update(value=transcript),
                gr.update(value=default_name),
                gr.update(value=context.original_audio_path),
                gr.update(value=context.source_type.lower()),
                gr.update(value=context.source_dataset_folder),
                gr.update(value=context.source_type),
                gr.update(value=context.source_identifier),
                gr.update(value=Path(context.original_audio_path).name),
                gr.update(value=dataset_target),
                gr.update(value=primary_label),
                gr.update(value=secondary_label),
                status,
            )

        def _on_sample_select(lister_value, request: gr.Request):
            selected = (lister_value or {}).get("selected", [])
            if len(selected) != 1:
                return None, "", " ", ""
            sample_name = selected[0]
            clean = sample_name[:-4] if sample_name.lower().endswith(".wav") else sample_name
            audio_path, transcript, info = load_sample_details(clean, request=request, strict=True)
            return audio_path, transcript, info, "Saved"

        def _on_dataset_select(folder, lister_value, current_prefix, auto_prefix_state, request: gr.Request):
            selected = (lister_value or {}).get("selected", [])
            current_prefix = (current_prefix or "").strip()
            auto_prefix_state = auto_prefix_state or ""

            if not folder or folder == "(Select Dataset)" or not selected:
                should_update_prefix = (not current_prefix) or (current_prefix == auto_prefix_state)
                prefix_update = gr.update(value="") if should_update_prefix else gr.update()
                auto_state_update = ""
                return None, "", "", prefix_update, auto_state_update

            if len(selected) != 1:
                should_update_prefix = (not current_prefix) or (current_prefix == auto_prefix_state)
                prefix_update = gr.update(value="") if should_update_prefix else gr.update()
                auto_state_update = "" if should_update_prefix else auto_prefix_state
                return None, "", "", prefix_update, auto_state_update

            file_name = selected[0]
            file_path = get_tenant_datasets_dir(request=request, strict=True) / folder / file_name
            transcript = ""
            txt_path = file_path.with_suffix(".txt")
            if txt_path.exists():
                transcript = txt_path.read_text(encoding="utf-8")

            suggested_prefix = _clean_name(Path(file_name).stem) or "clip"
            should_update_prefix = (not current_prefix) or (current_prefix == auto_prefix_state)
            prefix_update = gr.update(value=suggested_prefix) if should_update_prefix else gr.update()
            auto_state_update = suggested_prefix if should_update_prefix else auto_prefix_state

            return (
                str(file_path) if file_path.exists() else None,
                transcript,
                "Saved",
                prefix_update,
                auto_state_update,
            )

        def _on_open_sample_processing(lister_value, request: gr.Request):
            selected = (lister_value or {}).get("selected", [])
            if len(selected) != 1:
                msg = "Select exactly one sample first"
                return (
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), msg, msg, msg,
                )
            sample_name = selected[0]
            clean = sample_name[:-4] if sample_name.lower().endswith(".wav") else sample_name
            audio_path, transcript, _info = load_sample_details(clean, request=request, strict=True)
            if not audio_path:
                msg = "Sample not found"
                return (
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), msg, msg, msg,
                )

            ctx = ProcessingSourceContext(
                source_type="Sample",
                source_identifier=clean,
                original_audio_path=audio_path,
                source_dataset_folder="",
            )
            proc_updates = _prepare_processing_context(
                context=ctx,
                transcript=transcript,
                default_name=_clean_name(clean),
                dataset_target="(Select Dataset)",
                primary_label="Save to Samples (Primary)",
                secondary_label="Save to Dataset",
                status="Sample opened in Processing Studio. Pipeline starts from original source.",
            )
            return (*proc_updates, "Sample opened in Processing Studio", "Sample opened in Processing Studio")

        def _on_open_dataset_processing(folder, lister_value, request: gr.Request):
            selected = (lister_value or {}).get("selected", [])
            if len(selected) != 1:
                msg = "Select exactly one dataset file first"
                return (
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), msg, msg, msg,
                )
            if not folder or folder == "(Select Dataset)":
                msg = "Select dataset folder first"
                return (
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), msg, msg, msg,
                )

            file_name = selected[0]
            audio_path = get_tenant_datasets_dir(request=request, strict=True) / folder / file_name
            if not audio_path.exists():
                msg = "Selected file not found"
                return (
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), msg, msg, msg,
                )
            txt_path = audio_path.with_suffix(".txt")
            transcript = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""

            ctx = ProcessingSourceContext(
                source_type="Dataset",
                source_identifier=f"{folder}/{file_name}",
                original_audio_path=str(audio_path),
                source_dataset_folder=folder,
            )
            proc_updates = _prepare_processing_context(
                context=ctx,
                transcript=transcript,
                default_name=_clean_name(audio_path.stem),
                dataset_target=folder,
                primary_label="Save to Dataset (Primary)",
                secondary_label="Save to Samples",
                status="Dataset file opened in Processing Studio. Pipeline starts from original source.",
            )
            return (*proc_updates, "Dataset file opened in Processing Studio", "Dataset file opened in Processing Studio")

        def _on_proc_apply_pipeline(original_audio, enable_denoise, enable_normalize, enable_mono, request: gr.Request):
            if not original_audio:
                return gr.update(), "No source loaded"
            if not Path(original_audio).exists():
                return gr.update(), "Original source no longer exists"

            pipeline = ProcessingPipelineConfig(
                enable_denoise=bool(enable_denoise),
                enable_normalize=bool(enable_normalize),
                enable_mono=bool(enable_mono),
            )
            current_audio = original_audio
            applied_steps: list[str] = []

            if pipeline.enable_denoise:
                if not deepfilter_available:
                    return gr.update(visible=True, value=original_audio), "Denoise unavailable in this environment"
                current_audio, msg = clean_audio(current_audio)
                if not current_audio:
                    return gr.update(), msg or "Denoise failed"
                applied_steps.append("Denoise")

            if pipeline.enable_normalize:
                current_audio, msg = normalize_audio(current_audio, request=request)
                if not current_audio:
                    return gr.update(), msg or "Normalize failed"
                applied_steps.append("Normalize")

            if pipeline.enable_mono:
                current_audio, msg = convert_to_mono(current_audio, request=request)
                if not current_audio:
                    return gr.update(), msg or "Mono conversion failed"
                applied_steps.append("Mono")

            if not applied_steps:
                return gr.update(visible=True, value=original_audio), "No pipeline steps enabled. Using original source."

            return (
                gr.update(visible=True, value=current_audio),
                f"Pipeline applied from original source: {' -> '.join(applied_steps)}",
            )

        def _on_proc_reset(original_audio):
            if not original_audio:
                return gr.update(), "No source loaded"
            return gr.update(visible=True, value=original_audio), "Reset to original source"

        def _on_proc_transcribe_click(audio, model, lang, request: gr.Request):
            try:
                if run_heavy_job:
                    return run_heavy_job(
                        "library_manager_transcribe_single",
                        lambda: _run_transcribe(audio, model, lang),
                        request=request,
                    )
                return _run_transcribe(audio, model, lang)
            except MemoryAdmissionError as exc:
                return f"⚠ Memory safety guard rejected request: {str(exc)}", "Error"

        def _on_proc_separate_speakers_click(
            audio,
            expected_speakers,
            default_name,
            source_identifier,
            request: gr.Request,
        ):
            def _run_impl():
                audio_path = str(audio or "").strip()
                if not audio_path:
                    msg = "Load or generate audio in Processing Studio first"
                    return gr.update(), gr.update(), gr.update(), msg, msg, msg
                if not Path(audio_path).exists():
                    msg = f"Processing audio was not found: {audio_path}"
                    return gr.update(), gr.update(), gr.update(), msg, msg, msg

                try:
                    speaker_count = int(expected_speakers)
                except Exception:
                    msg = f"Invalid expected speaker count: {expected_speakers}"
                    return gr.update(), gr.update(), gr.update(), msg, msg, msg

                base_name = _clean_name(default_name) or _default_name_from_path(audio_path) or "separated_sample"
                source_label = str(source_identifier or Path(audio_path).name).strip() or Path(audio_path).name

                separated_tracks = speaker_separation_manager.separate_file(audio_path, speaker_count)
                created_names, status = _import_separated_tracks_to_samples(
                    separated_tracks,
                    base_name=base_name,
                    source_identifier=source_label,
                    expected_speakers=speaker_count,
                    request=request,
                )
                if not created_names:
                    return gr.update(), gr.update(), gr.update(), status, status, status

                return (
                    gr.update(selected="library_samples"),
                    get_sample_choices(request=request, strict=True),
                    _usage_banner(request),
                    status,
                    status,
                    status,
                )

            try:
                if run_heavy_job:
                    return run_heavy_job(
                        "library_manager_speaker_separation",
                        _run_impl,
                        request=request,
                    )
                return _run_impl()
            except MemoryAdmissionError as exc:
                msg = f"⚠ Memory safety guard rejected request: {str(exc)}"
                return gr.update(), gr.update(), gr.update(), msg, msg, msg
            except Exception as exc:
                msg = f"Speaker separation failed: {str(exc)}"
                return gr.update(), gr.update(), gr.update(), msg, msg, msg

        def _on_upload_samples(upload_files, request: gr.Request):
            paths = get_tenant_paths(request=request, strict=True)
            files = upload_files or []
            if not files:
                return "No files selected", gr.update(), _usage_banner(request), *(_legacy_sample_warning_updates(request))

            prepared, prep_msg = _prepare_sample_uploads(files, request=request)
            if prepared is None:
                return prep_msg, gr.update(), _usage_banner(request), *(_legacy_sample_warning_updates(request))

            ok, msg = _validate_generated_sample_uploads(paths, prepared, label="Sample upload")
            if not ok:
                return msg, gr.update(), _usage_banner(request), *(_legacy_sample_warning_updates(request))

            saved = 0
            for audio_data, sr, dest_name, _size in prepared:
                dst = collision_safe_path(paths.samples_dir, dest_name)
                sf.write(str(dst), audio_data, sr, subtype="PCM_16")
                saved += 1

            return (
                f"Uploaded {saved} sample file(s) as canonical WAV",
                get_sample_choices(request=request, strict=True),
                _usage_banner(request),
                *(_legacy_sample_warning_updates(request)),
            )

        def _on_convert_legacy_samples(request: gr.Request):
            sample_dir = get_tenant_samples_dir(request=request, strict=True)
            legacy_files = _legacy_sample_audio_files(request)
            if not legacy_files:
                return "No legacy sample files found", gr.update(), *(_legacy_sample_warning_updates(request)), _usage_banner(request)

            converted = 0
            failed = 0
            for src in legacy_files:
                try:
                    data, sr = sf.read(str(src), dtype="float32")
                    dst = collision_safe_path(sample_dir, f"{sanitize_filename(src.stem, keep_extension=False)}.wav")
                    sf.write(str(dst), data, sr, subtype="PCM_16")
                    src_meta = src.with_suffix(".json")
                    if src_meta.exists() and not dst.with_suffix(".json").exists():
                        shutil.copy2(src_meta, dst.with_suffix(".json"))
                    converted += 1
                except Exception:
                    failed += 1

            status = f"Legacy conversion complete: {converted} converted"
            if failed:
                status += f", {failed} failed"
            return (
                status,
                get_sample_choices(request=request, strict=True),
                *(_legacy_sample_warning_updates(request)),
                _usage_banner(request),
            )

        def _on_upload_dataset(folder, upload_files, request: gr.Request):
            if not folder or folder == "(Select Dataset)":
                return "Select dataset folder first", gr.update(), _usage_banner(request)
            paths = get_tenant_paths(request=request, strict=True)
            files = upload_files or []
            prepared, prep_msg = _prepare_uploads(files, request=request)
            if prepared is None:
                return prep_msg, gr.update(), _usage_banner(request)

            ok, msg = _validate_prepared_uploads(paths, prepared, label="Dataset upload")
            if not ok:
                return msg, gr.update(), _usage_banner(request)

            target_dir = paths.datasets_dir / _clean_name(folder)
            target_dir.mkdir(parents=True, exist_ok=True)
            saved = []
            for src_path, src_name, _size in prepared:
                dst = collision_safe_path(target_dir, src_name)
                shutil.copy2(src_path, dst)
                saved.append(dst)
            return (
                f"Uploaded {len(saved)} file(s) to {folder}",
                get_dataset_files(folder, request=request, strict=True),
                _usage_banner(request),
            )

        def _on_create_dataset(folder_name, request: gr.Request):
            if not folder_name or not folder_name.strip():
                return "Enter a dataset folder name", gr.update(), gr.update(), gr.update(), gr.update(), _usage_banner(request)
            paths = get_tenant_paths(request=request, strict=True)
            try:
                target = tenant_service.create_dataset_folder(paths, folder_name.strip())
            except FileExistsError:
                return "Folder already exists", gr.update(), gr.update(), gr.update(), gr.update(), _usage_banner(request)
            folder_dropdown, save_dropdown, split_dropdown = _dataset_folder_updates(
                target.name,
                request=request,
            )
            value = target.name
            return (
                f"Created dataset '{target.name}'",
                folder_dropdown,
                get_dataset_files(value, request=request, strict=True) if value != "(Select Dataset)" else [],
                save_dropdown,
                split_dropdown,
                _usage_banner(request),
            )

        def _autosave_sample_transcript(lister_value, transcript, request: gr.Request):
            selected = (lister_value or {}).get("selected", [])
            if len(selected) != 1:
                return "Select one sample first", "Error"

            sample_name = Path(selected[0]).stem
            paths = get_tenant_paths(request=request, strict=True)
            wav_path = paths.samples_dir / f"{sample_name}.wav"
            if not wav_path.exists():
                return f"Sample not found: {sample_name}", "Error"

            try:
                _save_sample_meta(paths.samples_dir, sample_name, transcript)
                return f"Saved transcript for {sample_name}", "Saved"
            except Exception as e:
                return f"Error saving transcript for {sample_name}: {str(e)}", "Error"

        def _autosave_dataset_transcript(folder, lister_value, transcript, request: gr.Request):
            selected = (lister_value or {}).get("selected", [])
            if len(selected) != 1 or not folder or folder == "(Select Dataset)":
                return "Select one dataset file first", "Error"
            paths = get_tenant_paths(request=request, strict=True)
            audio_path = paths.datasets_dir / folder / selected[0]
            txt_path = audio_path.with_suffix(".txt")
            try:
                txt_path.write_text((transcript or "").strip(), encoding="utf-8")
                return f"Saved transcript for {selected[0]}", "Saved"
            except Exception as e:
                return f"Error saving transcript for {selected[0]}: {str(e)}", "Error"

        def _on_batch_transcribe(folder, replace_existing, language, transcribe_model, request: gr.Request, progress=gr.Progress()):
            def _run_impl():
                if not folder or folder == "(Select Dataset)":
                    return "Select a dataset first"
                base_dir = get_tenant_datasets_dir(request=request, strict=True) / folder
                files = sorted(list(base_dir.glob("*.wav")) + list(base_dir.glob("*.mp3")))
                if not files:
                    return f"No files in {folder}"

                done = 0
                skipped = 0
                failed = 0
                for idx, audio_file in enumerate(files):
                    txt_path = audio_file.with_suffix(".txt")
                    if txt_path.exists() and not replace_existing:
                        skipped += 1
                        continue
                    progress((idx + 1) / max(len(files), 1), desc=f"{idx + 1}/{len(files)} {audio_file.name}")
                    try:
                        text, _status = _run_transcribe(str(audio_file), transcribe_model, language)
                        txt_path.write_text((text or "").strip(), encoding="utf-8")
                        done += 1
                    except Exception:
                        failed += 1
                if play_completion_beep:
                    play_completion_beep()
                return f"Batch transcribe complete: {done} done, {skipped} skipped, {failed} failed"

            try:
                if run_heavy_job:
                    return run_heavy_job("library_manager_transcribe_batch", _run_impl, request=request)
                return _run_impl()
            except MemoryAdmissionError as exc:
                return f"⚠ Memory safety guard rejected request: {str(exc)}"

        def _on_dataset_auto_split(
            folder,
            lister_value,
            target_folder,
            clip_prefix,
            transcribe_model,
            language,
            split_min,
            split_max,
            silence_trim,
            discard_under,
            request: gr.Request,
            progress=gr.Progress(),
        ):
            def _run_impl():
                selected = (lister_value or {}).get("selected", [])
                if not selected:
                    msg = "Select at least one dataset file first"
                    return msg, msg, gr.update(), _usage_banner(request)
                if not folder or folder == "(Select Dataset)":
                    msg = "Select dataset folder first"
                    return msg, msg, gr.update(), _usage_banner(request)
                target = target_folder if target_folder and target_folder != "(Select Dataset)" else folder
                fixed_prefix = _clean_name(clip_prefix or "")

                dataset_dir = get_tenant_datasets_dir(request=request, strict=True) / folder
                total = len(selected)
                success = 0
                details: list[str] = []

                for idx, filename in enumerate(selected, start=1):
                    progress((idx - 1) / max(total, 1), desc=f"Auto-split {idx}/{total}: {filename}")
                    audio_path = dataset_dir / filename
                    if not audio_path.exists():
                        details.append(f"{filename}: file not found")
                        continue

                    file_prefix = fixed_prefix or (_clean_name(Path(filename).stem) or "clip")
                    status = _auto_split_to_dataset(
                        audio_path=str(audio_path),
                        dataset_folder=target,
                        clip_prefix=file_prefix,
                        transcribe_model=transcribe_model,
                        language=language,
                        split_min=split_min,
                        split_max=split_max,
                        silence_trim=silence_trim,
                        discard_under=discard_under,
                        request=request,
                        progress=progress,
                    )
                    details.append(f"{filename} [{file_prefix}]: {status}")
                    if status.startswith("Auto-split complete"):
                        success += 1

                failed = total - success
                if total == 1:
                    summary = details[0] if details else "Auto-split did not run"
                else:
                    mode_hint = (
                        f"fixed prefix '{fixed_prefix}'" if fixed_prefix else "per-file source filename prefixes"
                    )
                    summary = f"Batch auto-split complete: {success} succeeded, {failed} failed ({mode_hint})."
                    if details:
                        preview = " | ".join(details[:3])
                        if len(details) > 3:
                            preview += f" | (+{len(details) - 3} more)"
                        summary = f"{summary} {preview}"

                files_update = get_dataset_files(folder, request=request, strict=True)
                return summary, summary, files_update, _usage_banner(request)

            try:
                if run_heavy_job:
                    return run_heavy_job("library_manager_auto_split", _run_impl, request=request)
                return _run_impl()
            except MemoryAdmissionError as exc:
                msg = f"⚠ Memory safety guard rejected request: {str(exc)}"
                return msg, msg, gr.update(), _usage_banner(request)

        def _on_dataset_folder_change(folder, request: gr.Request):
            return (
                _refresh_dataset_files(folder, request=request),
                None,
                "",
                "Saved",
                gr.update(value=folder if folder and folder != "(Select Dataset)" else "(Select Dataset)"),
                gr.update(value=""),
                "",
            )

        def _on_refresh_samples(request: gr.Request):
            return (
                get_sample_choices(request=request, strict=True),
                *(_legacy_sample_warning_updates(request)),
            )

        def _on_clear_sample_cache_click(lister_value, request: gr.Request):
            selected = (lister_value or {}).get("selected") or []
            sample_stem = Path(selected[0]).stem if selected else ""
            return _clear_sample_cache(sample_stem, request=request), _usage_banner(request)

        def _prepare_processing_save_modal_data(action_kind, source_kind, source_dataset_folder, dataset_folder, suggested_name, request: gr.Request):
            destination = resolve_processing_save_destination(action_kind, source_kind)
            if destination == "sample":
                existing = _list_existing_sample_stems(request)
            else:
                folder = dataset_folder if dataset_folder and dataset_folder != "(Select Dataset)" else source_dataset_folder
                existing = _list_existing_dataset_stems(folder, request) if folder else []
            return json.dumps(existing), suggested_name or ""

        def _prepare_primary_modal_data(source_kind, source_dataset_folder, dataset_folder, suggested_name, request: gr.Request):
            return _prepare_processing_save_modal_data(
                "primary",
                source_kind,
                source_dataset_folder,
                dataset_folder,
                suggested_name,
                request,
            )

        def _prepare_secondary_modal_data(source_kind, source_dataset_folder, dataset_folder, suggested_name, request: gr.Request):
            return _prepare_processing_save_modal_data(
                "secondary",
                source_kind,
                source_dataset_folder,
                dataset_folder,
                suggested_name,
                request,
            )

        def _refresh_all(folder: str | None, request: gr.Request):
            folders = ["(Select Dataset)"] + get_dataset_folders(request=request, strict=True)
            current_folder = folder if folder in folders else "(Select Dataset)"
            files = get_dataset_files(current_folder, request=request, strict=True) if current_folder != "(Select Dataset)" else []
            legacy_warning_update, legacy_btn_update = _legacy_sample_warning_updates(request)
            preferred_engine, preferred_model = resolve_preferred_asr_engine_and_model(_user_config)
            _ = preferred_engine
            def _choice_scalar(choice):
                return str(coerce_choice_value(choice) or "")

            asr_choices = [_choice_scalar(c) for c in (components["proc_transcribe_model"].choices or [])]
            preferred_visible = preferred_model if preferred_model in asr_choices else (asr_choices[0] if asr_choices else preferred_model)
            return (
                _tenant_banner(request),
                _usage_banner(request),
                get_sample_choices(request=request, strict=True),
                gr.update(choices=folders, value=current_folder),
                files,
                gr.update(choices=folders),
                gr.update(choices=folders),
                gr.update(value=preferred_visible),
                gr.update(value=preferred_visible),
                gr.update(value=preferred_visible),
                legacy_warning_update,
                legacy_btn_update,
                "Ready",
            )

        # ----- Destructive actions via confirmation modal -----
        def _on_confirm_action(action, samples_lister, dataset_folder, dataset_lister, request: gr.Request):
            no_update = (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
            if not action:
                return no_update

            if _is_confirm(action, "lm_delete_samples_"):
                selected = (samples_lister or {}).get("selected", [])
                if not selected:
                    return (
                        "No samples selected",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        _usage_banner(request),
                        *_legacy_sample_warning_updates(request),
                        gr.update(),
                        gr.update(),
                    )
                paths = get_tenant_paths(request=request, strict=True)
                deleted = tenant_service.delete_samples(paths, selected)
                return (
                    f"Deleted {deleted} sample(s)",
                    get_sample_choices(request=request, strict=True),
                    gr.update(),
                    gr.update(),
                    _usage_banner(request),
                    *_legacy_sample_warning_updates(request),
                    gr.update(),
                    gr.update(),
                )

            if _is_confirm(action, "lm_delete_dataset_folder_"):
                if not dataset_folder or dataset_folder == "(Select Dataset)":
                    return (
                        "Select a dataset folder first",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        _usage_banner(request),
                        *_legacy_sample_warning_updates(request),
                        gr.update(),
                        gr.update(),
                    )
                paths = get_tenant_paths(request=request, strict=True)
                tenant_service.delete_dataset_folder(paths, dataset_folder)
                folder_dropdown, save_dropdown, split_dropdown = _dataset_folder_updates(
                    "(Select Dataset)",
                    request=request,
                )
                return (
                    f"Deleted dataset '{dataset_folder}'",
                    gr.update(),
                    folder_dropdown,
                    [],
                    _usage_banner(request),
                    *_legacy_sample_warning_updates(request),
                    save_dropdown,
                    split_dropdown,
                )

            if _is_confirm(action, "lm_delete_dataset_files_"):
                selected = (dataset_lister or {}).get("selected", [])
                if not selected or not dataset_folder or dataset_folder == "(Select Dataset)":
                    return (
                        "No dataset files selected",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        _usage_banner(request),
                        *_legacy_sample_warning_updates(request),
                        gr.update(),
                        gr.update(),
                    )
                paths = get_tenant_paths(request=request, strict=True)
                deleted = 0
                for filename in selected:
                    audio_path = paths.datasets_dir / dataset_folder / filename
                    txt_path = audio_path.with_suffix(".txt")
                    if audio_path.exists():
                        audio_path.unlink()
                        deleted += 1
                    if txt_path.exists():
                        txt_path.unlink()
                return (
                    f"Deleted {deleted} file(s)",
                    gr.update(),
                    gr.update(),
                    get_dataset_files(dataset_folder, request=request, strict=True),
                    _usage_banner(request),
                    *_legacy_sample_warning_updates(request),
                    gr.update(),
                    gr.update(),
                )

            return no_update

        def _handle_input_action(
            action,
            proc_audio,
            proc_transcript,
            proc_save_dataset_folder,
            proc_source_kind,
            proc_source_dataset_folder_state,
            proc_source_identifier,
            request: gr.Request,
        ):
            no_update = gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            if not action:
                return no_update

            for action_kind, prefix in (("primary", "lm_save_primary_"), ("secondary", "lm_save_secondary_")):
                raw_name, save_mode = _parse_modal_value(action, prefix)
                if raw_name is None:
                    continue
                mode = save_mode or "new"
                destination = resolve_processing_save_destination(action_kind, proc_source_kind)

                if destination == "sample":
                    status = _save_sample_from_processing(
                        proc_audio,
                        proc_transcript,
                        raw_name,
                        mode,
                        request=request,
                    )
                    full_status = (
                        f"{status} | Source: {proc_source_identifier or proc_source_kind} | "
                        f"Destination: Samples | Mode: {'Replace existing' if mode == 'replace' else 'Save as new'}"
                    )
                    return (
                        full_status,
                        full_status,
                        get_sample_choices(request=request, strict=True),
                        gr.update(),
                        _usage_banner(request),
                    )

                dataset_folder = proc_save_dataset_folder
                if not dataset_folder or dataset_folder == "(Select Dataset)":
                    dataset_folder = proc_source_dataset_folder_state
                status = _save_dataset_clip_from_processing(
                    proc_audio,
                    proc_transcript,
                    dataset_folder,
                    raw_name,
                    mode,
                    request=request,
                )
                files_update = (
                    get_dataset_files(dataset_folder, request=request, strict=True)
                    if dataset_folder and dataset_folder != "(Select Dataset)"
                    else gr.update()
                )
                full_status = (
                    f"{status} | Source: {proc_source_identifier or proc_source_kind} | "
                    f"Destination: Dataset ({dataset_folder or '(Select Dataset)'}) | "
                    f"Mode: {'Replace existing' if mode == 'replace' else 'Save as new'}"
                )
                return (
                    full_status,
                    full_status,
                    gr.update(),
                    files_update,
                    _usage_banner(request),
                )

            return no_update

        def _on_audio_route_processing(trigger_value):
            payload = audio_route_parse_payload(trigger_value)
            no_update = tuple(gr.update() for _ in range(14))
            if not payload or payload.get("target_id") != "library_manager.processing":
                return (*no_update, gr.update())

            audio_path = str(payload.get("audio_path") or "").strip()
            source_label = payload.get("source_label", "external")
            if not audio_path:
                msg = "Incoming audio route was empty."
                return (*no_update[:-1], msg, msg)
            if not Path(audio_path).exists():
                msg = f"Incoming routed audio was not found: {audio_path}"
                return (*no_update[:-1], msg, msg)

            context = build_routed_processing_context(audio_path)
            default_name = sanitize_filename(Path(audio_path).stem, keep_extension=False) or "routed_audio"
            status = (
                f"Routed audio from {source_label} opened in Processing Studio. "
                "Pipeline starts from the routed source."
            )
            proc_updates = _prepare_processing_context(
                context=context,
                transcript="",
                default_name=default_name,
                dataset_target="(Select Dataset)",
                primary_label="Save to Samples (Primary)",
                secondary_label="Save to Dataset",
                status=status,
            )
            return (*proc_updates, status)

        # ----- Initial refresh -----
        components["library_tab"].select(
            _refresh_all,
            inputs=[components["dataset_folder_dropdown"]],
            outputs=[
                components["library_tenant_md"],
                components["library_usage_md"],
                components["samples_lister"],
                components["dataset_folder_dropdown"],
                components["dataset_lister"],
                components["proc_save_dataset_folder"],
                components["dataset_split_target_folder"],
                components["proc_transcribe_model"],
                components["dataset_transcribe_model"],
                components["dataset_split_transcribe_model"],
                components["legacy_sample_warning"],
                components["convert_legacy_samples_btn"],
                components["library_status"],
            ],
        )

        # ----- Samples tab -----
        components["samples_lister"].change(
            _on_sample_select,
            inputs=[components["samples_lister"]],
            outputs=[
                components["sample_preview"],
                components["sample_transcript"],
                components["sample_info"],
                components["sample_transcript_status"],
            ],
        )

        components["sample_transcript"].change(
            _autosave_sample_transcript,
            inputs=[components["samples_lister"], components["sample_transcript"]],
            outputs=[components["library_status"], components["sample_transcript_status"]],
        )

        components["refresh_samples_btn"].click(
            _on_refresh_samples,
            outputs=[components["samples_lister"], components["legacy_sample_warning"], components["convert_legacy_samples_btn"]],
        )

        components["upload_samples_btn"].click(
            _on_upload_samples,
            inputs=[components["samples_upload"]],
            outputs=[
                components["library_status"],
                components["samples_lister"],
                components["library_usage_md"],
                components["legacy_sample_warning"],
                components["convert_legacy_samples_btn"],
            ],
        )

        components["convert_legacy_samples_btn"].click(
            _on_convert_legacy_samples,
            outputs=[
                components["library_status"],
                components["samples_lister"],
                components["legacy_sample_warning"],
                components["convert_legacy_samples_btn"],
                components["library_usage_md"],
            ],
        )

        components["clear_sample_cache_btn"].click(
            _on_clear_sample_cache_click,
            inputs=[components["samples_lister"]],
            outputs=[components["library_status"], components["library_usage_md"]],
        )

        components["open_sample_processing_btn"].click(
            _on_open_sample_processing,
            inputs=[components["samples_lister"]],
            outputs=[
                components["library_sections"],
                components["proc_audio_editor"],
                components["proc_transcript"],
                components["lm_default_name"],
                components["proc_original_audio_path"],
                components["proc_source_kind"],
                components["proc_source_dataset_folder_state"],
                components["proc_source_type"],
                components["proc_source_identifier"],
                components["proc_original_file"],
                components["proc_save_dataset_folder"],
                components["proc_save_primary_btn"],
                components["proc_save_secondary_btn"],
                components["proc_status"],
                components["sample_action_status"],
                components["library_status"],
            ],
        )

        delete_samples_js = show_confirmation_modal_js(
            title="Delete Sample(s)?",
            message="This permanently deletes selected sample audio, metadata, and cache files.",
            confirm_button_text="Delete",
            context="lm_delete_samples_",
        )
        components["delete_samples_btn"].click(fn=None, js=delete_samples_js)

        # ----- Datasets tab -----
        components["dataset_folder_dropdown"].change(
            _on_dataset_folder_change,
            inputs=[components["dataset_folder_dropdown"]],
            outputs=[
                components["dataset_lister"],
                components["dataset_preview"],
                components["dataset_transcript"],
                components["dataset_transcript_status"],
                components["dataset_split_target_folder"],
                components["dataset_split_prefix"],
                components["dataset_split_prefix_auto_state"],
            ],
        )

        components["refresh_dataset_btn"].click(
            _refresh_dataset_files,
            inputs=[components["dataset_folder_dropdown"]],
            outputs=[components["dataset_lister"]],
        )

        components["dataset_lister"].change(
            _on_dataset_select,
            inputs=[
                components["dataset_folder_dropdown"],
                components["dataset_lister"],
                components["dataset_split_prefix"],
                components["dataset_split_prefix_auto_state"],
            ],
            outputs=[
                components["dataset_preview"],
                components["dataset_transcript"],
                components["dataset_transcript_status"],
                components["dataset_split_prefix"],
                components["dataset_split_prefix_auto_state"],
            ],
        )

        components["dataset_transcript"].change(
            _autosave_dataset_transcript,
            inputs=[components["dataset_folder_dropdown"], components["dataset_lister"], components["dataset_transcript"]],
            outputs=[components["library_status"], components["dataset_transcript_status"]],
        )

        components["upload_dataset_btn"].click(
            _on_upload_dataset,
            inputs=[components["dataset_folder_dropdown"], components["dataset_upload"]],
            outputs=[components["library_status"], components["dataset_lister"], components["library_usage_md"]],
        )

        components["create_dataset_btn"].click(
            _on_create_dataset,
            inputs=[components["new_dataset_name"]],
            outputs=[
                components["library_status"],
                components["dataset_folder_dropdown"],
                components["dataset_lister"],
                components["proc_save_dataset_folder"],
                components["dataset_split_target_folder"],
                components["library_usage_md"],
            ],
        )

        components["batch_transcribe_btn"].click(
            _on_batch_transcribe,
            inputs=[
                components["dataset_folder_dropdown"],
                components["batch_replace_existing"],
                components["dataset_language"],
                components["dataset_transcribe_model"],
            ],
            outputs=[components["library_status"]],
        )

        components["dataset_auto_split_btn"].click(
            _on_dataset_auto_split,
            inputs=[
                components["dataset_folder_dropdown"],
                components["dataset_lister"],
                components["dataset_split_target_folder"],
                components["dataset_split_prefix"],
                components["dataset_split_transcribe_model"],
                components["dataset_split_language"],
                components["split_min"],
                components["split_max"],
                components["silence_trim"],
                components["discard_under"],
            ],
            outputs=[
                components["library_status"],
                components["dataset_action_status"],
                components["dataset_lister"],
                components["library_usage_md"],
            ],
        )

        components["open_dataset_processing_btn"].click(
            _on_open_dataset_processing,
            inputs=[components["dataset_folder_dropdown"], components["dataset_lister"]],
            outputs=[
                components["library_sections"],
                components["proc_audio_editor"],
                components["proc_transcript"],
                components["lm_default_name"],
                components["proc_original_audio_path"],
                components["proc_source_kind"],
                components["proc_source_dataset_folder_state"],
                components["proc_source_type"],
                components["proc_source_identifier"],
                components["proc_original_file"],
                components["proc_save_dataset_folder"],
                components["proc_save_primary_btn"],
                components["proc_save_secondary_btn"],
                components["proc_status"],
                components["dataset_action_status"],
                components["library_status"],
            ],
        )

        delete_dataset_folder_js = show_confirmation_modal_js(
            title="Delete Dataset Folder?",
            message="This permanently deletes the selected dataset folder and all files inside.",
            confirm_button_text="Delete",
            context="lm_delete_dataset_folder_",
        )
        components["delete_dataset_btn"].click(fn=None, js=delete_dataset_folder_js)

        delete_dataset_files_js = show_confirmation_modal_js(
            title="Delete Dataset File(s)?",
            message="This permanently deletes selected dataset audio files and transcripts.",
            confirm_button_text="Delete",
            context="lm_delete_dataset_files_",
        )
        components["delete_dataset_files_btn"].click(fn=None, js=delete_dataset_files_js)

        # ----- Processing Studio -----
        components["proc_apply_pipeline_btn"].click(
            _on_proc_apply_pipeline,
            inputs=[
                components["proc_original_audio_path"],
                components["proc_enable_denoise"],
                components["proc_enable_normalize"],
                components["proc_enable_mono"],
            ],
            outputs=[components["proc_audio_editor"], components["proc_status"]],
        )

        components["proc_reset_btn"].click(
            _on_proc_reset,
            inputs=[components["proc_original_audio_path"]],
            outputs=[components["proc_audio_editor"], components["proc_status"]],
        )

        components["proc_transcribe_btn"].click(
            _on_proc_transcribe_click,
            inputs=[components["proc_audio_editor"], components["proc_transcribe_model"], components["proc_language"]],
            outputs=[components["proc_transcript"], components["proc_status"]],
        )

        components["proc_expected_speakers"].change(
            lambda value: (
                save_preference("speaker_separation_expected_speakers", str(value)),
                _speaker_separation_note(value),
            )[1],
            inputs=[components["proc_expected_speakers"]],
            outputs=[components["proc_speaker_separation_note"]],
        )

        components["proc_separate_speakers_btn"].click(
            _on_proc_separate_speakers_click,
            inputs=[
                components["proc_audio_editor"],
                components["proc_expected_speakers"],
                components["lm_default_name"],
                components["proc_source_identifier"],
            ],
            outputs=[
                components["library_sections"],
                components["samples_lister"],
                components["library_usage_md"],
                components["proc_status"],
                components["library_status"],
                components["sample_action_status"],
            ],
        )

        def _on_asr_change(model):
            save_preference("transcribe_model", model)
            visible = any(k in model for k in ("Qwen3 ASR", "Whisper"))
            return gr.update(visible=visible)

        components["proc_transcribe_model"].change(
            _on_asr_change,
            inputs=[components["proc_transcribe_model"]],
            outputs=[components["proc_language"]],
        )
        components["dataset_transcribe_model"].change(
            _on_asr_change,
            inputs=[components["dataset_transcribe_model"]],
            outputs=[components["dataset_language"]],
        )
        components["dataset_split_transcribe_model"].change(
            _on_asr_change,
            inputs=[components["dataset_split_transcribe_model"]],
            outputs=[components["dataset_split_language"]],
        )

        components["proc_language"].change(
            lambda x: save_preference("whisper_language", x),
            inputs=[components["proc_language"]],
            outputs=[],
        )
        components["dataset_language"].change(
            lambda x: save_preference("whisper_language", x),
            inputs=[components["dataset_language"]],
            outputs=[],
        )
        components["dataset_split_language"].change(
            lambda x: save_preference("whisper_language", x),
            inputs=[components["dataset_split_language"]],
            outputs=[],
        )

        components["split_min"].change(lambda x: save_preference("split_min", x), inputs=[components["split_min"]], outputs=[])
        components["split_max"].change(lambda x: save_preference("split_max", x), inputs=[components["split_max"]], outputs=[])
        components["silence_trim"].change(lambda x: save_preference("silence_trim", x), inputs=[components["silence_trim"]], outputs=[])
        components["discard_under"].change(lambda x: save_preference("discard_under", x), inputs=[components["discard_under"]], outputs=[])

        save_primary_modal_js = show_input_modal_js(
            title="Save Output",
            message="Enter target name:",
            placeholder="e.g. narrator_ref",
            context="lm_save_primary_",
            submit_button_text="Save",
            show_save_mode=True,
            default_save_mode="new",
        )
        save_secondary_modal_js = show_input_modal_js(
            title="Save Output",
            message="Enter target name:",
            placeholder="e.g. clip_001",
            context="lm_save_secondary_",
            submit_button_text="Save",
            show_save_mode=True,
            default_save_mode="new",
        )

        save_primary_js = f"""
        (existingFilesJson, suggestedName) => {{
            try {{
                window.inputModalExistingFiles = JSON.parse(existingFilesJson || '[]');
            }} catch (e) {{
                window.inputModalExistingFiles = [];
            }}
            const openModal = {save_primary_modal_js};
            openModal(suggestedName || '');
        }}
        """
        save_secondary_js = f"""
        (existingFilesJson, suggestedName) => {{
            try {{
                window.inputModalExistingFiles = JSON.parse(existingFilesJson || '[]');
            }} catch (e) {{
                window.inputModalExistingFiles = [];
            }}
            const openModal = {save_secondary_modal_js};
            openModal(suggestedName || '');
        }}
        """

        components["proc_save_primary_btn"].click(
            _prepare_primary_modal_data,
            inputs=[
                components["proc_source_kind"],
                components["proc_source_dataset_folder_state"],
                components["proc_save_dataset_folder"],
                components["lm_default_name"],
            ],
            outputs=[components["lm_existing_names_json"], components["lm_default_name"]],
        ).then(
            fn=None,
            inputs=[components["lm_existing_names_json"], components["lm_default_name"]],
            js=save_primary_js,
        )

        components["proc_save_secondary_btn"].click(
            _prepare_secondary_modal_data,
            inputs=[
                components["proc_source_kind"],
                components["proc_source_dataset_folder_state"],
                components["proc_save_dataset_folder"],
                components["lm_default_name"],
            ],
            outputs=[components["lm_existing_names_json"], components["lm_default_name"]],
        ).then(
            fn=None,
            inputs=[components["lm_existing_names_json"], components["lm_default_name"]],
            js=save_secondary_js,
        )

        input_trigger.change(
            _handle_input_action,
            inputs=[
                input_trigger,
                components["proc_audio_editor"],
                components["proc_transcript"],
                components["proc_save_dataset_folder"],
                components["proc_source_kind"],
                components["proc_source_dataset_folder_state"],
                components["proc_source_identifier"],
            ],
            outputs=[
                components["library_status"],
                components["proc_status"],
                components["samples_lister"],
                components["dataset_lister"],
                components["library_usage_md"],
            ],
        )

        if audio_route_trigger is not None:
            audio_route_trigger.change(
                _on_audio_route_processing,
                inputs=[audio_route_trigger],
                outputs=[
                    components["library_sections"],
                    components["proc_audio_editor"],
                    components["proc_transcript"],
                    components["lm_default_name"],
                    components["proc_original_audio_path"],
                    components["proc_source_kind"],
                    components["proc_source_dataset_folder_state"],
                    components["proc_source_type"],
                    components["proc_source_identifier"],
                    components["proc_original_file"],
                    components["proc_save_dataset_folder"],
                    components["proc_save_primary_btn"],
                    components["proc_save_secondary_btn"],
                    components["proc_status"],
                    components["library_status"],
                ],
            )

        confirm_trigger.change(
            _on_confirm_action,
            inputs=[confirm_trigger, components["samples_lister"], components["dataset_folder_dropdown"], components["dataset_lister"]],
            outputs=[
                components["library_status"],
                components["samples_lister"],
                components["dataset_folder_dropdown"],
                components["dataset_lister"],
                components["library_usage_md"],
                components["legacy_sample_warning"],
                components["convert_legacy_samples_btn"],
                components["proc_save_dataset_folder"],
                components["dataset_split_target_folder"],
            ],
        )


get_tool_class = lambda: LibraryManagerTool
