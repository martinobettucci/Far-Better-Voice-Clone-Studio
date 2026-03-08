"""
Settings Tab

Configure global application settings.

Standalone testing:
    python -m modules.core_components.tools.settings
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import gradio as gr
from pathlib import Path
import traceback

import modules.core_components.prompt_hub as prompt_hub
from modules.core_components.ai_models import get_source_separation_manager
from modules.core_components.ai_models import get_speaker_separation_manager
from modules.core_components.ai_models.source_separation_manager import DEFAULT_SOURCE_SEPARATION_MODEL_FILENAMES
from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.constants import (
    ASR_ENGINES,
    TTS_ENGINES,
    coerce_choice_value,
    get_asr_models_for_engine,
    get_tts_models_for_engine,
    resolve_preferred_asr_engine_and_model,
    resolve_preferred_tts_engine_and_model,
)

# Tools that can be toggled (everything except Settings)
# Format: (config_key, display_label)
TOGGLEABLE_TOOLS = [
    ("Voice Clone", "Voice Clone"),
    ("Voice Changer", "Voice Changer"),
    ("Singing Enhancements", "Singing Enhancements"),
    ("Voice Presets", "Voice Presets"),
    ("Conversation", "Conversation"),
    ("Voice Design", "Voice Design"),
    ("Library Manager", "Library Manager"),
    ("Train Model", "Train Model"),
    ("Sound Effects", "Sound Effects"),
    ("Prompt Manager", "Prompt Manager"),
    ("Output History", "Output History"),
]


class SettingsTool(Tool):
    """Settings tool implementation."""

    config = ToolConfig(
        name="Settings",
        module_name="tool_settings",
        description="Application settings and preferences",
        enabled=True,
        category="utility",
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Settings tool UI."""
        components = {}

        _user_config = shared_state.get("_user_config", {})
        project_root = Path(__file__).parent.parent.parent.parent
        configured_models_dir = Path(_user_config.get("models_folder", "models"))
        if not configured_models_dir.is_absolute():
            configured_models_dir = project_root / configured_models_dir

        source_separation_manager = get_source_separation_manager(
            user_config=_user_config,
            models_dir=configured_models_dir,
        )
        try:
            source_sep_models = source_separation_manager.list_models(refresh=False)
        except Exception:
            source_sep_models = []
        source_sep_download_choices = []
        source_sep_download_map = {}
        grouped_source_sep_models = {}
        for model in source_sep_models:
            grouped_source_sep_models.setdefault(model.architecture or "Other", []).append(model)
        for architecture in sorted(grouped_source_sep_models):
            source_sep_download_choices.append(f"--- Source Separation ({architecture}) ---")
            for model in grouped_source_sep_models[architecture]:
                label = f"Source Separation - {model.display_name} [{model.model_filename}]"
                source_sep_download_choices.append(label)
                source_sep_download_map[label] = f"source-separation://{model.model_filename}"

        speaker_separation_manager = get_speaker_separation_manager(
            user_config=_user_config,
            models_dir=configured_models_dir,
        )
        speaker_sep_models = speaker_separation_manager.list_models()
        speaker_sep_download_choices = ["--- Speaker Separation ---"]
        speaker_sep_download_map = {}
        for model in speaker_sep_models:
            speaker_sep_download_choices.append(model.display_name)
            speaker_sep_download_map[model.display_name] = f"speaker-separation://{model.expected_speakers}"

        with gr.TabItem("⚙️"):
            gr.Markdown("# ⚙️ Settings")
            gr.Markdown("Configure global application settings")

            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Displayed Tools")
                        tool_settings = _user_config.get("enabled_tools", {})
                        with gr.Column():
                            for key, label in TOGGLEABLE_TOOLS:
                                is_enabled = tool_settings.get(key, True)
                                components[f"tool_toggle_{key}"] = gr.Checkbox(
                                    label=label,
                                    value=is_enabled,
                                    interactive=True,
                                )

                    with gr.Column():
                        gr.Markdown("### Model Optimizations")
                        components["settings_low_cpu_mem"] = gr.Checkbox(
                            label="Low CPU Memory Usage (Slower loading time)",
                            value=_user_config.get("low_cpu_mem_usage", False),
                            info="Reduces CPU RAM usage when loading models.",
                        )
                        components["settings_attention_mechanism"] = gr.Dropdown(
                            label="Choose Attention Mechanism",
                            choices=["auto", "flash_attention_2", "sdpa", "eager"],
                            value=_user_config.get("attention_mechanism", "auto"),
                            info="Auto = fastest available.",
                        )
                        components["settings_deterministic_mode"] = gr.Checkbox(
                            label="Deterministic Mode (Best Reproducibility)",
                            value=_user_config.get("deterministic_mode", False),
                            info="Uses deterministic backend/seed behavior. May be slower.",
                        )
                        components["settings_skip_engine_check"] = gr.Checkbox(
                            label="Skip Engine Check at Startup",
                            value=_user_config.get("skip_engine_check", False),
                            info="Assumes all engines are available. Faster launch. (Restart required)",
                        )

                        gr.Markdown("### Network")
                        components["settings_listen_on_network"] = gr.Checkbox(
                            label="Listen on Network",
                            value=_user_config.get("listen_on_network", False),
                            info="Allow other devices on your local network to connect. (Restart required)",
                        )

                        gr.Markdown("### Multi-Tenant")
                        components["tenant_header_name"] = gr.Textbox(
                            label="Tenant Header Name",
                            value=_user_config.get("tenant_header_name", "X-Tenant-Id"),
                            info="Required header set by reverse proxy/auth gateway.",
                        )
                        components["tenant_file_limit_mb"] = gr.Number(
                            label="Per-file Upload Limit (MB)",
                            value=int(_user_config.get("tenant_file_limit_mb", 200)),
                            precision=0,
                        )
                        components["tenant_media_quota_gb"] = gr.Number(
                            label="Per-tenant Media Quota (GB)",
                            value=int(_user_config.get("tenant_media_quota_gb", 5)),
                            precision=0,
                        )

                    with gr.Column():
                        gr.Markdown("### Model Downloading")
                        components["settings_offline_mode"] = gr.Checkbox(
                            label="Offline Mode",
                            value=_user_config.get("offline_mode", False),
                            info="When enabled, only uses models found in models folder",
                        )

                        model_download_choices = [
                            "--- Qwen3-TTS Base ---",
                            "Qwen3-TTS-12Hz-0.6B-Base",
                            "Qwen3-TTS-12Hz-1.7B-Base",
                            "--- Qwen3-TTS CustomVoice ---",
                            "Qwen3-TTS-12Hz-0.6B-CustomVoice",
                            "Qwen3-TTS-12Hz-1.7B-CustomVoice",
                            "--- Qwen3-TTS VoiceDesign ---",
                            "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                            "--- VibeVoice TTS ---",
                            "VibeVoice-1.5B",
                            "VibeVoice-Large (4-bit)",
                            "VibeVoice-Large",
                            "--- VibeVoice ASR ---",
                            "VibeVoice-ASR",
                            "--- Qwen3 ASR ---",
                            "Qwen3-ASR-0.6B",
                            "Qwen3-ASR-1.7B",
                            "Qwen3-ForcedAligner-0.6B",
                            "--- Tokenizer Dependencies ---",
                            "Qwen3-TTS-Tokenizer-12Hz",
                            "Qwen2.5-1.5B (VibeVoice Tokenizer)",
                            "--- Whisper ASR ---",
                            "Whisper-Medium",
                            "Whisper-Large",
                            "--- LuxTTS ---",
                            "LuxTTS",
                            "--- Chatterbox ---",
                            "Chatterbox",
                            "--- Source Separation ---",
                            "Source Separation - Recommended Defaults",
                        ]
                        model_download_choices.extend(source_sep_download_choices)
                        model_download_choices.extend(speaker_sep_download_choices)
                        model_download_choices.extend(
                            [
                                "--- Sound Effects (MMAudio) ---",
                                "MMAudio - Medium (44kHz)",
                                "MMAudio - Large v2 (44kHz)",
                                "--- Sound Effects Dependencies ---",
                                "MMAudio-CLIP-DFN5B",
                                "MMAudio-BigVGAN-v2-44k",
                            ]
                        )

                        components["model_select"] = gr.Dropdown(
                            label="Select Model to Download to Models Folder",
                            info=(
                                "Needed for strict offline mode. "
                                "Download all required dependencies here. "
                                "Source-separation and speaker-separation models are listed here too, "
                                "and 'Download them all' includes them."
                            ),
                            choices=model_download_choices,
                            value="Qwen3-TTS-12Hz-0.6B-Base",
                        )
                        components["ALL_MODEL_CHOICES"] = model_download_choices
                        components["download_all_models"] = gr.Checkbox(
                            label="Download them all",
                            value=False,
                            info=(
                                "When enabled, the selected model is ignored and all listed models are downloaded."
                            ),
                        )
                        components["download_btn"] = gr.Button("Download Model", scale=1)

                        gr.Markdown(
                            "Source-separation and speaker-separation models auto-download on first use unless "
                            "Offline Mode is enabled."
                        )

                        # Mapping from display names to HuggingFace model IDs
                        model_id_map = {
                            "Qwen3-TTS-12Hz-0.6B-Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                            "Qwen3-TTS-12Hz-1.7B-Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                            "Qwen3-TTS-12Hz-0.6B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                            "Qwen3-TTS-12Hz-1.7B-CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                            "Qwen3-TTS-12Hz-1.7B-VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                            "VibeVoice-1.5B": "FranckyB/VibeVoice-1.5B",
                            "VibeVoice-Large (4-bit)": "FranckyB/VibeVoice-Large-4bit",
                            "VibeVoice-Large": "FranckyB/VibeVoice-Large",
                            "VibeVoice-ASR": "microsoft/VibeVoice-ASR",
                            "Qwen3-ASR-0.6B": "Qwen/Qwen3-ASR-0.6B",
                            "Qwen3-ASR-1.7B": "Qwen/Qwen3-ASR-1.7B",
                            "Qwen3-ForcedAligner-0.6B": "Qwen/Qwen3-ForcedAligner-0.6B",
                            "Qwen3-TTS-Tokenizer-12Hz": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
                            "Qwen2.5-1.5B (VibeVoice Tokenizer)": "Qwen/Qwen2.5-1.5B",
                            "Whisper-Medium": "whisper://medium",
                            "Whisper-Large": "whisper://large",
                            "LuxTTS": "YatharthS/LuxTTS",
                            "Chatterbox": "ResembleAI/chatterbox",
                            "Source Separation - Recommended Defaults": "source-separation://recommended",
                            "MMAudio - Medium (44kHz)": "mmaudio://Medium (44kHz)",
                            "MMAudio - Large v2 (44kHz)": "mmaudio://Large v2 (44kHz)",
                            "MMAudio-CLIP-DFN5B": "apple/DFN5B-CLIP-ViT-H-14-384",
                            "MMAudio-BigVGAN-v2-44k": "nvidia/bigvgan_v2_44khz_128band_512x",
                        }
                        model_id_map.update(source_sep_download_map)
                        model_id_map.update(speaker_sep_download_map)
                        components["MODEL_ID_MAP"] = model_id_map

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Available Voice Clone Engines")
                        engine_settings = _user_config.get("enabled_engines", {})
                        for engine_key, engine_info in TTS_ENGINES.items():
                            is_enabled = engine_settings.get(
                                engine_key, engine_info.get("default_enabled", True)
                            )
                            components[f"engine_toggle_{engine_key}"] = gr.Checkbox(
                                label=engine_info["label"],
                                value=is_enabled,
                                interactive=True,
                            )

                        preferred_tts_engine, preferred_tts_model = resolve_preferred_tts_engine_and_model(_user_config)
                        preferred_tts_models = get_tts_models_for_engine(preferred_tts_engine)
                        components["preferred_voice_clone_engine"] = gr.Dropdown(
                            label="Preferred Voice Clone Engine",
                            choices=list(TTS_ENGINES.keys()),
                            value=preferred_tts_engine if preferred_tts_engine in TTS_ENGINES else list(TTS_ENGINES.keys())[0],
                            interactive=True,
                        )
                        components["preferred_voice_clone_model"] = gr.Dropdown(
                            label="Preferred Voice Clone Model",
                            choices=preferred_tts_models,
                            value=preferred_tts_model if preferred_tts_model in preferred_tts_models else (preferred_tts_models[-1] if preferred_tts_models else None),
                            interactive=True,
                        )

                        gr.Markdown("### Audio Notifications")
                        components["settings_audio_notifications"] = gr.Checkbox(
                            label="Enable Audio Notifications",
                            value=_user_config.get("browser_notifications", True),
                            info="Play sound when audio generation completes",
                        )

                    with gr.Column():
                        gr.Markdown("### Available Transcription Engines")
                        asr_settings = _user_config.get("enabled_asr_engines", {})
                        with gr.Column():
                            for engine_key, engine_info in ASR_ENGINES.items():
                                is_enabled = asr_settings.get(
                                    engine_key, engine_info.get("default_enabled", True)
                                )
                                components[f"asr_toggle_{engine_key}"] = gr.Checkbox(
                                    label=engine_info["label"],
                                    value=is_enabled,
                                    interactive=True,
                                )

                            preferred_asr_engine, preferred_asr_model = resolve_preferred_asr_engine_and_model(_user_config)
                            preferred_asr_models = get_asr_models_for_engine(preferred_asr_engine)
                            components["preferred_asr_engine"] = gr.Dropdown(
                                label="Preferred Transcription Engine",
                                choices=list(ASR_ENGINES.keys()),
                                value=preferred_asr_engine if preferred_asr_engine in ASR_ENGINES else list(ASR_ENGINES.keys())[0],
                                interactive=True,
                            )
                            components["preferred_asr_model"] = gr.Dropdown(
                                label="Preferred Transcription Model",
                                choices=preferred_asr_models,
                                value=preferred_asr_model if preferred_asr_model in preferred_asr_models else (preferred_asr_models[-1] if preferred_asr_models else None),
                                interactive=True,
                            )

                            components["bypass_split_limit"] = gr.Checkbox(
                                label="Allow Qwen3 extended audio splitting (beyond 5 min)",
                                value=_user_config.get("bypass_split_limit", False),
                                info="Qwen3 ASR alignment may be less accurate and \ndemand tons of VRAM for very long audio. Enable with caution.",
                                interactive=True,
                            )

                    with gr.Column():
                        gr.Markdown("### LLM Endpoint (Prompt Manager)")
                        components["settings_llm_api_key"] = gr.Textbox(
                            label="OpenAI-Compatible API Key",
                            value=_user_config.get("llm_api_key", ""),
                            type="password",
                            info="Used for non-local endpoints. Stored in config.json as plain text.",
                            placeholder="e.g. sk-...",
                        )
                        components["settings_llm_endpoint_url"] = gr.Textbox(
                            label="Default Endpoint URL",
                            value=_user_config.get("llm_endpoint_url", "https://api.openai.com/v1"),
                            info="Base URL for OpenAI-compatible chat completion endpoints.",
                            placeholder="e.g. https://api.openai.com/v1",
                        )
                        components["settings_llm_ollama_url"] = gr.Textbox(
                            label="Local Ollama URL",
                            value=_user_config.get("llm_ollama_url", "http://127.0.0.1:11434/v1"),
                            info="Used when 'Use local Ollama' is enabled in Prompt Manager.",
                            placeholder="e.g. http://127.0.0.1:11434/v1",
                        )
                        components["settings_prompt_assistant_default_language"] = gr.Dropdown(
                            label="Prompt Assistant Default Language",
                            choices=prompt_hub.PROMPT_ASSISTANT_LANGUAGE_CHOICES[1:],
                            value=prompt_hub.get_prompt_assistant_default_language(_user_config),
                            info="Default language used by Prompt Assistant tabs. Can be overridden per tab.",
                            interactive=True,
                        )

                gr.Markdown("Configure where files are stored. Changes apply after clicking **Apply Changes**.")
                default_folders = {
                    "samples": "samples",
                    "output": "output",
                    "datasets": "datasets",
                    "temp": "temp",
                    "models": "models",
                }
                components["default_folders"] = default_folders

                with gr.Row():
                    with gr.Column():
                        components["settings_samples_folder"] = gr.Textbox(
                            label="Voice Samples Folder",
                            value=_user_config.get("samples_folder", default_folders["samples"]),
                            info="Folder for voice sample files (.wav + .json)",
                        )
                        components["reset_samples_btn"] = gr.Button("Reset", size="sm")

                    with gr.Column():
                        components["settings_output_folder"] = gr.Textbox(
                            label="Output Folder",
                            value=_user_config.get("output_folder", default_folders["output"]),
                            info="Folder for generated audio files",
                        )
                        components["reset_output_btn"] = gr.Button("Reset", size="sm")

                    with gr.Column():
                        components["settings_datasets_folder"] = gr.Textbox(
                            label="Datasets Folder",
                            value=_user_config.get("datasets_folder", default_folders["datasets"]),
                            info="Folder for training/finetuning datasets",
                        )
                        components["reset_datasets_btn"] = gr.Button("Reset", size="sm")

                with gr.Row():
                    with gr.Column():
                        components["settings_models_folder"] = gr.Textbox(
                            label="Downloaded Models Folder",
                            value=_user_config.get("models_folder", default_folders["models"]),
                            info="Folder for downloaded model files (Qwen, VibeVoice, Chatterbox)",
                        )
                        components["reset_models_btn"] = gr.Button("Reset", size="sm")

                    with gr.Column():
                        components["settings_trained_models_folder"] = gr.Textbox(
                            label="Trained Models Folder",
                            value=_user_config.get("trained_models_folder", default_folders["models"]),
                            info="Folder for your custom trained models",
                        )
                        components["reset_trained_models_btn"] = gr.Button("Reset", size="sm")

                    with gr.Column():
                        gr.Markdown("")

            with gr.Column():
                components["apply_folders_btn"] = gr.Button("Apply Changes", variant="primary", size="lg")
                components["settings_status"] = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=10,
                )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Settings tab events."""

        _user_config = shared_state.get("_user_config", {})
        save_preference = shared_state.get("save_preference")
        download_model_from_huggingface = shared_state.get("download_model_from_huggingface")
        project_root = Path(__file__).parent.parent.parent.parent

        def _get_source_separation_manager():
            models_dir = Path(_user_config.get("models_folder", "models"))
            if not models_dir.is_absolute():
                models_dir = project_root / models_dir
            return get_source_separation_manager(user_config=_user_config, models_dir=models_dir)

        def _get_speaker_separation_manager():
            models_dir = Path(_user_config.get("models_folder", "models"))
            if not models_dir.is_absolute():
                models_dir = project_root / models_dir
            return get_speaker_separation_manager(user_config=_user_config, models_dir=models_dir)

        # Lazy import to avoid circular dependency
        from modules.core_components.tools import save_config

        components["settings_low_cpu_mem"].change(
            lambda x: save_preference("low_cpu_mem_usage", x),
            inputs=[components["settings_low_cpu_mem"]],
            outputs=[],
        )
        components["settings_attention_mechanism"].change(
            lambda x: save_preference("attention_mechanism", x),
            inputs=[components["settings_attention_mechanism"]],
            outputs=[],
        )

        def apply_deterministic_mode(enabled):
            from modules.core_components.ai_models.model_utils import configure_runtime_reproducibility

            configure_runtime_reproducibility(bool(enabled))
            save_preference("deterministic_mode", bool(enabled))

        components["settings_deterministic_mode"].change(
            apply_deterministic_mode,
            inputs=[components["settings_deterministic_mode"]],
            outputs=[],
        )

        components["settings_offline_mode"].change(
            lambda x: save_preference("offline_mode", x),
            inputs=[components["settings_offline_mode"]],
            outputs=[],
        )
        components["settings_skip_engine_check"].change(
            lambda x: save_preference("skip_engine_check", x),
            inputs=[components["settings_skip_engine_check"]],
            outputs=[],
        )
        components["settings_audio_notifications"].change(
            lambda x: save_preference("browser_notifications", x),
            inputs=[components["settings_audio_notifications"]],
            outputs=[],
        )
        components["settings_listen_on_network"].change(
            lambda x: save_preference("listen_on_network", x),
            inputs=[components["settings_listen_on_network"]],
            outputs=[],
        )

        def save_tenant_header(value):
            value = (value or "").strip() or "X-Tenant-Id"
            save_preference("tenant_header_name", value)

        def save_file_limit(value):
            limit = int(value) if value is not None else 200
            save_preference("tenant_file_limit_mb", max(1, limit))

        def save_quota(value):
            quota = int(value) if value is not None else 5
            save_preference("tenant_media_quota_gb", max(1, quota))

        components["tenant_header_name"].change(
            save_tenant_header,
            inputs=[components["tenant_header_name"]],
            outputs=[],
        )
        components["tenant_file_limit_mb"].change(
            save_file_limit,
            inputs=[components["tenant_file_limit_mb"]],
            outputs=[],
        )
        components["tenant_media_quota_gb"].change(
            save_quota,
            inputs=[components["tenant_media_quota_gb"]],
            outputs=[],
        )

        def toggle_tool(tool_name, enabled):
            if "enabled_tools" not in _user_config:
                _user_config["enabled_tools"] = {}
            _user_config["enabled_tools"][tool_name] = enabled
            save_preference("enabled_tools", _user_config["enabled_tools"])
            return "Restart the app to apply changes."

        TTS_ENGINES = shared_state.get("TTS_ENGINES", {})

        def toggle_engine(engine_key, enabled):
            if "enabled_engines" not in _user_config:
                _user_config["enabled_engines"] = {}
            _user_config["enabled_engines"][engine_key] = enabled
            save_preference("enabled_engines", _user_config["enabled_engines"])
            return "Restart the app to apply changes."

        for key, _label in TOGGLEABLE_TOOLS:
            comp = components[f"tool_toggle_{key}"]
            comp.change(
                lambda enabled, k=key: toggle_tool(k, enabled),
                inputs=[comp],
                outputs=[components["settings_status"]],
            )

        for engine_key in TTS_ENGINES:
            comp = components[f"engine_toggle_{engine_key}"]
            comp.change(
                lambda enabled, k=engine_key: toggle_engine(k, enabled),
                inputs=[comp],
                outputs=[components["settings_status"]],
            )

        ASR_ENGINES = shared_state.get("ASR_ENGINES", {})

        def toggle_asr_engine(engine_key, enabled):
            if "enabled_asr_engines" not in _user_config:
                _user_config["enabled_asr_engines"] = {}
            _user_config["enabled_asr_engines"][engine_key] = enabled
            save_preference("enabled_asr_engines", _user_config["enabled_asr_engines"])
            return "Restart the app to apply changes."

        for engine_key in ASR_ENGINES:
            comp = components[f"asr_toggle_{engine_key}"]
            comp.change(
                lambda enabled, k=engine_key: toggle_asr_engine(k, enabled),
                inputs=[comp],
                outputs=[components["settings_status"]],
            )

        tool_components = shared_state.get("tool_components", {})
        library_components = tool_components.get("Library Manager", {}).get("components", {})
        voice_clone_components = tool_components.get("Voice Clone", {}).get("components", {})
        lm_asr_dropdown = library_components.get("proc_transcribe_model")
        lm_batch_asr_dropdown = library_components.get("dataset_transcribe_model")
        vc_model_dropdown = voice_clone_components.get("clone_model_dropdown")

        def _visible_asr_models():
            qwen_available = bool(shared_state.get("QWEN3_ASR_AVAILABLE", False))
            whisper_available = bool(shared_state.get("WHISPER_AVAILABLE", False))
            asr_settings = _user_config.get("enabled_asr_engines", {})
            visible = []
            for engine_key, engine_info in ASR_ENGINES.items():
                if not asr_settings.get(engine_key, engine_info.get("default_enabled", True)):
                    continue
                if engine_key == "Qwen3 ASR" and not qwen_available:
                    continue
                if engine_key == "Whisper" and not whisper_available:
                    continue
                visible.extend(engine_info.get("choices", []))
            return visible

        def _visible_tts_models():
            engine_settings = _user_config.get("enabled_engines", {})
            visible = []
            for engine_key, engine_info in TTS_ENGINES.items():
                if engine_settings.get(engine_key, engine_info.get("default_enabled", True)):
                    visible.extend(engine_info.get("choices", []))
            return visible

        def _choose_visible_or_fallback(preferred_model, visible_models):
            if preferred_model in visible_models:
                return preferred_model
            return visible_models[0] if visible_models else preferred_model

        def _save_preferred_asr(engine_key, model):
            norm_engine = str(coerce_choice_value(engine_key) or "")
            norm_model = str(coerce_choice_value(model) or "")
            _user_config["preferred_asr_engine"] = norm_engine
            _user_config["transcribe_model"] = norm_model
            save_preference("preferred_asr_engine", norm_engine)
            save_preference("transcribe_model", norm_model)

        def _save_preferred_tts(engine_key, model):
            norm_engine = str(coerce_choice_value(engine_key) or "")
            norm_model = str(coerce_choice_value(model) or "")
            _user_config["preferred_voice_clone_engine"] = norm_engine
            _user_config["voice_clone_model"] = norm_model
            save_preference("preferred_voice_clone_engine", norm_engine)
            save_preference("voice_clone_model", norm_model)

        def _on_preferred_asr_engine_change(engine_key, current_model):
            engine_key = str(coerce_choice_value(engine_key) or "")
            current_model = str(coerce_choice_value(current_model) or "")
            models = get_asr_models_for_engine(engine_key)
            selected_model = current_model if current_model in models else (models[-1] if models else current_model)
            _save_preferred_asr(engine_key, selected_model)

            visible_models = _visible_asr_models()
            lm_model = _choose_visible_or_fallback(selected_model, visible_models)
            status = f"Preferred ASR set to {engine_key} / {selected_model}"
            updates = [status, gr.update(choices=models, value=selected_model)]
            if lm_asr_dropdown is not None:
                updates.append(gr.update(value=lm_model))
            if lm_batch_asr_dropdown is not None:
                updates.append(gr.update(value=lm_model))
            return tuple(updates)

        def _on_preferred_asr_model_change(engine_key, model):
            engine_key = str(coerce_choice_value(engine_key) or "")
            model = str(coerce_choice_value(model) or "")
            models = get_asr_models_for_engine(engine_key)
            selected_model = model if model in models else (models[-1] if models else model)
            _save_preferred_asr(engine_key, selected_model)
            visible_models = _visible_asr_models()
            lm_model = _choose_visible_or_fallback(selected_model, visible_models)
            status = f"Preferred ASR model set to {selected_model}"
            updates = [status]
            if lm_asr_dropdown is not None:
                updates.append(gr.update(value=lm_model))
            if lm_batch_asr_dropdown is not None:
                updates.append(gr.update(value=lm_model))
            return tuple(updates)

        def _on_preferred_tts_engine_change(engine_key, current_model):
            engine_key = str(coerce_choice_value(engine_key) or "")
            current_model = str(coerce_choice_value(current_model) or "")
            models = get_tts_models_for_engine(engine_key)
            selected_model = current_model if current_model in models else (models[-1] if models else current_model)
            _save_preferred_tts(engine_key, selected_model)
            visible_models = _visible_tts_models()
            vc_model = _choose_visible_or_fallback(selected_model, visible_models)
            status = f"Preferred voice clone set to {engine_key} / {selected_model}"
            updates = [status, gr.update(choices=models, value=selected_model)]
            if vc_model_dropdown is not None:
                updates.append(gr.update(value=vc_model))
            return tuple(updates)

        def _on_preferred_tts_model_change(engine_key, model):
            engine_key = str(coerce_choice_value(engine_key) or "")
            model = str(coerce_choice_value(model) or "")
            models = get_tts_models_for_engine(engine_key)
            selected_model = model if model in models else (models[-1] if models else model)
            _save_preferred_tts(engine_key, selected_model)
            visible_models = _visible_tts_models()
            vc_model = _choose_visible_or_fallback(selected_model, visible_models)
            status = f"Preferred voice clone model set to {selected_model}"
            updates = [status]
            if vc_model_dropdown is not None:
                updates.append(gr.update(value=vc_model))
            return tuple(updates)

        asr_engine_outputs = [components["settings_status"], components["preferred_asr_model"]]
        asr_engine_outputs.extend([c for c in [lm_asr_dropdown, lm_batch_asr_dropdown] if c is not None])
        components["preferred_asr_engine"].change(
            _on_preferred_asr_engine_change,
            inputs=[components["preferred_asr_engine"], components["preferred_asr_model"]],
            outputs=asr_engine_outputs,
        )

        asr_model_outputs = [components["settings_status"]]
        asr_model_outputs.extend([c for c in [lm_asr_dropdown, lm_batch_asr_dropdown] if c is not None])
        components["preferred_asr_model"].change(
            _on_preferred_asr_model_change,
            inputs=[components["preferred_asr_engine"], components["preferred_asr_model"]],
            outputs=asr_model_outputs,
        )

        tts_engine_outputs = [components["settings_status"], components["preferred_voice_clone_model"]]
        if vc_model_dropdown is not None:
            tts_engine_outputs.append(vc_model_dropdown)
        components["preferred_voice_clone_engine"].change(
            _on_preferred_tts_engine_change,
            inputs=[components["preferred_voice_clone_engine"], components["preferred_voice_clone_model"]],
            outputs=tts_engine_outputs,
        )

        tts_model_outputs = [components["settings_status"]]
        if vc_model_dropdown is not None:
            tts_model_outputs.append(vc_model_dropdown)
        components["preferred_voice_clone_model"].change(
            _on_preferred_tts_model_change,
            inputs=[components["preferred_voice_clone_engine"], components["preferred_voice_clone_model"]],
            outputs=tts_model_outputs,
        )

        components["bypass_split_limit"].change(
            lambda x: save_preference("bypass_split_limit", x),
            inputs=[components["bypass_split_limit"]],
            outputs=[],
        )

        def reset_folder(folder_key):
            return components["default_folders"][folder_key]

        components["reset_samples_btn"].click(
            lambda: reset_folder("samples"),
            outputs=[components["settings_samples_folder"]],
        )
        components["reset_output_btn"].click(
            lambda: reset_folder("output"),
            outputs=[components["settings_output_folder"]],
        )
        components["reset_datasets_btn"].click(
            lambda: reset_folder("datasets"),
            outputs=[components["settings_datasets_folder"]],
        )
        components["reset_models_btn"].click(
            lambda: reset_folder("models"),
            outputs=[components["settings_models_folder"]],
        )
        components["reset_trained_models_btn"].click(
            lambda: reset_folder("models"),
            outputs=[components["settings_trained_models_folder"]],
        )

        def _download_one_model(model_display_name):
            if not model_display_name or model_display_name.startswith("---"):
                return False, "❌ Please select an actual model (not a category header)"
            model_id = components["MODEL_ID_MAP"].get(model_display_name, model_display_name)

            if model_id.startswith("whisper://"):
                whisper_size = model_id.split("whisper://", 1)[1].strip().lower()
                if whisper_size not in {"medium", "large"}:
                    return False, f"❌ Unsupported Whisper size: {whisper_size}"
                try:
                    import whisper
                except ImportError:
                    return False, "❌ Whisper is not installed. Install with: pip install openai-whisper"

                from modules.core_components.ai_models.model_utils import get_configured_models_dir

                whisper_dir = get_configured_models_dir() / "whisper"
                whisper_dir.mkdir(parents=True, exist_ok=True)
                model_url = whisper._MODELS.get(whisper_size)
                if not model_url:
                    return False, f"❌ Whisper model registry missing entry for: {whisper_size}"

                expected_file = whisper_dir / Path(model_url).name
                if expected_file.exists():
                    return True, f"✓ Whisper {whisper_size} already downloaded at {expected_file}"

                try:
                    whisper._download(model_url, str(whisper_dir), False)
                except Exception as e:
                    return False, (
                        f"❌ Failed to download Whisper {whisper_size}: {e}\n"
                        "Check internet access, then retry."
                    )

                if expected_file.exists():
                    return True, f"✓ Whisper {whisper_size} downloaded to {expected_file}"
                return False, (
                    "❌ Whisper download finished but expected file was not found:\n"
                    f"{expected_file}"
                )

            if model_id.startswith("mmaudio://"):
                mmaudio_display = model_id.split("mmaudio://", 1)[1]
                from modules.core_components.ai_models.foley_manager import get_foley_manager
                from modules.core_components.ai_models.model_utils import get_configured_models_dir

                models_dir = get_configured_models_dir()
                foley_manager = get_foley_manager(user_config=_user_config, models_dir=models_dir)
                try:
                    files = foley_manager.download_model_files(display_name=mmaudio_display)
                    return True, (
                        "✓ MMAudio files downloaded successfully\n"
                        f"Model: {files['model_weight']}\n"
                        f"VAE: {files['vae']}\n"
                        f"Synchformer: {files['synchformer']}"
                    )
                except Exception as e:
                    return False, f"❌ {str(e)}"

            if model_id == "source-separation://recommended":
                manager = _get_source_separation_manager()
                try:
                    models = manager.list_models(refresh=False)
                except Exception as exc:
                    print(
                        f"[Source Separation Download] Failed to load recommended model list: {exc}\n"
                        f"{traceback.format_exc()}",
                        flush=True,
                    )
                    return False, f"❌ Failed to load source-separation models: {str(exc)}"
                default_models = manager.get_default_models(models=models)
                if not default_models:
                    filenames = ", ".join(DEFAULT_SOURCE_SEPARATION_MODEL_FILENAMES)
                    return False, (
                        "❌ No recommended source-separation models were found in the current catalog.\n"
                        f"Expected one of: {filenames}"
                    )

                ok_count, fail_count, lines = _download_source_sep_models(default_models)
                summary = (
                    f"Recommended source-separation bundle finished: {ok_count} succeeded, "
                    f"{fail_count} failed.\n" + "\n".join(lines)
                )
                return fail_count == 0, summary

            if model_id.startswith("source-separation://"):
                model_filename = model_id.split("source-separation://", 1)[1].strip()
                manager = _get_source_separation_manager()
                try:
                    print(
                        f"[Source Separation Download] Download request for model={model_filename}",
                        flush=True,
                    )
                    info = manager.download_model(model_filename)
                    return True, f"✓ Downloaded {info['display_name']} to {info['model_dir']}"
                except Exception as exc:
                    print(
                        f"[Source Separation Download] Download failed for model={model_filename}: {exc}\n"
                        f"{traceback.format_exc()}",
                        flush=True,
                    )
                    return False, f"❌ {str(exc)}"

            if model_id.startswith("speaker-separation://"):
                expected_speakers = model_id.split("speaker-separation://", 1)[1].strip()
                manager = _get_speaker_separation_manager()
                try:
                    info = manager.download_model(int(expected_speakers))
                    return True, f"✓ Downloaded {info['display_name']} to {info['model_dir']}"
                except Exception as exc:
                    return False, f"❌ {str(exc)}"

            success, message, _path = download_model_from_huggingface(model_id, progress=None)
            status = f"✓ {message}" if success else f"❌ {message}"
            return success, status

        def _download_source_sep_models(models, *, progress=None):
            manager = _get_source_separation_manager()
            lines = []
            ok_count = 0
            fail_count = 0
            total = len(models)

            if total == 0:
                return ok_count, fail_count, ["Source Separation: no matching models found."]

            for index, model in enumerate(models, start=1):
                if progress is not None:
                    progress((index - 1) / total, desc=f"Downloading source-separation model {index}/{total}...")
                try:
                    print(
                        f"[Source Separation Download] Bulk download {index}/{total}: "
                        f"{model.display_name} ({model.model_filename})",
                        flush=True,
                    )
                    info = manager.download_model(model.model_filename, progress_callback=None)
                    ok_count += 1
                    lines.append(f"Source Separation / {model.display_name}: ✓ {info['model_dir']}")
                except Exception as exc:
                    print(
                        f"[Source Separation Download] Bulk download failed for "
                        f"{model.model_filename}: {exc}\n{traceback.format_exc()}",
                        flush=True,
                    )
                    fail_count += 1
                    lines.append(f"Source Separation / {model.display_name}: ❌ {str(exc)}")

            if progress is not None:
                progress(1.0, desc="Source-separation downloads finished.")

            return ok_count, fail_count, lines

        def download_model_clicked(model_display_name, download_all_models, progress=gr.Progress()):
            if download_all_models:
                model_names = []
                for item in components["ALL_MODEL_CHOICES"]:
                    if isinstance(item, (tuple, list)) and item:
                        name = str(item[0])
                    else:
                        name = str(item)
                    if name and not name.startswith("---") and name != "Source Separation - Recommended Defaults":
                        model_names.append(name)

                lines = []
                ok_count = 0
                fail_count = 0
                for name in model_names:
                    success, message = _download_one_model(name)
                    if success:
                        ok_count += 1
                    else:
                        fail_count += 1
                    lines.append(f"{name}: {message}")

                summary = (
                    f"Bulk download finished: {ok_count} succeeded, {fail_count} failed.\n\n"
                    + "\n\n".join(lines)
                )
                return summary

            _success, status = _download_one_model(model_display_name)
            return status

        def toggle_download_all(download_all_models):
            return gr.update(interactive=not download_all_models)

        components["download_all_models"].change(
            fn=toggle_download_all,
            inputs=[components["download_all_models"]],
            outputs=[components["model_select"]],
        )

        def apply_folder_changes(
            samples,
            output,
            datasets,
            models,
            trained_models,
            llm_api_key,
            llm_endpoint_url,
            llm_ollama_url,
            prompt_assistant_default_language,
            tenant_header_name,
            tenant_file_limit_mb,
            tenant_media_quota_gb,
        ):
            try:
                base_dir = Path(__file__).parent.parent.parent.parent

                new_samples = base_dir / samples
                new_output = base_dir / output
                new_datasets = base_dir / datasets
                new_models = base_dir / models
                new_trained_models = base_dir / trained_models

                new_samples.mkdir(exist_ok=True)
                new_output.mkdir(exist_ok=True)
                new_datasets.mkdir(exist_ok=True)
                new_models.mkdir(exist_ok=True)
                new_trained_models.mkdir(exist_ok=True)

                import os

                os.environ["HF_HOME"] = str(new_models)

                _user_config["samples_folder"] = samples
                _user_config["output_folder"] = output
                _user_config["datasets_folder"] = datasets
                _user_config["models_folder"] = models
                _user_config["trained_models_folder"] = trained_models
                _user_config["llm_api_key"] = llm_api_key.strip()
                _user_config["llm_endpoint_url"] = llm_endpoint_url.strip()
                _user_config["llm_ollama_url"] = llm_ollama_url.strip()
                _user_config["prompt_assistant_default_language"] = prompt_hub.resolve_prompt_assistant_language(
                    _user_config,
                    prompt_assistant_default_language,
                )
                _user_config["tenant_header_name"] = (tenant_header_name or "X-Tenant-Id").strip() or "X-Tenant-Id"
                _user_config["tenant_file_limit_mb"] = max(1, int(tenant_file_limit_mb or 200))
                _user_config["tenant_media_quota_gb"] = max(1, int(tenant_media_quota_gb or 5))
                save_config(_user_config)

                status_lines = [
                    "Folder paths updated successfully!",
                    f"\nSamples: {new_samples}",
                    f"Output: {new_output}",
                    f"Datasets: {new_datasets}",
                    f"Downloaded Models: {new_models}",
                    f"Trained Models: {new_trained_models}",
                ]
                status_lines.append(
                    f"LLM Endpoint: {(_user_config.get('llm_endpoint_url') or 'https://api.openai.com/v1')}"
                )
                status_lines.append(
                    f"Ollama URL: {(_user_config.get('llm_ollama_url') or 'http://127.0.0.1:11434/v1')}"
                )
                status_lines.append(
                    f"LLM API Key: {'[set]' if _user_config.get('llm_api_key') else '[not set]'}"
                )
                status_lines.append(
                    f"Prompt Assistant default language: {_user_config.get('prompt_assistant_default_language', 'English')}"
                )
                status_lines.append(
                    f"Tenant header: {_user_config.get('tenant_header_name', 'X-Tenant-Id')}"
                )
                status_lines.append(
                    f"Tenant limits: {_user_config.get('tenant_file_limit_mb', 200)}MB/file, "
                    f"{_user_config.get('tenant_media_quota_gb', 5)}GB/tenant"
                )
                status_lines.append("\nNote: Restart the app to fully apply changes to all components.")
                return "\n".join(status_lines)

            except Exception as e:
                return f"❌ Error applying changes: {str(e)}"

        components["download_btn"].click(
            fn=download_model_clicked,
            inputs=[components["model_select"], components["download_all_models"]],
            outputs=[components["settings_status"]],
        )

        components["apply_folders_btn"].click(
            apply_folder_changes,
            inputs=[
                components["settings_samples_folder"],
                components["settings_output_folder"],
                components["settings_datasets_folder"],
                components["settings_models_folder"],
                components["settings_trained_models_folder"],
                components["settings_llm_api_key"],
                components["settings_llm_endpoint_url"],
                components["settings_llm_ollama_url"],
                components["settings_prompt_assistant_default_language"],
                components["tenant_header_name"],
                components["tenant_file_limit_mb"],
                components["tenant_media_quota_gb"],
            ],
            outputs=[components["settings_status"]],
        )


# Export for tab registry
get_tool_class = lambda: SettingsTool


# Standalone testing
if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone

    run_tool_standalone(SettingsTool, port=7870, title="Settings - Standalone")
