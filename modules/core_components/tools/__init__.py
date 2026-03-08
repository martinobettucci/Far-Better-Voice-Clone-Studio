"""
Tool modules registry and loader.

This module manages all available tools and their configurations.
Tools can be enabled/disabled through configuration.

Also provides shared utilities for standalone tool testing.
"""

import json
import markdown
import platform
from pathlib import Path
from typing import Any
# from modules.core_components.tool_base import ToolConfig, Tool
from modules.core_components.tenant_context import (
    TenantResolutionError,
    tenant_error_message,
    validate_tenant_id,
)
from modules.core_components.tenant_storage import (
    BaseStoragePaths,
    TenantPaths,
    TenantStorageService,
)

# Import all tool modules here
from modules.core_components.tools import voice_clone
from modules.core_components.tools import voice_changer
from modules.core_components.tools import singing_enhancements
from modules.core_components.tools import voice_presets
from modules.core_components.tools import conversation
from modules.core_components.tools import voice_design
from modules.core_components.tools import library_manager
from modules.core_components.tools import train_model
from modules.core_components.tools import sound_effects
from modules.core_components.tools import prompt_manager
from modules.core_components.tools import output_history
from modules.core_components.tools import settings
from modules.core_components import help_page

# Registry of available tools
# Format: 'tool_name': (module, ToolConfig)
ALL_TOOLS = {
    'voice_clone': (voice_clone, voice_clone.VoiceCloneTool.config),
    'voice_changer': (voice_changer, voice_changer.VoiceChangerTool.config),
    'singing_enhancements': (
        singing_enhancements,
        singing_enhancements.SingingEnhancementsTool.config,
    ),
    'voice_presets': (voice_presets, voice_presets.VoicePresetsTool.config),
    'conversation': (conversation, conversation.ConversationTool.config),
    'voice_design': (voice_design, voice_design.VoiceDesignTool.config),
    'library_manager': (library_manager, library_manager.LibraryManagerTool.config),
    'train_model': (train_model, train_model.TrainModelTool.config),
    'sound_effects': (sound_effects, sound_effects.SoundEffectsTool.config),
    'prompt_manager': (prompt_manager, prompt_manager.PromptManagerTool.config),
    'output_history': (output_history, output_history.OutputHistoryTool.config),
    'help_guide': (help_page, help_page.HelpGuideTool.config),
    'settings': (settings, settings.SettingsTool.config),
}

_RUNTIME_ALLOW_CONFIG_API = False
_RUNTIME_DEFAULT_TENANT = None


def set_runtime_options(default_tenant=None, allow_config=None):
    """Set runtime-only options that are not persisted to config.json."""
    global _RUNTIME_ALLOW_CONFIG_API, _RUNTIME_DEFAULT_TENANT, _tenant_service

    if allow_config is not None:
        _RUNTIME_ALLOW_CONFIG_API = bool(allow_config)

    if default_tenant is not None:
        tenant = str(default_tenant).strip()
        if tenant and not validate_tenant_id(tenant):
            raise ValueError(
                "Invalid default tenant id. Expected pattern: [a-zA-Z0-9][a-zA-Z0-9._-]{0,63}"
            )
        _RUNTIME_DEFAULT_TENANT = tenant or None
    else:
        _RUNTIME_DEFAULT_TENANT = None

    if _tenant_service is not None:
        _tenant_service.update_default_tenant(_RUNTIME_DEFAULT_TENANT)

    return get_runtime_options()


def get_runtime_options():
    return {
        "allow_config": bool(_RUNTIME_ALLOW_CONFIG_API),
        "default_tenant": _RUNTIME_DEFAULT_TENANT,
    }


def is_config_api_enabled():
    return bool(_RUNTIME_ALLOW_CONFIG_API)


def get_tool_registry():
    """Get registry of all available tools and their configs."""
    return {name: config for name, (_, config) in ALL_TOOLS.items()}


def get_enabled_tools(user_config):
    """
    Get list of enabled tool modules based on user config.

    Args:
        user_config: User configuration dict

    Returns:
        List of (tool_module, ToolConfig) tuples for enabled tools
    """
    # Get tool settings from config (with defaults)
    tool_settings = user_config.get("enabled_tools", {})

    enabled_tools = []
    for name, (module, config) in ALL_TOOLS.items():
        if config.name == "Help Guide":
            enabled_tools.append((module, config))
            continue
        if config.name == "Settings" and not is_config_api_enabled():
            continue
        # Default to enabled if not specified
        is_enabled = tool_settings.get(config.name, config.enabled)
        if is_enabled:
            enabled_tools.append((module, config))

    return enabled_tools


def save_tool_settings(user_config, tool_name, enabled):
    """
    Save tool enabled/disabled setting.

    Args:
        user_config: User configuration dict (will be modified)
        tool_name: Tool name
        enabled: Whether tool is enabled
    """
    if "enabled_tools" not in user_config:
        user_config["enabled_tools"] = {}
    user_config["enabled_tools"][tool_name] = enabled


def create_enabled_tools(shared_state):
    """
    Create UI for all enabled tools.

    Args:
        shared_state: Shared globals (must include: _user_config, _active_emotions, and all helper functions)

    Returns:
        Dict mapping tool name to component references
    """
    user_config = shared_state.get('user_config', {})
    enabled_tools = get_enabled_tools(user_config)

    tool_components = {}
    for tool_module, config in enabled_tools:
        try:
            # Create tool UI - use get_tool_class if available
            if hasattr(tool_module, 'get_tool_class'):
                tool_class = tool_module.get_tool_class()
            else:
                # Fallback: find first Tool subclass
                tool_class = None
                for attr_name in dir(tool_module):
                    attr = getattr(tool_module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, Tool) and attr is not Tool:
                        tool_class = attr
                        break
                if not tool_class:
                    raise ValueError(f"No Tool class found in module {tool_module}")

            components = tool_class.create_tool(shared_state)
            tool_components[config.name] = {
                'module': tool_module,
                'config': config,
                'components': components,
                'tool_class': tool_class
            }
        except Exception as e:
            print(f"Warning: Failed to create tool '{config.name}': {e}")

    shared_state["tool_components"] = tool_components
    return tool_components


def setup_tool_events(tool_components, shared_state):
    """
    Set up event handlers for all tools.

    Args:
        tool_components: Dictionary returned by create_enabled_tools()
        shared_state: Shared globals
    """
    for tool_name, tool_info in tool_components.items():
        try:
            tool_class = tool_info['tool_class']
            components = tool_info['components']

            # Setup events
            tool_class.setup_events(components, shared_state)
        except Exception as e:
            print(f"Warning: Failed to setup events for tool '{tool_name}': {e}")


# Add this to modules so it can be accessed
__all__ = [
    'ALL_TOOLS',
    'get_tool_registry',
    'get_enabled_tools',
    'save_tool_settings',
    'set_runtime_options',
    'get_runtime_options',
    'is_config_api_enabled',
    'create_enabled_tools',
    'setup_tool_events',
    'PROJECT_ROOT',
    'CONFIG_FILE',
    'get_configured_dir',
    'get_base_storage_paths',
    'get_tenant_service',
    'resolve_tenant_paths',
    'format_tenant_usage_meter',
    'configure_tts_manager_for_tenant',
    'load_config',
    'save_config',
    'save_preference',
    'format_help_html',
    'play_completion_beep',
    'get_sample_choices',
    'strip_sample_extension',
    'get_available_samples',
    'get_prompt_cache_path',
    'load_sample_details',
    'get_dataset_folders',
    'get_dataset_files',
    'get_or_create_voice_prompt_standalone',
    'build_shared_state',
    'run_tool_standalone',
    'SHARED_CSS',
    'TRIGGER_HIDE_CSS',
]


# ============================================================================
# Shared utilities for standalone tool testing
# ============================================================================

# Config file path (relative to project root)
# Find project root by searching upward for voice_clone_studio.py
def _find_project_root():
    """Find project root by searching upward for voice_clone_studio.py."""
    current = Path(__file__).parent
    for _ in range(10):  # Limit search depth
        if (current / "voice_clone_studio.py").exists():
            return current
        current = current.parent
    # Fallback to best guess (4 levels up from tools/__init__.py)
    return Path(__file__).parent.parent.parent.parent

PROJECT_ROOT = _find_project_root()
CONFIG_FILE = PROJECT_ROOT / "config.json"


def get_configured_dir(folder_key, default):
    """Get a directory path from config, with fallback default.

    Reads the current config each time to pick up changes from Settings.

    Args:
        folder_key: Config key (e.g. 'samples_folder', 'models_folder')
        default: Default folder name if not in config

    Returns:
        Path object for the configured directory
    """
    config = load_config()
    return PROJECT_ROOT / config.get(folder_key, default)


_tenant_service = None


def get_base_storage_paths(config=None):
    """Get configured base paths before tenant routing is applied."""
    cfg = config or load_config()
    return BaseStoragePaths(
        samples_dir=PROJECT_ROOT / cfg.get("samples_folder", "samples"),
        datasets_dir=PROJECT_ROOT / cfg.get("datasets_folder", "datasets"),
        output_dir=PROJECT_ROOT / cfg.get("output_folder", "output"),
        trained_models_dir=PROJECT_ROOT / cfg.get("trained_models_folder", "models"),
        temp_dir=PROJECT_ROOT / cfg.get("temp_folder", "temp"),
    )


def get_tenant_service(config=None):
    """Get singleton tenant storage service, refreshed from config when needed."""
    global _tenant_service
    cfg = config or load_config()
    base_paths = get_base_storage_paths(cfg)
    header_name = cfg.get("tenant_header_name", "X-Tenant-Id")
    file_limit = cfg.get("tenant_file_limit_mb", 200)
    quota_gb = cfg.get("tenant_media_quota_gb", 5)

    if _tenant_service is None:
        _tenant_service = TenantStorageService(
            base_paths=base_paths,
            tenant_header_name=header_name,
            tenant_file_limit_mb=file_limit,
            tenant_media_quota_gb=quota_gb,
            default_tenant_id=_RUNTIME_DEFAULT_TENANT,
        )
    else:
        _tenant_service.base_paths = base_paths
        _tenant_service.update_header_name(header_name)
        _tenant_service.update_limits(file_limit, quota_gb)
        _tenant_service.update_default_tenant(_RUNTIME_DEFAULT_TENANT)

    return _tenant_service


def resolve_tenant_paths(request=None, required=True, config=None):
    """Resolve tenant paths from request header."""
    service = get_tenant_service(config=config)
    return service.get_tenant_paths(request=request, required=required)


def format_tenant_usage_meter(usage_summary):
    """Render compact usage meter text for UI labels."""
    used = usage_summary.get("used_bytes", 0)
    limit = usage_summary.get("limit_bytes", 0)
    pct = usage_summary.get("percent", 0.0)
    from modules.core_components.tenant_storage import format_bytes

    return f"Tenant storage: {format_bytes(used)} / {format_bytes(limit)} ({pct:.1f}%)"

# Shared CSS for all tools
# - Hides trigger widgets used by modal system
# - Styles file list groups for prep_audio, finetune_dataset, etc.
SHARED_CSS = """
#confirm-trigger {
    display: none !important;
}
#input-trigger {
    display: none !important;
}
#prompt-apply-trigger {
    display: none !important;
}
#audio-route-trigger {
    display: none !important;
}
#finetune-files-group > div {
    display: grid !important;
}
#finetune-files-container {
    max-height: 400px;
    overflow-y: auto;
}
#finetune-files-group label {
    background: none !important;
    border: none !important;
    padding: 4px 8px !important;
    margin: 2px 0 !important;
    box-shadow: none !important;
}
#finetune-files-group label:hover {
    background: var(--color-accent-soft) !important;
}
#output-files-group > div {
    display: grid !important;
}
#output-files-container {
    max-height: 800px;
    overflow-y: auto;
}
#output-files-group label {
    background: none !important;
    border: none !important;
    padding: 4px 8px !important;
    margin: 2px 0 !important;
    box-shadow: none !important;
}
#output-files-group label:hover {
    background: var(--color-accent-soft) !important;
}

/* Push Settings (last tab) to the far right - only top-level tabs */
#main-tabs > .tab-wrapper > .tab-container[role="tablist"] {
    display: flex !important;
}
#main-tabs > .tab-wrapper > .tab-container[role="tablist"] > button:last-child {
    margin-left: auto !important;
}
"""

# Alias for backward compatibility
TRIGGER_HIDE_CSS = SHARED_CSS


def load_config():
    """Load user preferences from config file.

    Returns:
        dict: User configuration with defaults
    """
    default_config = {
        "transcribe_model": "Whisper",
        "preferred_asr_engine": "Qwen3 ASR",
        "tts_base_size": "Large",
        "custom_voice_size": "Large",
        "voice_clone_model": "Qwen3 - Large",
        "preferred_voice_clone_engine": "Qwen3",
        "language": "Auto",
        "conv_pause_duration": 0.5,
        "whisper_language": "Auto-detect",
        "low_cpu_mem_usage": False,
        "attention_mechanism": "auto",
        "qwen_asr_max_inference_batch_size": 8,
        "qwen_asr_aggressive_cleanup": True,
        "qwen_asr_oom_retry": True,
        "deterministic_mode": False,
        "offline_mode": False,
        "memory_max_rss_pct": 70.0,
        "memory_min_available_mb": 2048,
        "memory_max_gpu_reserved_pct": 90.0,
        "heavy_job_timeout_s": 7200,
        "voice_prompt_memory_cache_limit": 8,
        "luxtts_prompt_memory_cache_limit": 8,
        "browser_notifications": True,
        "samples_folder": "samples",
        "output_folder": "output",
        "datasets_folder": "datasets",
        "temp_folder": "temp",
        "models_folder": "models",
        "trained_models_folder": "models",
        "tenant_header_name": "X-Tenant-Id",
        "tenant_file_limit_mb": 200,
        "tenant_media_quota_gb": 5,
        "emotions": None,
        "conv_model_type": "Qwen Speakers",
        "conv_model_size": "Large",
        "conv_base_model_size": "Large",
        "vibevoice_model_size": "Small",
        "conv_pause_linebreak": 0.5,
        "conv_pause_period": 0.3,
        "conv_pause_comma": 0.2,
        "conv_pause_question": 0.4,
        "conv_pause_hyphen": 0.15,
        "voice_clone_show_expert_params": False,
        "conversation_show_expert_params": False,
        "voice_changer_show_expert_params": False,
        "voice_changer_profile": "speech",
        "chatterbox_min_p": 0.05,
        "chatterbox_max_new_tokens": 2048,
        "voice_changer_vc_n_cfm_timesteps": 0,
        "llm_endpoint_url": "https://api.openai.com/v1",
        "llm_api_key": "",
        "llm_use_local_ollama": False,
        "llm_ollama_url": "http://127.0.0.1:11434/v1",
        "llm_model": "gpt-4o-mini",
        "prompt_assistant_default_language": "English",
    }

    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                saved_config = json.load(f)
                # Merge with defaults to handle new settings
                default_config.update(saved_config)
        else:
            # Create config file with defaults if it doesn't exist
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created new config file: {CONFIG_FILE}")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")

    # Migration guards for Prompt Manager endpoint config.
    config_changed = False
    def _scalar_choice(value):
        if isinstance(value, (list, tuple)):
            if len(value) >= 2:
                return value[1]
            if len(value) == 1:
                return value[0]
            return ""
        return value

    # Normalize any legacy tuple/list dropdown payloads.
    normalized_asr = _scalar_choice(default_config.get("transcribe_model"))
    if normalized_asr != default_config.get("transcribe_model"):
        default_config["transcribe_model"] = normalized_asr
        config_changed = True

    normalized_vc = _scalar_choice(default_config.get("voice_clone_model"))
    if normalized_vc != default_config.get("voice_clone_model"):
        default_config["voice_clone_model"] = normalized_vc
        config_changed = True

    llm_model = str(default_config.get("llm_model", "")).strip()
    if not llm_model or llm_model.lower().endswith(".gguf"):
        default_config["llm_model"] = "gpt-4o-mini"
        config_changed = True

    if not str(default_config.get("llm_endpoint_url", "")).strip():
        default_config["llm_endpoint_url"] = "https://api.openai.com/v1"
        config_changed = True

    if not str(default_config.get("llm_ollama_url", "")).strip():
        default_config["llm_ollama_url"] = "http://127.0.0.1:11434/v1"
        config_changed = True

    enabled_tools = default_config.get("enabled_tools", {})
    if (
        isinstance(enabled_tools, dict)
        and "Prep Samples" in enabled_tools
        and "Library Manager" not in enabled_tools
    ):
        enabled_tools["Library Manager"] = bool(enabled_tools["Prep Samples"])
        config_changed = True

    if config_changed:
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save migrated config: {e}")

    # Initialize emotions if not present (first launch or corrupted config)
    if not default_config.get("emotions"):
        from modules.core_components import CORE_EMOTIONS
        # Sort alphabetically (case-insensitive)
        default_config["emotions"] = dict(sorted(CORE_EMOTIONS.items(), key=lambda x: x[0].lower()))
        # Save config with emotions
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(default_config, f, indent=2)
            print("Initialized emotions in config")
        except Exception as e:
            print(f"Warning: Could not save initial emotions: {e}")

    return default_config


def save_config(config, key=None, value=None, force=False):
    """Save user preferences to config file.

    Optionally update a single preference before saving.

    Args:
        config: Dictionary of user preferences
        key: Optional - preference key to update before saving
        value: Optional - preference value to set
    """
    if not force and not is_config_api_enabled():
        return False

    if key is not None:
        config[key] = value

    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Warning: Could not save config: {e}")
        return False


def format_help_html(markdown_text, height="70vh"):
    """Convert markdown to HTML with scrollable container styling that matches Gradio components.

    Args:
        markdown_text: Markdown content to convert
        height: CSS height value (default: "70vh")
    """
    html_content = markdown.markdown(
        markdown_text,
        extensions=['fenced_code', 'tables', 'nl2br']
    )
    return f"""
    <div style="
        width: 100%;
        max-height: {height};
        overflow-y: auto;
        box-sizing: border-box;
        color: var(--block-label-text-color);
        font-size: var(--block-text-size);
        font-family: var(--font);
        line-height: 1.6;
    ">
        {html_content}
    </div>
    """

# Audio notification helper
def play_completion_beep():
    """Play audio notification when generation completes (uses notification.wav file)."""
    try:
        # Check if notifications are enabled in settings
        config = load_config()
        if not config.get("browser_notifications", True):
            return  # User disabled notifications

        # Print completion message to console
        print("\n=== Generation Complete! ===\n", flush=True)

        # Play notification sound from audio file
        # Path is relative to tools/__init__.py -> go up to core_components/
        notification_path = Path(__file__).parent.parent / "notification.wav"

        if notification_path.exists():
            try:
                if platform.system() == "Windows":
                    # Windows: Use winsound.PlaySound with audio file (synchronous to ensure it plays)
                    import winsound
                    winsound.PlaySound(str(notification_path), winsound.SND_FILENAME)
                elif platform.system() == "Darwin":
                    # macOS: Use afplay
                    import subprocess
                    subprocess.Popen(["afplay", str(notification_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    # Linux: Try aplay (ALSA), fallback to paplay (PulseAudio), fail silently if neither exists
                    import subprocess
                    try:
                        subprocess.Popen(["aplay", "-q", str(notification_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except FileNotFoundError:
                        try:
                            subprocess.Popen(["paplay", str(notification_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        except FileNotFoundError:
                            pass  # No audio player available, fail silently
            except Exception:
                # Fail silently for notification beeps
                pass
        else:
            # Notification file missing, use ASCII bell
            print('\a', end='', flush=True)
    except Exception as outer_e:
        # Final fallback - at least print the message
        try:
            print("\n=== Generation Complete! ===\n", flush=True)
            print(f"(Notification error: {outer_e})", flush=True)
        except:
            pass


# ===== Sample Management Helpers (Voice Clone & related tools) =====

def _resolve_tenant_paths_or_error(request=None, strict=False, config=None):
    """Resolve tenant paths with optional non-raising behavior."""
    service = get_tenant_service(config=config)
    try:
        paths = service.get_tenant_paths(request=request, required=strict)
        return paths, None
    except TenantResolutionError:
        return None, tenant_error_message(service.tenant_header_name)


def _tenant_error_list_entry(message: str) -> list[str]:
    return [f"(Tenant Error) {message}"]


def get_sample_choices(request=None, strict=False, config=None):
    """Get list of sample names for FileLister/dropdown for current tenant."""
    import json

    paths, error_msg = _resolve_tenant_paths_or_error(request=request, strict=strict, config=config)
    if error_msg:
        return _tenant_error_list_entry(error_msg)
    if paths is None:
        return ["(No samples found)"]

    samples = []
    for wav_file in paths.samples_dir.glob("*.wav"):
        json_file = wav_file.with_suffix(".json")
        name = wav_file.stem
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    name = meta.get("Name", meta.get("name", wav_file.stem))
            except Exception:
                pass
        if not name.lower().endswith(".wav"):
            name += ".wav"
        samples.append(name)
    return samples if samples else ["(No samples found)"]


def strip_sample_extension(name):
    """Strip .wav extension from a sample name for use with load_sample_details etc."""
    if name and name.lower().endswith(".wav"):
        return name[:-4]
    return name

def get_available_samples(request=None, strict=False, config=None):
    """Get full sample data for current tenant."""
    import json

    paths, _error_msg = _resolve_tenant_paths_or_error(request=request, strict=strict, config=config)
    if paths is None:
        return []

    samples = []
    for wav_file in paths.samples_dir.glob("*.wav"):
        json_file = wav_file.with_suffix(".json")
        meta = {}
        name = wav_file.stem
        ref_text = ""
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                name = meta.get("Name", meta.get("name", wav_file.stem))
                ref_text = meta.get("Text", meta.get("text", ""))
            except Exception:
                pass
        samples.append({
            "name": name,
            "wav_path": str(wav_file),
            "ref_text": ref_text,
            "meta": meta
        })
    return samples

def get_prompt_cache_path(sample_name, model_size, request=None, strict=False, config=None):
    """Get cache path for voice prompt in tenant sample directory."""
    paths, _error_msg = _resolve_tenant_paths_or_error(request=request, strict=strict, config=config)
    if paths is None:
        return get_configured_dir("samples_folder", "samples") / f"{sample_name}_{model_size}.pt"
    return paths.samples_dir / f"{sample_name}_{model_size}.pt"

def load_sample_details(sample_name, request=None, strict=False, config=None):
    """
    Load full details for a sample: audio path, text, and info.

    Returns:
        tuple: (audio_path, ref_text, info_string) or (None, "", "") if not found
    """
    if not sample_name:
        return None, "", ""

    import soundfile as sf
    samples = get_available_samples(request=request, strict=strict, config=config)

    for s in samples:
        if s["name"] == sample_name:
            # Check cache status for both model sizes
            cache_small = get_prompt_cache_path(
                sample_name, "0.6B", request=request, strict=False, config=config
            ).exists()
            cache_large = get_prompt_cache_path(
                sample_name, "1.7B", request=request, strict=False, config=config
            ).exists()

            if cache_small and cache_large:
                cache_status = "Qwen Cache: ⚡ Small, Large"
            elif cache_small:
                cache_status = "Qwen Cache: ⚡ Small"
            elif cache_large:
                cache_status = "Qwen Cache: ⚡ Large"
            else:
                cache_status = "Qwen Cache: 📦 Not cached"

            # Check LuxTTS cache status
            paths, _error_msg = _resolve_tenant_paths_or_error(request=request, strict=False, config=config)
            sample_dir = paths.samples_dir if paths is not None else get_configured_dir("samples_folder", "samples")
            luxtts_cached = (sample_dir / f"{sample_name}_luxtts.pt").exists()
            lux_status = "LuxTTS: ⚡ Cached" if luxtts_cached else "LuxTTS: 📦 Not cached"

            try:
                audio_data, sr = sf.read(s["wav_path"])
                duration = len(audio_data) / sr
                info = f"**Info**\n\nDuration: {duration:.2f}s | {cache_status} | {lux_status}"
            except:
                info = f"**Info**\n\n{cache_status} | {lux_status}"

            # Add design instructions if this was a Voice Design sample
            meta = s.get("meta", {})
            if meta.get("Type") == "Voice Design" and meta.get("Instruct"):
                info += f"\n\n**Voice Design:**\n{meta['Instruct']}"

            return s["wav_path"], s["ref_text"], info

    return None, "", ""


# ===== Dataset Management Helpers (Library Manager & Train Model tools) =====

def get_dataset_folders(request=None, strict=False, config=None):
    """Get list of subfolders in current tenant datasets directory."""
    paths, error_msg = _resolve_tenant_paths_or_error(request=request, strict=strict, config=config)
    if error_msg:
        return _tenant_error_list_entry(error_msg)
    if paths is None or not paths.datasets_dir.exists():
        return ["(No folders)"]
    folders = sorted([d.name for d in paths.datasets_dir.iterdir() if d.is_dir()])
    return folders if folders else ["(No folders)"]


def get_dataset_files(folder=None, request=None, strict=False, config=None):
    """Get list of audio file names in a tenant dataset subfolder for FileLister."""
    paths, _error_msg = _resolve_tenant_paths_or_error(request=request, strict=strict, config=config)
    if paths is None or not paths.datasets_dir.exists():
        return []

    if folder and folder not in ("(No folders)", "(Select Dataset)"):
        scan_dir = paths.datasets_dir / folder
    else:
        scan_dir = paths.datasets_dir

    if not scan_dir.exists():
        return []

    audio_files = sorted(
        list(scan_dir.glob("*.wav")) + list(scan_dir.glob("*.mp3")),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    return [f.name for f in audio_files]


def get_tenant_usage_summary(request=None, strict=False, config=None):
    """Return tenant media usage summary dict or None if tenant unresolved."""
    service = get_tenant_service(config=config)
    paths, _error_msg = _resolve_tenant_paths_or_error(request=request, strict=strict, config=config)
    if paths is None:
        return None
    return service.compute_usage_summary(paths)


def get_tenant_path_bundle(request=None, strict=False, config=None):
    """Return tenant path dataclass for current request."""
    if strict:
        return resolve_tenant_paths(request=request, required=True, config=config)
    paths, _error_msg = _resolve_tenant_paths_or_error(request=request, strict=False, config=config)
    return paths


def get_tenant_samples_dir(request=None, strict=False, config=None):
    paths = get_tenant_path_bundle(request=request, strict=strict, config=config)
    if paths is None:
        return get_configured_dir("samples_folder", "samples")
    return paths.samples_dir


def get_tenant_datasets_dir(request=None, strict=False, config=None):
    paths = get_tenant_path_bundle(request=request, strict=strict, config=config)
    if paths is None:
        return get_configured_dir("datasets_folder", "datasets")
    return paths.datasets_dir


def get_tenant_output_dir(request=None, strict=False, config=None):
    paths = get_tenant_path_bundle(request=request, strict=strict, config=config)
    if paths is None:
        return get_configured_dir("output_folder", "output")
    return paths.output_dir


def get_tenant_trained_models_dir(request=None, strict=False, config=None):
    paths = get_tenant_path_bundle(request=request, strict=strict, config=config)
    if paths is None:
        return get_configured_dir("trained_models_folder", "models")
    return paths.trained_models_dir


def get_or_create_voice_prompt_standalone(
    model,
    sample_name,
    wav_path,
    ref_text,
    model_size,
    progress_callback=None,
    request=None,
):
    """
    Get cached voice prompt or create new one using tts_manager.

    This is the real implementation that handles voice prompt caching.
    """
    from modules.core_components.ai_models.tts_manager import get_tts_manager

    tts_manager = get_tts_manager()
    tts_manager.samples_dir = get_tenant_samples_dir(request=request, strict=False)

    # Compute hash to check if sample has changed
    sample_hash = tts_manager.compute_sample_hash(wav_path, ref_text)

    # Try to load from cache
    prompt_items = tts_manager.load_voice_prompt(sample_name, sample_hash, model_size)

    if prompt_items is not None:
        if progress_callback:
            progress_callback(0.35, desc="Using cached voice prompt...")
        return prompt_items, True  # True = was cached

    # Create new prompt
    if progress_callback:
        progress_callback(0.2, desc="Processing voice sample (first time)...")

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=wav_path,
        ref_text=ref_text,
        x_vector_only_mode=False,
    )

    # Save to cache
    if progress_callback:
        progress_callback(0.35, desc="Caching voice prompt...")

    tts_manager.save_voice_prompt(sample_name, prompt_items, sample_hash, model_size)

    return prompt_items, False  # False = newly created


def configure_tts_manager_for_tenant(request=None, strict=False, config=None):
    """Ensure global TTS manager uses current tenant sample directory for caches."""
    from modules.core_components.ai_models.tts_manager import get_tts_manager

    tts_manager = get_tts_manager()
    tts_manager.samples_dir = get_tenant_samples_dir(request=request, strict=strict, config=config)
    return tts_manager


def build_shared_state(
    user_config,
    active_emotions,
    directories,
    constants,
    managers=None,
    confirm_trigger=None,
    input_trigger=None,
    prompt_apply_trigger=None,
    audio_route_trigger=None,
    main_tabs_component=None,
):
    """
    Build shared_state dictionary for main app or standalone testing.

    Centralizes all the boilerplate for creating shared_state with proper structure.

    Args:
        user_config: User configuration dict
        active_emotions: Active emotions dict
        directories: Dict with keys: OUTPUT_DIR, SAMPLES_DIR, DATASETS_DIR, TEMP_DIR
        constants: Dict with keys: LANGUAGES, CUSTOM_VOICE_SPEAKERS, MODEL_SIZES_*, etc.
        managers: Optional dict with keys: tts_manager, asr_manager (for main app)
        confirm_trigger: Gradio component for confirmation modal
        input_trigger: Gradio component for input modal

    Returns:
        Dict ready to pass to create_enabled_tabs() and setup_tab_events()
    """
    from modules.core_components import (
        show_confirmation_modal_js,
        show_input_modal_js,
        get_emotion_choices,
        calculate_emotion_values,
        handle_save_emotion,
        handle_delete_emotion
    )
    from modules.core_components.ui_components import (
        create_qwen_advanced_params,
        create_vibevoice_advanced_params,
        create_luxtts_advanced_params,
        create_chatterbox_advanced_params,
        create_emotion_intensity_slider,
        create_pause_controls
    )
    from modules.core_components.ui_components.audio_routing import (
        audio_route_build_payload,
        audio_route_get_available_targets,
        audio_route_get_target_label,
        audio_route_get_target_tab_id,
        audio_route_get_targets_for_tool_names,
    )
    from modules.core_components.ai_models.model_utils import (
        get_trained_models as get_trained_models_util,
        get_trained_model_names as get_trained_model_names_util,
        train_model as train_model_util,
        download_model_from_huggingface as download_model_util
    )
    from modules.core_components.runtime import get_memory_governor

    # Import audio utilities BEFORE building shared_state
    from modules.core_components.audio_utils import (
        is_audio_file as is_audio_file_util,
        is_video_file as is_video_file_util,
        extract_audio_from_video as extract_audio_from_video_util,
        get_audio_duration as get_audio_duration_util,
        format_time as format_time_util,
        normalize_audio as normalize_audio_util,
        remove_silences as remove_silences_util,
        convert_to_mono as convert_to_mono_util,
        save_audio_as_sample as save_as_sample_util,
        clean_audio as clean_audio_util,
        check_audio_format as check_audio_format_util
    )
    import modules.core_components.prompt_hub as prompt_hub

    # Check optional dependencies
    try:
        import whisper
        WHISPER_AVAILABLE = True
    except ImportError:
        WHISPER_AVAILABLE = False

    try:
        from qwen_asr import Qwen3ASRModel
        QWEN3_ASR_AVAILABLE = True
    except ImportError:
        QWEN3_ASR_AVAILABLE = False

    # DeepFilterNet / Torchaudio Compatibility Shim
    try:
        from modules.deepfilternet import deepfilternet_torchaudio_patch
        deepfilternet_torchaudio_patch.apply_patches()
    except ImportError:
        print("Warning: compatibility_patches module not found. DeepFilterNet may fail to load.")

    # Try importing DeepFilterNet
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        from df.io import load_audio as df_load_audio
        DEEPFILTER_AVAILABLE = True
    except ImportError as e:
        # If it still fails with the specific backend error, print guidance
        if "torchaudio.backend" in str(e):
            print(f"⚠ DeepFilterNet failed to load due to torchaudio incompatibility: {e}")
        else:
            print(f"⚠ DeepFilterNet not available: {e}")
        DEEPFILTER_AVAILABLE = False

    def clean_audio_standalone(audio_file, progress=None):
        """Clean audio using DeepFilterNet if available, otherwise return unchanged."""
        if not DEEPFILTER_AVAILABLE:
            if progress:
                progress(1.0, desc="DeepFilterNet not available")
            print("[WARN] DeepFilterNet not available in this environment")
            return audio_file, "⚠ DeepFilterNet not available"

        # DeepFilterNet is available - create a lazy loader for the model
        def get_deepfilter_lazy():
            """Lazy load DeepFilterNet model for standalone mode."""
            from df.enhance import init_df

            # Cache the model (simple module-level caching)
            if not hasattr(get_deepfilter_lazy, '_model_cache'):
                print("Loading DeepFilterNet model...")
                res = init_df()
                if isinstance(res, tuple):
                    model, state, params = res
                else:
                    model, state, params = res, None, None
                get_deepfilter_lazy._model_cache = (model, state, params)

            return get_deepfilter_lazy._model_cache

        # Use the real clean_audio function with lazy model loader
        result = clean_audio_util(audio_file, directories.get('TEMP_DIR'), get_deepfilter_lazy, progress)

        # Unload DeepFilterNet model from memory after use
        if hasattr(get_deepfilter_lazy, '_model_cache'):
            del get_deepfilter_lazy._model_cache
            try:
                from modules.core_components.ai_models.model_utils import empty_device_cache
                empty_device_cache()
            except ImportError:
                pass
            import gc
            gc.collect()
            print("DeepFilterNet model unloaded")

        return result

    memory_governor = get_memory_governor(user_config)

    def get_memory_snapshot():
        memory_governor.update_config(user_config)
        return memory_governor.snapshot()

    def run_heavy_job(job_name, fn, request=None, timeout_s=None):
        memory_governor.update_config(user_config)
        tenant_id = None
        if request is not None:
            try:
                tenant_paths = resolve_tenant_paths(request=request, required=False, config=user_config)
                if tenant_paths is not None:
                    tenant_id = tenant_paths.tenant_id
            except TenantResolutionError:
                tenant_id = None
        return memory_governor.run_heavy(
            job_name=job_name,
            fn=fn,
            tenant_id=tenant_id,
            timeout_s=timeout_s,
        )

    def run_heavy_stream_job(job_name, fn, request=None, timeout_s=None):
        memory_governor.update_config(user_config)
        tenant_id = None
        if request is not None:
            try:
                tenant_paths = resolve_tenant_paths(request=request, required=False, config=user_config)
                if tenant_paths is not None:
                    tenant_id = tenant_paths.tenant_id
            except TenantResolutionError:
                tenant_id = None
        return memory_governor.run_heavy_stream(
            job_name=job_name,
            fn=fn,
            tenant_id=tenant_id,
            timeout_s=timeout_s,
        )

    shared_state = {
        # Config & Emotions
        'user_config': user_config,
        '_user_config': user_config,
        '_active_emotions': active_emotions,

        # Directories
        'OUTPUT_DIR': directories.get('OUTPUT_DIR'),
        'SAMPLES_DIR': directories.get('SAMPLES_DIR'),
        'DATASETS_DIR': directories.get('DATASETS_DIR'),
        'TEMP_DIR': directories.get('TEMP_DIR'),
        'TENANT_HEADER_NAME': user_config.get("tenant_header_name", "X-Tenant-Id"),
        'TENANT_FILE_LIMIT_MB': int(user_config.get("tenant_file_limit_mb", 200)),
        'TENANT_MEDIA_QUOTA_GB': int(user_config.get("tenant_media_quota_gb", 5)),
        'tenant_service': get_tenant_service(user_config),
        'ALLOW_CONFIG_API': is_config_api_enabled(),
        'get_memory_snapshot': get_memory_snapshot,
        'run_heavy_job': run_heavy_job,
        'run_heavy_stream_job': run_heavy_stream_job,

        # Constants
        'LANGUAGES': constants.get('LANGUAGES', []),
        'CUSTOM_VOICE_SPEAKERS': constants.get('CUSTOM_VOICE_SPEAKERS', []),
        'MODEL_SIZES': constants.get('MODEL_SIZES'),
        'MODEL_SIZES_BASE': constants.get('MODEL_SIZES_BASE'),
        'MODEL_SIZES_CUSTOM': constants.get('MODEL_SIZES_CUSTOM'),
        'MODEL_SIZES_DESIGN': constants.get('MODEL_SIZES_DESIGN'),
        'MODEL_SIZES_VIBEVOICE': constants.get('MODEL_SIZES_VIBEVOICE'),
        'MODEL_SIZES_QWEN3_ASR': constants.get('MODEL_SIZES_QWEN3_ASR', ['Small', 'Large']),
        'MODEL_SIZES_MMAUDIO': constants.get('MODEL_SIZES_MMAUDIO', ['Medium (44kHz)', 'Large v2 (44kHz)']),
        'VOICE_CLONE_OPTIONS': constants.get('VOICE_CLONE_OPTIONS'),
        'DEFAULT_VOICE_CLONE_MODEL': constants.get('DEFAULT_VOICE_CLONE_MODEL'),
        'TTS_ENGINES': constants.get('TTS_ENGINES', {}),
        'ASR_ENGINES': constants.get('ASR_ENGINES', {}),
        'ASR_OPTIONS': constants.get('ASR_OPTIONS', []),
        'DEFAULT_ASR_MODEL': constants.get('DEFAULT_ASR_MODEL', 'Qwen3 ASR - Large'),
        'WHISPER_AVAILABLE': WHISPER_AVAILABLE,
        'QWEN3_ASR_AVAILABLE': QWEN3_ASR_AVAILABLE,
        'DEEPFILTER_AVAILABLE': DEEPFILTER_AVAILABLE,

        # UI component creators
        'create_qwen_advanced_params': create_qwen_advanced_params,
        'create_vibevoice_advanced_params': create_vibevoice_advanced_params,
        'create_luxtts_advanced_params': create_luxtts_advanced_params,
        'create_chatterbox_advanced_params': create_chatterbox_advanced_params,
        'create_emotion_intensity_slider': create_emotion_intensity_slider,
        'create_pause_controls': create_pause_controls,

        # Emotion management
        'get_emotion_choices': get_emotion_choices,

        # Core utilities
        'play_completion_beep': play_completion_beep,
        'format_help_html': format_help_html,

        # Modal triggers and helpers
        'confirm_trigger': confirm_trigger,
        'input_trigger': input_trigger,
        'prompt_apply_trigger': prompt_apply_trigger,
        'audio_route_trigger': audio_route_trigger,
        'main_tabs_component': main_tabs_component,
        'show_confirmation_modal_js': show_confirmation_modal_js,
        'show_input_modal_js': show_input_modal_js,
        'prompt_hub': prompt_hub,
        'prompt_system_prompt_choices': prompt_hub.SYSTEM_PROMPT_CHOICES,
        'prompt_system_prompts': prompt_hub.SYSTEM_PROMPTS,
        'prompt_targets': prompt_hub.PROMPT_TARGETS,
        'prompt_get_names': prompt_hub.get_prompt_names,
        'prompt_get_text': prompt_hub.get_prompt_text,
        'prompt_generate_for_target': prompt_hub.generate_for_target,
        'prompt_get_target_default_preset': prompt_hub.get_target_default_preset,
        'prompt_get_target_tab_id': prompt_hub.get_target_tab_id,
        'prompt_get_enabled_target_choices': prompt_hub.get_enabled_target_choices,
        'prompt_build_apply_payload': prompt_hub.build_apply_payload,
        'prompt_parse_apply_payload': prompt_hub.parse_apply_payload,
        'audio_route_build_payload': audio_route_build_payload,
        'audio_route_get_available_targets': audio_route_get_available_targets,
        'audio_route_get_target_label': audio_route_get_target_label,
        'audio_route_get_target_tab_id': audio_route_get_target_tab_id,
        'audio_route_get_initial_targets': lambda: audio_route_get_targets_for_tool_names(
            config.name for _module, config in get_enabled_tools(user_config)
        ),

        # Helper functions
        'get_tenant_paths': lambda request=None, strict=False: resolve_tenant_paths(
            request=request, required=strict, config=user_config
        ),
        'get_tenant_usage_summary': lambda request=None, strict=False: get_tenant_usage_summary(
            request=request, strict=strict, config=user_config
        ),
        'format_tenant_usage_meter': format_tenant_usage_meter,
        'get_tenant_samples_dir': lambda request=None, strict=False: get_tenant_samples_dir(
            request=request, strict=strict, config=user_config
        ),
        'get_tenant_datasets_dir': lambda request=None, strict=False: get_tenant_datasets_dir(
            request=request, strict=strict, config=user_config
        ),
        'get_tenant_output_dir': lambda request=None, strict=False: get_tenant_output_dir(
            request=request, strict=strict, config=user_config
        ),
        'get_tenant_trained_models_dir': lambda request=None, strict=False: get_tenant_trained_models_dir(
            request=request, strict=strict, config=user_config
        ),

        'get_trained_models': lambda request=None, strict=False: get_trained_models_util(
            get_tenant_trained_models_dir(request=request, strict=strict, config=user_config)
        ),
        'get_trained_model_names': lambda request=None, strict=False: get_trained_model_names_util(
            get_tenant_trained_models_dir(request=request, strict=strict, config=user_config)
        ),
        'train_model': lambda folder, speaker_name, ref_audio, batch_size, lr, epochs, save_interval, progress=None, request=None: train_model_util(
            folder, speaker_name, ref_audio, batch_size, lr, epochs, save_interval,
            user_config, get_tenant_datasets_dir(request=request, strict=True, config=user_config),
            directories.get('OUTPUT_DIR').parent,  # project_root
            play_completion_beep, progress,
            trained_models_root=get_tenant_trained_models_dir(request=request, strict=True, config=user_config),
        ),

        # Dataset management helpers
        'get_dataset_folders': lambda request=None, strict=False: get_dataset_folders(
            request=request, strict=strict, config=user_config
        ),
        'get_dataset_files': lambda folder=None, request=None, strict=False: get_dataset_files(
            folder=folder, request=request, strict=strict, config=user_config
        ),

        # Sample management helpers (Voice Clone & related tools)
        'get_sample_choices': lambda request=None, strict=False: get_sample_choices(
            request=request, strict=strict, config=user_config
        ),
        'get_available_samples': lambda request=None, strict=False: get_available_samples(
            request=request, strict=strict, config=user_config
        ),
        'get_prompt_cache_path': lambda sample_name, model_size, request=None, strict=False: get_prompt_cache_path(
            sample_name=sample_name, model_size=model_size, request=request, strict=strict, config=user_config
        ),
        'load_sample_details': lambda sample_name, request=None, strict=False: load_sample_details(
            sample_name=sample_name, request=request, strict=strict, config=user_config
        ),
        'get_or_create_voice_prompt': get_or_create_voice_prompt_standalone,  # Default mock for standalone, main app overrides
        'configure_tts_manager_for_tenant': lambda request=None, strict=False: configure_tts_manager_for_tenant(
            request=request, strict=strict, config=user_config
        ),
        'refresh_samples': lambda request=None, strict=False: __import__('gradio').update(
            choices=get_sample_choices(request=request, strict=strict, config=user_config)
        ),

        # Audio utilities (shared by Library Manager and related tools)
        'is_audio_file': is_audio_file_util,
        'is_video_file': is_video_file_util,
        'extract_audio_from_video': lambda path, request=None: extract_audio_from_video_util(
            path, get_tenant_path_bundle(request=request, strict=True, config=user_config).temp_dir
            if get_tenant_path_bundle(request=request, strict=True, config=user_config)
            else directories.get('TEMP_DIR')
        ),
        'get_audio_duration': get_audio_duration_util,
        'format_time': format_time_util,
        'normalize_audio': lambda audio, request=None: normalize_audio_util(
            audio,
            (get_tenant_path_bundle(request=request, strict=True, config=user_config).temp_dir
             if get_tenant_path_bundle(request=request, strict=True, config=user_config)
             else directories.get('TEMP_DIR'))
        ),
        'remove_silences': lambda audio, request=None: remove_silences_util(
            audio,
            (get_tenant_path_bundle(request=request, strict=True, config=user_config).temp_dir
             if get_tenant_path_bundle(request=request, strict=True, config=user_config)
             else directories.get('TEMP_DIR'))
        ),
        'convert_to_mono': lambda audio, request=None: convert_to_mono_util(
            audio,
            (get_tenant_path_bundle(request=request, strict=True, config=user_config).temp_dir
             if get_tenant_path_bundle(request=request, strict=True, config=user_config)
             else directories.get('TEMP_DIR'))
        ),
        'clean_audio': lambda audio, progress=None: clean_audio_standalone(audio, progress),
        'save_as_sample': lambda audio, text, name, request=None: save_as_sample_util(
            audio, text, name, get_tenant_samples_dir(request=request, strict=True, config=user_config)
        ),

        # Model downloading
        'download_model_from_huggingface': lambda model_id, progress=None: download_model_util(
            model_id,
            models_dir=directories.get('OUTPUT_DIR').parent / user_config.get("models_folder", "models"),
            progress=progress
        ),
    }

    # Lambdas that reference shared_state (must be added after dict creation)
    shared_state['save_emotion_handler'] = lambda name, intensity, temp, rep_pen, top_p: handle_save_emotion(
        shared_state['_active_emotions'], name, intensity, temp, rep_pen, top_p
    )
    shared_state['delete_emotion_handler'] = lambda confirm_val, emotion_name: handle_delete_emotion(
        shared_state['_active_emotions'], confirm_val, emotion_name
    )
    def _save_preference(k, v):
        # Gradio events that call this helper often declare outputs=[].
        # Return None so these callbacks are treated as side-effect-only.
        if not is_config_api_enabled():
            return None
        save_config(shared_state['_user_config'], k, v)
        return None

    shared_state['save_preference'] = _save_preference

    # Add managers if provided (for main app)
    if managers:
        shared_state['tts_manager'] = managers.get('tts_manager')
        shared_state['asr_manager'] = managers.get('asr_manager')
        shared_state['foley_manager'] = managers.get('foley_manager')

    return shared_state


def run_tool_standalone(ToolClass, port=7860, title="Tool - Standalone", extra_shared_state=None):
    """
    Run a tool in standalone mode for testing.

    Handles all boilerplate: config loading, shared_state setup, modal initialization, and app launch.

    Args:
        ToolClass: The Tool class to run (e.g., VoicePresetsTool)
        port: Server port (default: 7860)
        title: Window title (default: "Tool - Standalone")
        extra_shared_state: Optional dict of tool-specific shared_state entries to add/override

    Usage:
        if __name__ == "__main__":
            from modules.core_components.tools import run_tool_standalone
            run_tool_standalone(VoicePresetsTool, port=7863, title="Voice Presets - Standalone")

        # With tool-specific helpers:
        if __name__ == "__main__":
            extra = {'get_sample_choices': lambda: ['sample1', 'sample2']}
            run_tool_standalone(VoiceCloneTool, port=7862, extra_shared_state=extra)
    """
    import os
    import gradio as gr
    from pathlib import Path
    from modules.core_components import (
        CONFIRMATION_MODAL_CSS,
        CONFIRMATION_MODAL_HEAD,
        CONFIRMATION_MODAL_HTML,
        INPUT_MODAL_CSS,
        INPUT_MODAL_HEAD,
        INPUT_MODAL_HTML,
        SINGING_WAVEFORM_CSS,
        SINGING_WAVEFORM_HEAD,
        load_emotions_from_config
    )
    from modules.core_components.constants import (
        LANGUAGES,
        CUSTOM_VOICE_SPEAKERS,
        MODEL_SIZES_CUSTOM,
        MODEL_SIZES_BASE,
        MODEL_SIZES_VIBEVOICE,
        VOICE_CLONE_OPTIONS,
        DEFAULT_VOICE_CLONE_MODEL,
        TTS_ENGINES,
        ASR_ENGINES,
        ASR_OPTIONS,
        DEFAULT_ASR_MODEL
    )

    # Find project root
    project_root = CONFIG_FILE.parent

    # Load config and emotions
    user_config = load_config()
    active_emotions = load_emotions_from_config(user_config)

    # Setup directories
    OUTPUT_DIR = project_root / user_config.get("output_folder", "output")
    SAMPLES_DIR = project_root / user_config.get("samples_folder", "samples")
    DATASETS_DIR = project_root / user_config.get("datasets_folder", "datasets")
    TEMP_DIR = project_root / user_config.get("temp_folder", "temp")
    MODELS_DIR = project_root / user_config.get("models_folder", "models")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Keep HF cache aligned with configured models folder so online usage
    # accumulates files in the same place used for offline prep.
    MODELS_DIR.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(MODELS_DIR)

    # Load theme
    theme_path = Path(__file__).parent.parent / "ui_components" / "theme.json"
    theme = gr.themes.Base.load(str(theme_path)) if theme_path.exists() else None

    # Create Gradio app
    with gr.Blocks(title=title) as app:
        # Add modal HTML
        gr.HTML(CONFIRMATION_MODAL_HTML)
        gr.HTML(INPUT_MODAL_HTML)

        gr.Markdown(f"# 🎙️ {ToolClass.config.name} (Standalone Testing)")
        gr.Markdown("*Standalone mode with full modal support*")

        # Hidden trigger widgets
        with gr.Row():
            confirm_trigger = gr.Textbox(label="Confirm Trigger", value="", elem_id="confirm-trigger")
            input_trigger = gr.Textbox(label="Input Trigger", value="", elem_id="input-trigger")
            audio_route_trigger = gr.Textbox(label="Audio Route Trigger", value="", elem_id="audio-route-trigger")

        # Build shared_state using centralized helper
        shared_state = build_shared_state(
            user_config=user_config,
            active_emotions=active_emotions,
            directories={
                'OUTPUT_DIR': OUTPUT_DIR,
                'SAMPLES_DIR': SAMPLES_DIR,
                'DATASETS_DIR': DATASETS_DIR,
                'TEMP_DIR': TEMP_DIR
            },
            constants={
                'LANGUAGES': LANGUAGES,
                'CUSTOM_VOICE_SPEAKERS': CUSTOM_VOICE_SPEAKERS,
                'MODEL_SIZES_CUSTOM': MODEL_SIZES_CUSTOM,
                'MODEL_SIZES_BASE': MODEL_SIZES_BASE,
                'MODEL_SIZES_VIBEVOICE': MODEL_SIZES_VIBEVOICE,
                'VOICE_CLONE_OPTIONS': VOICE_CLONE_OPTIONS,
                'DEFAULT_VOICE_CLONE_MODEL': DEFAULT_VOICE_CLONE_MODEL,
                'TTS_ENGINES': TTS_ENGINES,
                'ASR_ENGINES': ASR_ENGINES,
                'ASR_OPTIONS': ASR_OPTIONS,
                'DEFAULT_ASR_MODEL': DEFAULT_ASR_MODEL,
            },
            confirm_trigger=confirm_trigger,
            input_trigger=input_trigger,
            audio_route_trigger=audio_route_trigger,
        )

        # Add tool-specific shared_state entries
        if extra_shared_state:
            shared_state.update(extra_shared_state)

        # Create and setup tool
        components = ToolClass.create_tool(shared_state)
        ToolClass.setup_events(components, shared_state)

    print(f"[*] Output: {OUTPUT_DIR}")
    from modules.core_components.ai_models.model_utils import get_trained_models
    models_dir = project_root / user_config.get("models_folder", "models")
    print(f"[*] Found {len(get_trained_models(models_dir))} trained models")
    print(f"\n✓ {ToolClass.config.name} UI loaded successfully!")
    print(f"[*] Launching on http://127.0.0.1:{port}")

    app.queue(default_concurrency_limit=1)
    app.launch(
        theme=theme,
        css=CONFIRMATION_MODAL_CSS + INPUT_MODAL_CSS + SINGING_WAVEFORM_CSS + SHARED_CSS,
        head=CONFIRMATION_MODAL_HEAD + INPUT_MODAL_HEAD + SINGING_WAVEFORM_HEAD,
        server_port=port,
        server_name="127.0.0.1",
        inbrowser=False,
        allowed_paths=[str(SAMPLES_DIR), str(OUTPUT_DIR), str(DATASETS_DIR), str(TEMP_DIR)]
    )
