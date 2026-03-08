"""
Voice Clone Studio - Main Application

Minimal orchestrator that loads modular tools and wires them together.
All tab implementations are in modules/core_components/tools/

ARCHITECTURE:
- Each tool is fully independent and self-contained
- Tools import get_tts_manager() / get_asr_manager() directly (singleton pattern)
- Tools implement their own generation logic, file I/O, progress updates
- This file only provides: directories, constants, shared utilities, modals
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import json
import random
import tempfile
import time
import logging
import html

# Suppress Gradio's noisy HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import gradio as gr

# Core imports
from modules.core_components import (
    CONFIRMATION_MODAL_CSS,
    CONFIRMATION_MODAL_HEAD,
    CONFIRMATION_MODAL_HTML,
    INPUT_MODAL_CSS,
    INPUT_MODAL_HEAD,
    INPUT_MODAL_HTML,
    SINGING_WAVEFORM_CSS,
    SINGING_WAVEFORM_HEAD,
    CORE_EMOTIONS,
    show_confirmation_modal_js,
    show_input_modal_js,
    load_emotions_from_config,
    get_emotion_choices,
    calculate_emotion_values,
    handle_save_emotion,
    handle_delete_emotion
)

# UI components
from modules.core_components.ui_components import (
    create_qwen_advanced_params,
    create_vibevoice_advanced_params,
    create_emotion_intensity_slider,
    create_pause_controls
)

# AI Managers
from modules.core_components.ai_models import (
    get_tts_manager,
    get_asr_manager,
    get_foley_manager
)
from modules.core_components.ai_models.model_utils import (
    get_trained_models,
    configure_runtime_reproducibility,
)
from modules.core_components.runtime import (
    build_resource_monitor_payload,
    get_memory_governor,
)

# Modular tools
from modules.core_components.tools import (
    create_enabled_tools,
    setup_tool_events,
    load_config,
    save_config,
    build_shared_state,
    play_completion_beep,
    format_help_html,
    TRIGGER_HIDE_CSS,
    CONFIG_FILE,
    set_runtime_options,
    get_tenant_service,
)
from modules.core_components.tenant_context import validate_tenant_id

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent / "modules"))

# ============================================================================
# CONFIG & SETUP
# ============================================================================


def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Voice Clone Studio")
    parser.add_argument(
        "--default-tenant",
        type=str,
        default=None,
        help=(
            "Fallback tenant id to use when tenant header is missing. "
            "Must match: [a-zA-Z0-9][a-zA-Z0-9._-]{0,63}"
        ),
    )
    parser.add_argument(
        "--allow-config",
        action="store_true",
        help="Enable configuration UI/API mutations (Settings tab and preference writes).",
    )
    args, _ = parser.parse_known_args()

    if args.default_tenant:
        args.default_tenant = args.default_tenant.strip()
        if not validate_tenant_id(args.default_tenant):
            parser.error(
                "--default-tenant is invalid. Expected pattern: "
                "[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}"
            )

    return args


_cli_args = _parse_cli_args()
set_runtime_options(
    default_tenant=_cli_args.default_tenant,
    allow_config=_cli_args.allow_config,
)

# Load config (CONFIG_FILE imported from tools)
_user_config = load_config()

if _cli_args.default_tenant:
    print(f"Default tenant fallback enabled: {_cli_args.default_tenant}")
if _cli_args.allow_config:
    print("Configuration UI/API enabled (--allow-config).")
else:
    print("Configuration UI/API disabled (pass --allow-config to enable).")

# Apply deterministic runtime behavior early (before model loading)
configure_runtime_reproducibility(_user_config.get("deterministic_mode", False))

# Check which engines are available before building UI
if _user_config.get("skip_engine_check", False):
    print("\nSkipping engine availability check (skip_engine_check enabled)\n")
else:
    from modules.core_components.constants import check_engine_availability
    print()
    print("=" * 50)
    print("Checking available engines...")
    print("=" * 50)
    check_engine_availability(
        _user_config,
        save_config_fn=lambda key, value: save_config(_user_config, key, value)
    )
    print("=" * 50)
    print()

_active_emotions = load_emotions_from_config(_user_config)

# On macOS (MPS), training requires CUDA — auto-disable Train Model tab
import platform
if platform.system() == "Darwin":
    if "enabled_tools" not in _user_config:
        _user_config["enabled_tools"] = {}
    _user_config["enabled_tools"]["Train Model"] = False
    print("macOS detected: Train Model tab disabled (requires CUDA)")

# Ensure config has emotions key set (emotion_manager expects it)
if 'emotions' not in _user_config or _user_config['emotions'] is None:
    _user_config['emotions'] = _active_emotions

# Initialize directories
SAMPLES_DIR = Path(__file__).parent / _user_config.get("samples_folder", "samples")
OUTPUT_DIR = Path(__file__).parent / _user_config.get("output_folder", "output")
DATASETS_DIR = Path(__file__).parent / _user_config.get("datasets_folder", "datasets")
TEMP_DIR = Path(__file__).parent / _user_config.get("temp_folder", "temp")
MODELS_DIR = Path(__file__).parent / _user_config.get("models_folder", "models")

for dir_path in [SAMPLES_DIR, OUTPUT_DIR, DATASETS_DIR, TEMP_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Ensure HF cache lives inside configured models folder so online usage
# naturally accumulates offline assets in the same location.
os.environ["HF_HOME"] = str(MODELS_DIR)

# Clean temp folder at startup
for f in TEMP_DIR.iterdir():
    try:
        if f.is_file():
            f.unlink()
    except Exception:
        pass

# ============================================================================
# CONSTANTS - Import from central location
# ============================================================================

from modules.core_components.constants import (
    MODEL_SIZES,
    MODEL_SIZES_BASE,
    MODEL_SIZES_CUSTOM,
    MODEL_SIZES_DESIGN,
    MODEL_SIZES_VIBEVOICE,
    MODEL_SIZES_QWEN3_ASR,
    VOICE_CLONE_OPTIONS,
    DEFAULT_VOICE_CLONE_MODEL,
    TTS_ENGINES,
    ASR_ENGINES,
    ASR_OPTIONS,
    DEFAULT_ASR_MODEL,
    LANGUAGES,
    CUSTOM_VOICE_SPEAKERS,
    SUPPORTED_MODELS,
    SAMPLE_RATE,
    DEFAULT_CONFIG as DEFAULT_CONFIG_TEMPLATE,
    QWEN_GENERATION_DEFAULTS,
    VIBEVOICE_GENERATION_DEFAULTS,
    MODEL_SIZES_MMAUDIO,
    MMAUDIO_GENERATION_DEFAULTS,
)

# ============================================================================
# GLOBAL MANAGERS - Tools access via shared_state
# ============================================================================
_tts_manager = None
_asr_manager = None
_foley_manager = None

RESOURCE_MONITOR_CSS = """
#resource-monitor-card {
    margin-top: 0.4rem;
}
#resource-monitor-card .resource-monitor {
    border: 1px solid var(--block-border-color);
    border-radius: 12px;
    padding: 0.65rem;
    background: linear-gradient(
        180deg,
        color-mix(in srgb, var(--background-fill-primary) 88%, transparent),
        var(--background-fill-secondary)
    );
}
#resource-monitor-card .resource-monitor-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
    margin-bottom: 0.55rem;
}
#resource-monitor-card .resource-monitor-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--body-text-color);
}
#resource-monitor-card .resource-monitor-time {
    font-size: 0.75rem;
    color: var(--body-text-color-subdued);
}
#resource-monitor-card .resource-status-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    padding: 0.2rem 0.5rem;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    border: 1px solid transparent;
}
#resource-monitor-card .resource-status-badge--ok {
    color: #2f8f46;
    background: color-mix(in srgb, #2f8f46 16%, transparent);
    border-color: color-mix(in srgb, #2f8f46 35%, transparent);
}
#resource-monitor-card .resource-status-badge--warn {
    color: #b67800;
    background: color-mix(in srgb, #b67800 16%, transparent);
    border-color: color-mix(in srgb, #b67800 35%, transparent);
}
#resource-monitor-card .resource-status-badge--danger {
    color: #c93737;
    background: color-mix(in srgb, #c93737 16%, transparent);
    border-color: color-mix(in srgb, #c93737 35%, transparent);
}
#resource-monitor-card .resource-monitor-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.45rem;
}
#resource-monitor-card .resource-metric {
    border: 1px solid color-mix(in srgb, var(--block-border-color) 70%, transparent);
    border-radius: 10px;
    padding: 0.45rem 0.5rem;
    background: color-mix(in srgb, var(--background-fill-secondary) 75%, transparent);
}
#resource-monitor-card .resource-metric-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--body-text-color-subdued);
    text-transform: uppercase;
    letter-spacing: 0.02em;
}
#resource-monitor-card .resource-metric-value {
    margin-top: 0.15rem;
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--body-text-color);
}
#resource-monitor-card .resource-meter {
    margin-top: 0.35rem;
    height: 6px;
    background: color-mix(in srgb, var(--body-text-color-subdued) 20%, transparent);
    border-radius: 999px;
    overflow: hidden;
}
#resource-monitor-card .resource-meter-fill {
    display: block;
    height: 100%;
    border-radius: inherit;
}
#resource-monitor-card .resource-meter-fill--ok {
    background: #2f8f46;
}
#resource-monitor-card .resource-meter-fill--warn {
    background: #b67800;
}
#resource-monitor-card .resource-meter-fill--danger {
    background: #c93737;
}
#resource-monitor-card .resource-metric-detail {
    margin-top: 0.3rem;
    font-size: 0.74rem;
    color: var(--body-text-color-subdued);
}
#resource-monitor-card .resource-monitor-note {
    margin-top: 0.55rem;
    font-size: 0.75rem;
    color: var(--body-text-color-subdued);
    line-height: 1.35;
}
@media (max-width: 900px) {
    #resource-monitor-card .resource-monitor-grid {
        grid-template-columns: 1fr;
    }
}
"""

# ============================================================================
# UI CREATION
# ============================================================================

def create_ui():
    """Create the Gradio interface with modular tools."""

    # Initialize AI managers and make them available to wrapper functions
    global _tts_manager, _asr_manager, _foley_manager
    _tts_manager = get_tts_manager(_user_config, SAMPLES_DIR)
    _asr_manager = get_asr_manager(_user_config)
    _foley_manager = get_foley_manager(_user_config, MODELS_DIR)

    with gr.Blocks(title="Voice Clone Studio") as app:
        memory_governor = get_memory_governor(_user_config)
        tenant_service = get_tenant_service(_user_config)

        def _clamp_pct(value: float) -> float:
            return max(0.0, min(100.0, float(value)))

        def _metric_html(
            title: str,
            value: str,
            detail: str,
            pct: float | None = None,
            level: str = "ok",
        ) -> str:
            meter_html = ""
            if pct is not None:
                meter_html = (
                    '<div class="resource-meter">'
                    f'<span class="resource-meter-fill resource-meter-fill--{level}" style="width:{_clamp_pct(pct):.1f}%"></span>'
                    "</div>"
                )
            return (
                '<div class="resource-metric">'
                f'<div class="resource-metric-title">{html.escape(title)}</div>'
                f'<div class="resource-metric-value">{html.escape(value)}</div>'
                f"{meter_html}"
                f'<div class="resource-metric-detail">{html.escape(detail)}</div>'
                "</div>"
            )

        def _tenant_unresolved_reason(request: gr.Request | None) -> str:
            header_name = tenant_service.tenant_header_name
            default_tenant = (tenant_service.default_tenant_id or "").strip()
            headers = getattr(request, "headers", None) if request is not None else None
            if headers is None and request is not None and hasattr(request, "request"):
                headers = getattr(request.request, "headers", None)

            header_value = None
            if headers:
                header_value = headers.get(header_name) or headers.get(header_name.lower())

            if not header_value and not default_tenant:
                return f"Missing tenant header '{header_name}' and no default tenant fallback."
            if header_value and not validate_tenant_id(str(header_value).strip()):
                return f"Invalid tenant id in header '{header_name}'."
            return f"Tenant could not be resolved from header '{header_name}'."

        def _resource_monitor_html(request: gr.Request | None = None):
            snapshot = memory_governor.snapshot()
            tenant_id = None
            tenant_usage_summary = None
            tenant_unresolved_reason = None
            try:
                tenant_paths = tenant_service.get_tenant_paths(request=request, required=False)
                if tenant_paths is None:
                    tenant_unresolved_reason = _tenant_unresolved_reason(request)
                else:
                    tenant_id = tenant_paths.tenant_id
                    tenant_usage_summary = tenant_service.compute_usage_summary(tenant_paths)
            except Exception as e:
                tenant_unresolved_reason = f"Tenant resolution error: {str(e)}"

            payload = build_resource_monitor_payload(
                snapshot,
                memory_max_rss_pct=memory_governor.memory_max_rss_pct,
                memory_min_available_mb=memory_governor.memory_min_available_mb,
                memory_max_gpu_reserved_pct=memory_governor.memory_max_gpu_reserved_pct,
                max_active_heavy_jobs=memory_governor.max_active_heavy_jobs,
                tenant_id=tenant_id,
                tenant_usage_summary=tenant_usage_summary,
                tenant_unresolved_reason=tenant_unresolved_reason,
            )
            metrics_html = "".join(
                _metric_html(
                    metric["title"],
                    metric["value"],
                    metric["detail"],
                    pct=metric["pct"],
                    level=metric["level"],
                )
                for metric in payload["metrics"]
            )
            return (
                '<div class="resource-monitor">'
                '<div class="resource-monitor-header">'
                '<div>'
                '<div class="resource-monitor-title">Resource Monitor</div>'
                f'<div class="resource-monitor-time">Updated {payload["updated_at_text"]}</div>'
                "</div>"
                f'<span class="resource-status-badge resource-status-badge--{payload["status_class"]}">{html.escape(payload["status_label"])}</span>'
                "</div>"
                '<div class="resource-monitor-grid">'
                f"{metrics_html}"
                "</div>"
                f'<div class="resource-monitor-note">{html.escape(payload["guide_hint"])}</div>'
                "</div>"
            )

        # Modal HTML
        gr.HTML(CONFIRMATION_MODAL_HTML)
        gr.HTML(INPUT_MODAL_HTML)

        # Hidden triggers for modals
        confirm_trigger = gr.Textbox(label="Confirm Trigger", value="", elem_id="confirm-trigger")
        input_trigger = gr.Textbox(label="Input Trigger", value="", elem_id="input-trigger")
        prompt_apply_trigger = gr.Textbox(label="Prompt Apply Trigger", value="", elem_id="prompt-apply-trigger")
        audio_route_trigger = gr.Textbox(label="Audio Route Trigger", value="", elem_id="audio-route-trigger")

        # Header with unload button
        with gr.Row():
            with gr.Column(scale=20):
                gr.Markdown("""
                    # 🎙️ Voice Clone Studio
                    <p style="font-size: 0.9em; color: var(--body-text-color-subdued); margin-top: -10px;">Powered by Qwen3-TTS, VibeVoice, LuxTTS, Chatterbox and Whisper</p>
                    """)

            with gr.Column(scale=7, min_width=330):
                with gr.Row():
                    unload_all_btn = gr.Button("Clear VRAM", size="sm", variant="secondary")
                    refresh_memory_btn = gr.Button("Refresh", size="sm", variant="secondary")
                unload_status = gr.Markdown(" ", visible=True)
                with gr.Accordion("Resource Monitor", open=False, elem_id="resource-monitor-accordion"):
                    resource_status = gr.HTML(_resource_monitor_html(), visible=True, elem_id="resource-monitor-card")

        # ============================================================
        # BUILD SHARED STATE - everything tools need
        # ============================================================
        shared_state = build_shared_state(
            user_config=_user_config,
            active_emotions=_active_emotions,
            directories={
                'OUTPUT_DIR': OUTPUT_DIR,
                'SAMPLES_DIR': SAMPLES_DIR,
                'DATASETS_DIR': DATASETS_DIR,
                'TEMP_DIR': TEMP_DIR
            },
            constants={
                'MODEL_SIZES': MODEL_SIZES,
                'MODEL_SIZES_BASE': MODEL_SIZES_BASE,
                'MODEL_SIZES_CUSTOM': MODEL_SIZES_CUSTOM,
                'MODEL_SIZES_DESIGN': MODEL_SIZES_DESIGN,
                'MODEL_SIZES_VIBEVOICE': MODEL_SIZES_VIBEVOICE,
                'MODEL_SIZES_QWEN3_ASR': MODEL_SIZES_QWEN3_ASR,
                'MODEL_SIZES_MMAUDIO': MODEL_SIZES_MMAUDIO,
                'VOICE_CLONE_OPTIONS': VOICE_CLONE_OPTIONS,
                'DEFAULT_VOICE_CLONE_MODEL': DEFAULT_VOICE_CLONE_MODEL,
                'TTS_ENGINES': TTS_ENGINES,
                'ASR_ENGINES': ASR_ENGINES,
                'ASR_OPTIONS': ASR_OPTIONS,
                'DEFAULT_ASR_MODEL': DEFAULT_ASR_MODEL,
                'LANGUAGES': LANGUAGES,
                'CUSTOM_VOICE_SPEAKERS': CUSTOM_VOICE_SPEAKERS,
            },
            managers={
                'tts_manager': _tts_manager,
                'asr_manager': _asr_manager,
                'foley_manager': _foley_manager
            },
            confirm_trigger=confirm_trigger,
            input_trigger=input_trigger,
            prompt_apply_trigger=prompt_apply_trigger,
            audio_route_trigger=audio_route_trigger,
        )

        # ============================================================
        # LOAD ALL MODULAR TOOLS
        # ============================================================
        with gr.Tabs(elem_id="main-tabs") as main_tabs:
            shared_state['main_tabs_component'] = main_tabs
            tool_components = create_enabled_tools(shared_state)
        setup_tool_events(tool_components, shared_state)

        # Wire up unload button
        def on_unload_all():
            import gc
            _tts_manager.unload_all()
            _asr_manager.unload_all()
            _foley_manager.unload_all()
            gc.collect()
            from modules.core_components.ai_models.model_utils import empty_device_cache
            empty_device_cache()
            return "VRAM freed."

        # Clear status after 3 seconds to keep UI tidy
        def clear_status():
            time.sleep(3)
            return " "

        unload_all_btn.click(
            on_unload_all,
            outputs=[unload_status]
        ).then(
            fn=_resource_monitor_html,
            outputs=[resource_status],
        ).then(
            fn=clear_status,
            inputs=[],
            outputs=[unload_status],
            show_progress="hidden"
        )

        refresh_memory_btn.click(
            fn=_resource_monitor_html,
            outputs=[resource_status],
            show_progress="hidden"
        )

        app.load(
            fn=_resource_monitor_html,
            outputs=[resource_status],
            show_progress="hidden"
        )

    return app


if __name__ == "__main__":
    theme = gr.themes.Base.load('modules/core_components/ui_components/theme.json')
    app = create_ui()
    try:
        # Use 0.0.0.0 if user enabled network listening, otherwise localhost only
        network_mode = _user_config.get("listen_on_network", False)
        default_host = "0.0.0.0" if network_mode else "127.0.0.1"
        server_host = os.getenv("GRADIO_SERVER_NAME", default_host)

        # In network mode, don't auto-open browser — print the LAN address instead
        if network_mode or server_host == "0.0.0.0":
            import socket
            # Get actual LAN IP by checking which interface routes to external networks
            # This never sends data — just checks which local IP the OS would use
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("10.255.255.255", 1))
                lan_ip = s.getsockname()[0]
                s.close()
            except Exception:
                lan_ip = "<your-local-ip>"
            print()
            print("=" * 50)
            print("Network listening enabled")
            print("Local:   http://127.0.0.1:7860")
            print(f"Network: http://{lan_ip}:7860")
            print()
            print("NOTE: Only devices on your local network can")
            print("connect. Your router's firewall blocks outside")
            print("traffic unless you have port forwarding enabled.")
            print("Do NOT use this on public/untrusted WiFi.")
            print("=" * 50)
            print()

        app.queue(default_concurrency_limit=1)
        app.launch(
            server_name=server_host,
            server_port=7860,
            share=False,
            inbrowser=not (network_mode or server_host == "0.0.0.0"),
            theme=theme,
            css=(
                TRIGGER_HIDE_CSS
                + CONFIRMATION_MODAL_CSS
                + INPUT_MODAL_CSS
                + SINGING_WAVEFORM_CSS
                + RESOURCE_MONITOR_CSS
            ),
            head=CONFIRMATION_MODAL_HEAD + INPUT_MODAL_HEAD + SINGING_WAVEFORM_HEAD
        )
    except OSError:
        print()
        print("=" * 50)
        print("Voice Clone Studio is already running!")
        if network_mode or server_host == "0.0.0.0":
            print("Local:   http://127.0.0.1:7860")
            print(f"Network: http://{lan_ip}:7860")
        else:
            print("Check your browser at http://127.0.0.1:7860")
        print("=" * 50)
        print()
