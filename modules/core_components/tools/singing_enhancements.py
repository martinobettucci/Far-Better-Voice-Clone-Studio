"""
Singing Enhancements Tab

Studio-style vocal processing for singing clips with segment selection, effects,
autotune, and optional backing-track mixing.
"""
from __future__ import annotations

# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

from datetime import datetime
import json
from pathlib import Path
from textwrap import dedent
import traceback

import gradio as gr

from modules.core_components.ai_models import get_source_separation_manager
from modules.core_components.ai_models.source_separation_manager import (
    DETAILED_STEMS,
    SourceSeparationModel,
    SourceSeparationResult,
)
from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.runtime import MemoryAdmissionError
from modules.core_components.tools.generated_output_save import (
    get_existing_wav_stems,
    parse_modal_submission,
    sanitize_output_name,
    save_generated_output,
)
from modules.core_components.tools.singing_audio_pipeline import (
    AUTO_KEY_CONFIDENCE_THRESHOLD,
    NOTE_NAMES,
    AutotuneConfig,
    MixConfig,
    SegmentSelection,
    SingingEffectConfig,
    estimate_key_from_audio,
    process_singing_audio,
)
from modules.core_components.ui_components import SINGING_WAVEFORM_HTML
from modules.core_components.ui_components.audio_routing import (
    create_audio_route_controls,
    wire_audio_route_dropdown_refresh,
    wire_audio_route_listener,
    wire_audio_route_source,
)


WAVEFORM_SELECTOR_JS = """
(sourceValue) => {
    if (window.refreshSingingWaveformStudio) {
        return window.refreshSingingWaveformStudio(sourceValue, true);
    }
    return '';
}
"""

SOURCE_SEPARATION_STEM_ORDER = ("Vocals", "Backing", *DETAILED_STEMS)
SOURCE_SEPARATION_GOAL_CHOICES = (
    "Recommended",
    "Vocals + Backing",
    "Karaoke / Instrumental",
    "Detailed Stems",
    "All Models",
)
SOURCE_SEPARATION_VOCALS_DESTINATIONS = ("Studio", "Voice Changer")
DETECT_KEY_SOURCE_PRIORITY = (
    ("Source Separation Original", "sep_source_audio"),
    ("Source Separation Vocals", "sep_vocals_audio"),
    ("Source Separation Backing", "sep_backing_audio"),
    ("Current Lead Vocal", "source_audio"),
)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clean_audio_path(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def detect_key_from_available_audio(
    *,
    source_audio,
    sep_source_audio,
    sep_vocals_audio,
    sep_backing_audio,
):
    candidates = {
        "source_audio": _clean_audio_path(source_audio),
        "sep_source_audio": _clean_audio_path(sep_source_audio),
        "sep_vocals_audio": _clean_audio_path(sep_vocals_audio),
        "sep_backing_audio": _clean_audio_path(sep_backing_audio),
    }
    attempts: list[tuple[str, object]] = []
    best_label: str | None = None
    best_estimate = None

    for label, key in DETECT_KEY_SOURCE_PRIORITY:
        audio_path = candidates.get(key)
        if not audio_path:
            continue
        estimate = estimate_key_from_audio(audio_path, source_label=label.lower().replace(" ", "_"))
        attempts.append((label, estimate))
        if best_estimate is None or estimate.confidence > best_estimate.confidence:
            best_label = label
            best_estimate = estimate
        if estimate.confidence >= AUTO_KEY_CONFIDENCE_THRESHOLD and estimate.scale in {"major", "minor"}:
            return {
                "mode": "Manual",
                "tonic": estimate.tonic,
                "scale": estimate.scale,
                "status": (
                    f"Detected {estimate.tonic} {estimate.scale} from {label} "
                    f"(confidence {estimate.confidence:.3f}). Mode set to Manual."
                ),
                "show_manual_controls": True,
                "used_label": label,
                "best_estimate": estimate,
                "attempts": attempts,
            }

    if best_estimate is None:
        return {
            "mode": None,
            "tonic": None,
            "scale": None,
            "status": (
                "No audio available for key detection. "
                "Upload a lead vocal or run source separation first."
            ),
            "show_manual_controls": False,
            "used_label": None,
            "best_estimate": None,
            "attempts": attempts,
        }

    if best_estimate.scale not in {"major", "minor"}:
        return {
            "mode": None,
            "tonic": None,
            "scale": None,
            "status": (
                f"Unable to detect a usable key from {best_label}. "
                "Try a fuller mix or backing track."
            ),
            "show_manual_controls": False,
            "used_label": best_label,
            "best_estimate": best_estimate,
            "attempts": attempts,
        }

    attempt_summary = ", ".join(
        f"{label}: {estimate.tonic} {estimate.scale} ({estimate.confidence:.3f})"
        for label, estimate in attempts
    )
    return {
        "mode": "Manual",
        "tonic": best_estimate.tonic,
        "scale": best_estimate.scale,
        "status": (
            f"Low-confidence detection; filled {best_estimate.tonic} {best_estimate.scale} from {best_label} "
            f"(confidence {best_estimate.confidence:.3f}). Attempts: {attempt_summary}. "
            "If tuning sounds wrong, try Chromatic or provide a backing/full-song source."
        ),
        "show_manual_controls": True,
        "used_label": best_label,
        "best_estimate": best_estimate,
        "attempts": attempts,
    }


def _build_metadata_text(
    *,
    timestamp: str,
    source_audio,
    backing_audio,
    selection: SegmentSelection,
    autotune_result,
    effects: SingingEffectConfig,
    mix: MixConfig,
    output_type: str,
) -> str:
    source_name = Path(str(source_audio)).name if source_audio else "Unknown"
    backing_name = Path(str(backing_audio)).name if backing_audio else "None"
    return dedent(
        f"""\
        Generated: {timestamp}
        Type: Singing Enhancements
        Output: {output_type}
        Source Vocal: {source_name}
        Backing Track: {backing_name}
        Selection Enabled: {selection.enabled}
        Selection Start: {selection.start_sec:.4f}
        Selection End: {selection.end_sec:.4f}
        Requested Autotune Mode: {autotune_result.requested_mode}
        Resolved Autotune Mode: {autotune_result.resolved_mode}
        Resolved Tonic: {autotune_result.tonic or "N/A"}
        Resolved Scale: {autotune_result.scale}
        Autotune Confidence: {autotune_result.confidence:.4f}
        Autotune Source: {autotune_result.source}
        Autotune Applied Regions: {autotune_result.applied_regions}
        Autotune Status: {autotune_result.message}
        Pitch Shift (semitones): {effects.pitch_shift_semitones:.2f}
        High-Pass (Hz): {effects.highpass_hz:.1f}
        Low-Pass (Hz): {effects.lowpass_hz:.1f}
        Compression: {effects.compression_amount:.4f}
        Echo: {effects.echo_amount:.4f}
        Reverb: {effects.reverb_amount:.4f}
        Vocal Gain (dB): {mix.vocal_gain_db:.2f}
        Backing Gain (dB): {mix.backing_gain_db:.2f}
        """
    )


def _serialize_source_separation_result(result: SourceSeparationResult | None) -> str:
    if result is None:
        return ""
    return json.dumps(result.to_dict())


def deserialize_source_separation_result(value: str | None) -> SourceSeparationResult | None:
    if not value:
        return None
    try:
        return SourceSeparationResult.from_dict(json.loads(value))
    except Exception:
        return None


def _source_separation_stem_suffix(stem_name: str) -> str:
    return sanitize_output_name(stem_name).lower() or "stem"


def save_source_separation_outputs(
    result: SourceSeparationResult,
    output_dir: Path,
    raw_name: str | None,
    *,
    stem_name: str | None = None,
) -> list[Path]:
    """Persist one or all source-separation stems to the output directory."""
    if stem_name is not None:
        source_path = result.stem_paths.get(stem_name)
        if not source_path:
            raise ValueError(f"No {stem_name} stem is available to save.")
        saved_path = save_generated_output(
            audio_value=source_path,
            output_dir=output_dir,
            raw_name=raw_name,
            metadata_text=result.metadata_by_stem.get(stem_name) or result.metadata_text,
            default_sample_rate=44_100,
        )
        return [saved_path]

    clean_name = sanitize_output_name(raw_name)
    if not clean_name:
        raise ValueError("Invalid filename.")

    saved_paths: list[Path] = []
    for stem_name in SOURCE_SEPARATION_STEM_ORDER:
        source_path = result.stem_paths.get(stem_name)
        if not source_path:
            continue
        saved_path = save_generated_output(
            audio_value=source_path,
            output_dir=output_dir,
            raw_name=f"{clean_name}_{_source_separation_stem_suffix(stem_name)}",
            metadata_text=result.metadata_by_stem.get(stem_name) or result.metadata_text,
            default_sample_rate=44_100,
        )
        saved_paths.append(saved_path)
    return saved_paths


def _get_source_separation_manager_for_state(shared_state):
    user_config = shared_state.get("_user_config", {})
    output_dir = shared_state.get("OUTPUT_DIR")
    if output_dir is None:
        models_dir = Path(user_config.get("models_folder", "models"))
    else:
        models_dir = Path(output_dir).parent / user_config.get("models_folder", "models")
    return get_source_separation_manager(user_config=user_config, models_dir=models_dir)


def _source_separation_model_haystack(model: SourceSeparationModel) -> str:
    return f"{model.display_name} {model.model_filename} {model.architecture}".lower()


def _filter_source_separation_models_for_goal(manager, models, goal: str) -> list[SourceSeparationModel]:
    source = list(models)
    goal = str(goal or "Recommended").strip()

    if goal == "Recommended":
        recommended = manager.get_default_models(models=source)
        return recommended or source
    if goal == "Vocals + Backing":
        filtered = [
            model for model in source
            if model.stem_count == 2 and "karaoke" not in _source_separation_model_haystack(model)
        ]
        return filtered or [model for model in source if model.stem_count == 2] or source
    if goal == "Karaoke / Instrumental":
        filtered = [
            model for model in source
            if any(token in _source_separation_model_haystack(model) for token in ("karaoke", "_kara", "inst_", "inst-"))
        ]
        return filtered or [model for model in source if model.stem_count == 2] or source
    if goal == "Detailed Stems":
        filtered = [model for model in source if model.stem_count >= 4]
        return filtered or source
    return source


def _build_source_separation_goal_help(goal: str, models: list[SourceSeparationModel]) -> str:
    count = len(models)
    if goal == "Recommended":
        return (
            f"Showing {count} recommended models. Start here unless you know you specifically need "
            "karaoke-focused instrumental recovery or detailed stems."
        )
    if goal == "Vocals + Backing":
        return (
            f"Showing {count} direct 2-stem models. Use this when you mainly want a clean vocal track "
            "and a backing track quickly."
        )
    if goal == "Karaoke / Instrumental":
        return (
            f"Showing {count} karaoke/instrumental-focused models. Use this when keeping the backing "
            "track strong matters more than extracting every band stem."
        )
    if goal == "Detailed Stems":
        return (
            f"Showing {count} multi-stem models. Use this when you want drums/bass/other and the app can "
            "rebuild the backing track from those stems."
        )
    return f"Showing all {count} known music-separation models."


def _get_source_separation_model_choices_for_goal(manager, models, goal: str) -> list[tuple[str, str]]:
    filtered = _filter_source_separation_models_for_goal(manager, models, goal)
    return manager.get_dropdown_choices(models=filtered)


def _get_initial_source_separation_ui(shared_state):
    manager = _get_source_separation_manager_for_state(shared_state)
    try:
        models = manager.list_models(refresh=False)
    except Exception:
        models = []
    default_goal = SOURCE_SEPARATION_GOAL_CHOICES[0]
    filtered_models = _filter_source_separation_models_for_goal(manager, models, default_goal)
    model_choices = manager.get_dropdown_choices(models=filtered_models)
    default_model = manager.get_default_model_filename(models=filtered_models)
    if default_model is None and model_choices:
        default_model = model_choices[0][1]
    goal_help = _build_source_separation_goal_help(default_goal, filtered_models)
    return manager, models, default_goal, model_choices, default_model, goal_help


def build_source_separation_ui_payload(
    result: SourceSeparationResult | None,
    *,
    status_message: str,
) -> dict:
    """Prepare normalized UI payload for the source-separation outputs."""
    payload = {
        "vocals_audio": None,
        "backing_audio": None,
        "button_update": gr.update(interactive=False),
        "all_button_update": gr.update(interactive=False),
        "use_button_update": gr.update(interactive=False),
        "vocals_name": "",
        "backing_name": "",
        "all_name": "",
        "result_json": "",
        "status": status_message,
        "accordion_update": gr.update(visible=False),
        "detailed_updates": {
            stem: gr.update(value=None, visible=False)
            for stem in DETAILED_STEMS
        },
    }
    if result is None:
        return payload

    has_detailed = any(stem in result.stem_paths for stem in DETAILED_STEMS)
    payload.update(
        {
            "vocals_audio": result.stem_paths.get("Vocals"),
            "backing_audio": result.stem_paths.get("Backing"),
            "button_update": gr.update(interactive=True),
            "all_button_update": gr.update(interactive=bool(result.stem_paths)),
            "use_button_update": gr.update(interactive=True),
            "vocals_name": result.suggested_names.get("Vocals", ""),
            "backing_name": result.suggested_names.get("Backing", ""),
            "all_name": result.suggested_names.get("All", ""),
            "result_json": _serialize_source_separation_result(result),
            "status": status_message,
            "accordion_update": gr.update(visible=has_detailed),
        }
    )
    payload["detailed_updates"] = {
        stem: gr.update(value=result.stem_paths.get(stem), visible=stem in result.stem_paths)
        for stem in DETAILED_STEMS
    }
    return payload


def build_source_separation_studio_route_update(audio_path: str | None, label: str) -> tuple[dict, dict, str]:
    """Build the update tuple used when sending separated stems into Studio."""
    if not audio_path:
        return gr.update(), gr.update(), f"No {label.lower()} stem is available."
    return (
        gr.update(value=audio_path),
        gr.update(selected="singing_studio"),
        f"Loaded {label.lower()} into Studio.",
    )


def build_source_separation_vocals_route_update(
    *,
    audio_path: str | None,
    destination: str,
    audio_route_build,
    main_tabs_component_available: bool,
) -> tuple[dict, str, dict, str, dict]:
    """Route separated vocals into Studio or Voice Changer."""
    if not audio_path:
        return (
            gr.update(),
            "",
            gr.update(),
            "No vocals stem is available.",
            gr.update(),
        )

    normalized_destination = str(destination or "Studio").strip()
    if normalized_destination == "Voice Changer":
        if audio_route_build is None:
            return (
                gr.update(),
                "",
                gr.update(),
                "Voice Changer routing is unavailable in this session.",
                gr.update(),
            )
        payload = audio_route_build(
            "voice_changer.source",
            audio_path,
            source_label="Singing Enhancements: Separated Vocals",
        )
        return (
            gr.update(),
            payload,
            gr.update(),
            "Loaded vocals into Voice Changer.",
            gr.update(selected="tab_voice_changer") if main_tabs_component_available else gr.update(),
        )

    return (
        gr.update(value=audio_path),
        "",
        gr.update(selected="singing_studio"),
        "Loaded vocals into Studio.",
        gr.update(),
    )


class SingingEnhancementsTool(Tool):
    """Singing-focused studio tab."""

    config = ToolConfig(
        name="Singing Enhancements",
        module_name="tool_singing_enhancements",
        description="Singing vocal effects, autotune, backing-track mixing, and source separation",
        enabled=True,
        category="generation",
    )

    @classmethod
    def create_tool(cls, shared_state):
        components = {}
        initial_audio_route_choices = shared_state.get("audio_route_get_initial_targets", lambda: [])()
        manager, initial_models, default_sep_goal, sep_model_choices, default_sep_model, sep_goal_help = (
            _get_initial_source_separation_ui(shared_state)
        )
        _ = manager
        _ = initial_models

        with gr.TabItem("Singing Enhancements", id="tab_singing_enhancements") as singing_tab:
            components["singing_tab"] = singing_tab
            gr.Markdown(
                "Voice studio for sung vocals. Use **Studio** for processing and mixing, or "
                "**Source Separation** to split a full song into vocal and backing stems first."
            )

            with gr.Tabs(selected="singing_studio") as singing_sections:
                components["singing_sections"] = singing_sections

                with gr.TabItem("Studio", id="singing_studio"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### Source Tracks")
                            components["source_audio"] = gr.Audio(
                                label="Lead Vocal",
                                type="filepath",
                                sources=["upload", "microphone"],
                                elem_id="singing-source-audio",
                            )
                            components["backing_audio"] = gr.Audio(
                                label="Backing Track (Optional)",
                                type="filepath",
                                sources=["upload"],
                                elem_id="singing-backing-audio",
                            )
                            components["waveform_selector"] = gr.HTML(SINGING_WAVEFORM_HTML)
                            components["selection_start_sec"] = gr.Textbox(
                                value="",
                                visible=False,
                                elem_id="singing-selection-start",
                            )
                            components["selection_end_sec"] = gr.Textbox(
                                value="",
                                visible=False,
                                elem_id="singing-selection-end",
                            )
                            components["selection_enabled"] = gr.Textbox(
                                value="false",
                                visible=False,
                                elem_id="singing-selection-enabled",
                            )

                            with gr.Accordion("Autotune", open=True):
                                with gr.Row():
                                    components["autotune_mode"] = gr.Dropdown(
                                        label="Mode",
                                        choices=["Off", "Auto", "Manual"],
                                        value="Auto",
                                        scale=1,
                                    )
                                    components["autotune_strength"] = gr.Slider(
                                        minimum=0,
                                        maximum=100,
                                        value=70,
                                        step=1,
                                        label="Strength",
                                        scale=2,
                                    )
                                    components["preserve_formants"] = gr.Checkbox(
                                        label="Preserve Formants",
                                        value=True,
                                        scale=1,
                                    )
                                    components["detect_key_btn"] = gr.Button(
                                        "Detect Key",
                                        variant="secondary",
                                        scale=1,
                                    )

                                with gr.Row(visible=False) as manual_key_row:
                                    components["autotune_tonic"] = gr.Dropdown(
                                        label="Tonic",
                                        choices=NOTE_NAMES,
                                        value="C",
                                    )
                                    components["autotune_scale"] = gr.Dropdown(
                                        label="Scale",
                                        choices=["major", "minor", "chromatic"],
                                        value="major",
                                    )
                                components["manual_key_row"] = manual_key_row

                            with gr.Accordion("Singing Effects", open=False):
                                components["pitch_shift_semitones"] = gr.Slider(
                                    minimum=-12.0,
                                    maximum=12.0,
                                    value=0.0,
                                    step=0.1,
                                    label="Creative Pitch Shift (semitones)",
                                )
                                with gr.Row():
                                    components["highpass_hz"] = gr.Slider(
                                        minimum=0,
                                        maximum=600,
                                        value=80,
                                        step=5,
                                        label="High-Pass Filter (Hz)",
                                    )
                                    components["lowpass_hz"] = gr.Slider(
                                        minimum=0,
                                        maximum=20000,
                                        value=0,
                                        step=50,
                                        label="Low-Pass Filter (Hz, 0 = off)",
                                    )
                                with gr.Row():
                                    components["compression_amount"] = gr.Slider(
                                        minimum=0,
                                        maximum=100,
                                        value=45,
                                        step=1,
                                        label="Compression",
                                    )
                                    components["echo_amount"] = gr.Slider(
                                        minimum=0,
                                        maximum=100,
                                        value=20,
                                        step=1,
                                        label="Echo",
                                    )
                                    components["reverb_amount"] = gr.Slider(
                                        minimum=0,
                                        maximum=100,
                                        value=35,
                                        step=1,
                                        label="Reverb",
                                    )
                                with gr.Row():
                                    components["vocal_gain_db"] = gr.Slider(
                                        minimum=-24,
                                        maximum=24,
                                        value=0,
                                        step=0.5,
                                        label="Vocal Gain (dB)",
                                    )
                                    components["backing_gain_db"] = gr.Slider(
                                        minimum=-24,
                                        maximum=24,
                                        value=-4,
                                        step=0.5,
                                        label="Backing Gain (dB)",
                                    )

                            with gr.Row():
                                components["process_btn"] = gr.Button(
                                    "Process Singing",
                                    variant="primary",
                                    size="lg",
                                )
                                components["stop_btn"] = gr.Button(
                                    "Stop",
                                    variant="stop",
                                    size="lg",
                                )
                            components["status"] = gr.Textbox(
                                label="Studio Status",
                                interactive=False,
                                lines=4,
                                max_lines=8,
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("### Outputs")
                            components["processed_vocal_audio"] = gr.Audio(
                                label="Processed Vocal",
                                type="filepath",
                                interactive=True,
                            )
                            processed_vocal_route = create_audio_route_controls(initial_audio_route_choices)
                            components["processed_vocal_route_target"] = processed_vocal_route["target_dropdown"]
                            components["processed_vocal_route_btn"] = processed_vocal_route["send_button"]
                            components["final_mix_audio"] = gr.Audio(
                                label="Final Mix (Vocal + Backing)",
                                type="filepath",
                                interactive=True,
                            )
                            final_mix_route = create_audio_route_controls(initial_audio_route_choices)
                            components["final_mix_route_target"] = final_mix_route["target_dropdown"]
                            components["final_mix_route_btn"] = final_mix_route["send_button"]

                            with gr.Row():
                                components["save_vocal_btn"] = gr.Button(
                                    "Save Vocal",
                                    variant="primary",
                                    interactive=False,
                                )
                                components["save_mix_btn"] = gr.Button(
                                    "Save Mix",
                                    variant="secondary",
                                    interactive=False,
                                )

                            components["temp_vocal_output_path"] = gr.State(value=None)
                            components["temp_mix_output_path"] = gr.State(value=None)
                            components["vocal_suggested_name"] = gr.State(value="")
                            components["mix_suggested_name"] = gr.State(value="")
                            components["vocal_metadata_text"] = gr.State(value="")
                            components["mix_metadata_text"] = gr.State(value="")

                with gr.TabItem("Source Separation", id="singing_source_separation") as sep_tab:
                    components["singing_source_separation_tab"] = sep_tab
                    gr.Markdown(
                        "Upload a mixed song, choose what you want out of it, then split it into "
                        "**Vocals** and **Backing/Instrumental**."
                    )
                    with gr.Row():
                        with gr.Column(scale=2):
                            components["sep_source_audio"] = gr.Audio(
                                label="Mixed Song",
                                type="filepath",
                                sources=["upload"],
                            )
                            components["sep_goal_select"] = gr.Radio(
                                label="What do you want?",
                                choices=list(SOURCE_SEPARATION_GOAL_CHOICES),
                                value=default_sep_goal,
                                info="This narrows the model list to the options that fit your goal.",
                            )
                            components["sep_goal_help"] = gr.Markdown(sep_goal_help)
                            components["sep_model_select"] = gr.Dropdown(
                                label="Model For This Goal",
                                choices=sep_model_choices,
                                value=default_sep_model,
                                interactive=bool(sep_model_choices),
                            )
                            components["sep_model_info"] = gr.Markdown(
                                manager.describe_model(default_sep_model) if default_sep_model else "No source-separation models available yet."
                            )

                            with gr.Accordion("Advanced", open=False):
                                components["sep_use_autocast"] = gr.Checkbox(
                                    label="Prefer Faster GPU Run",
                                    value=True,
                                    info="Usually safe on GPU. Turn it off only if a model behaves badly on your setup.",
                                )

                            with gr.Row():
                                components["sep_process_btn"] = gr.Button(
                                    "Split Audio",
                                    variant="primary",
                                    size="lg",
                                )
                                components["sep_stop_btn"] = gr.Button(
                                    "Stop",
                                    variant="stop",
                                    size="lg",
                                )
                            components["sep_status"] = gr.Textbox(
                                label="Source Separation Status",
                                interactive=False,
                                lines=4,
                                max_lines=8,
                            )

                        with gr.Column(scale=1):
                            components["sep_vocals_audio"] = gr.Audio(
                                label="Separated Vocals",
                                type="filepath",
                                interactive=True,
                            )
                            with gr.Row():
                                components["sep_vocals_destination"] = gr.Dropdown(
                                    label="Use Vocals In",
                                    choices=list(SOURCE_SEPARATION_VOCALS_DESTINATIONS),
                                    value="Studio",
                                    interactive=True,
                                    allow_custom_value=False,
                                    scale=2,
                                )
                                components["use_sep_vocals_btn"] = gr.Button(
                                    "Use Vocals",
                                    variant="secondary",
                                    interactive=False,
                                    scale=1,
                                )
                                components["save_sep_vocals_btn"] = gr.Button(
                                    "Save Vocals",
                                    variant="primary",
                                    interactive=False,
                                )
                            components["sep_backing_audio"] = gr.Audio(
                                label="Backing / Instrumental",
                                type="filepath",
                                interactive=True,
                            )
                            with gr.Row():
                                components["use_sep_backing_btn"] = gr.Button(
                                    "Use Backing in Studio",
                                    variant="secondary",
                                    interactive=False,
                                )
                                components["save_sep_backing_btn"] = gr.Button(
                                    "Save Backing",
                                    variant="secondary",
                                    interactive=False,
                                )

                            with gr.Accordion("Detailed Stems", open=False, visible=False) as detailed_stems_accordion:
                                components["sep_drums_audio"] = gr.Audio(
                                    label="Drums",
                                    type="filepath",
                                    interactive=True,
                                    visible=False,
                                )
                                components["sep_bass_audio"] = gr.Audio(
                                    label="Bass",
                                    type="filepath",
                                    interactive=True,
                                    visible=False,
                                )
                                components["sep_other_audio"] = gr.Audio(
                                    label="Other",
                                    type="filepath",
                                    interactive=True,
                                    visible=False,
                                )
                                components["sep_guitar_audio"] = gr.Audio(
                                    label="Guitar",
                                    type="filepath",
                                    interactive=True,
                                    visible=False,
                                )
                                components["sep_piano_audio"] = gr.Audio(
                                    label="Piano",
                                    type="filepath",
                                    interactive=True,
                                    visible=False,
                                )
                            components["sep_detailed_accordion"] = detailed_stems_accordion

                            components["save_sep_all_btn"] = gr.Button(
                                "Save All Stems",
                                variant="primary",
                                interactive=False,
                            )
                            components["sep_result_json"] = gr.State(value="")
                            components["sep_vocals_suggested_name"] = gr.State(value="")
                            components["sep_backing_suggested_name"] = gr.State(value="")
                            components["sep_all_suggested_name"] = gr.State(value="")

            components["existing_files_json"] = gr.State(value="[]")

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        get_tenant_output_dir = shared_state["get_tenant_output_dir"]
        get_tenant_paths = shared_state["get_tenant_paths"]
        input_trigger = shared_state["input_trigger"]
        show_input_modal_js = shared_state["show_input_modal_js"]
        play_completion_beep = shared_state.get("play_completion_beep")
        run_heavy_job = shared_state.get("run_heavy_job")
        audio_route_trigger = shared_state.get("audio_route_trigger")
        audio_route_build = shared_state.get("audio_route_build_payload")
        main_tabs_component = shared_state.get("main_tabs_component")
        source_separation_manager = _get_source_separation_manager_for_state(shared_state)

        def toggle_manual_controls(mode: str):
            is_manual = (mode or "").strip().lower() == "manual"
            return gr.update(visible=is_manual)

        def detect_key_handler(
            source_audio,
            sep_source_audio,
            sep_vocals_audio,
            sep_backing_audio,
        ):
            result = detect_key_from_available_audio(
                source_audio=source_audio,
                sep_source_audio=sep_source_audio,
                sep_vocals_audio=sep_vocals_audio,
                sep_backing_audio=sep_backing_audio,
            )
            mode_update = gr.update(value=result["mode"]) if result["mode"] else gr.update()
            tonic_update = gr.update(value=result["tonic"]) if result["tonic"] else gr.update()
            scale_update = gr.update(value=result["scale"]) if result["scale"] else gr.update()
            manual_row_update = gr.update(visible=bool(result["show_manual_controls"]))
            return (
                mode_update,
                tonic_update,
                scale_update,
                manual_row_update,
                result["status"],
            )

        components["autotune_mode"].change(
            toggle_manual_controls,
            inputs=[components["autotune_mode"]],
            outputs=[components["manual_key_row"]],
        )
        components["detect_key_btn"].click(
            detect_key_handler,
            inputs=[
                components["source_audio"],
                components["sep_source_audio"],
                components["sep_vocals_audio"],
                components["sep_backing_audio"],
            ],
            outputs=[
                components["autotune_mode"],
                components["autotune_tonic"],
                components["autotune_scale"],
                components["manual_key_row"],
                components["status"],
            ],
        )

        components["singing_tab"].select(
            fn=None,
            inputs=[components["source_audio"]],
            js=WAVEFORM_SELECTOR_JS,
        )
        components["source_audio"].change(
            fn=None,
            inputs=[components["source_audio"]],
            js=WAVEFORM_SELECTOR_JS,
            queue=False,
        )

        wire_audio_route_dropdown_refresh(
            components["singing_tab"],
            components["processed_vocal_route_target"],
            shared_state,
        )
        wire_audio_route_dropdown_refresh(
            components["singing_tab"],
            components["final_mix_route_target"],
            shared_state,
        )
        wire_audio_route_source(
            send_button=components["processed_vocal_route_btn"],
            target_dropdown=components["processed_vocal_route_target"],
            audio_value_component=components["processed_vocal_audio"],
            status_component=components["status"],
            shared_state=shared_state,
            source_label="Singing Enhancements: Processed Vocal",
        )
        wire_audio_route_source(
            send_button=components["final_mix_route_btn"],
            target_dropdown=components["final_mix_route_target"],
            audio_value_component=components["final_mix_audio"],
            status_component=components["status"],
            shared_state=shared_state,
            source_label="Singing Enhancements: Final Mix",
        )
        if audio_route_trigger is not None:
            wire_audio_route_listener(
                audio_route_trigger=audio_route_trigger,
                target_components={
                    "singing_enhancements.source": components["source_audio"],
                    "singing_enhancements.backing": components["backing_audio"],
                },
                status_component=components["status"],
            )

        def refresh_source_separation_model_choices(goal, current_model, force_refresh=False):
            try:
                models = source_separation_manager.list_models(refresh=bool(force_refresh))
            except Exception as exc:
                return (
                    gr.update(choices=[], value=None, interactive=False),
                    f"❌ {str(exc)}",
                    f"Source-separation models unavailable: {str(exc)}",
                )

            filtered_models = _filter_source_separation_models_for_goal(source_separation_manager, models, goal)
            choices = source_separation_manager.get_dropdown_choices(models=filtered_models)
            available_values = [value for _label, value in choices]
            next_model = current_model if current_model in available_values else (available_values[0] if available_values else None)
            info_text = source_separation_manager.describe_model(next_model)
            help_text = _build_source_separation_goal_help(goal, filtered_models)
            return (
                gr.update(choices=choices, value=next_model, interactive=bool(choices)),
                help_text,
                info_text,
            )

        def update_source_separation_model_info(model_filename):
            return source_separation_manager.describe_model(model_filename)

        def process_singing_handler(
            source_audio,
            backing_audio,
            selection_start_sec,
            selection_end_sec,
            selection_enabled,
            autotune_mode,
            autotune_strength,
            autotune_tonic,
            autotune_scale,
            preserve_formants,
            pitch_shift_semitones,
            highpass_hz,
            lowpass_hz,
            compression_amount,
            echo_amount,
            reverb_amount,
            vocal_gain_db,
            backing_gain_db,
            request: gr.Request = None,
            progress=gr.Progress(),
        ):
            if not source_audio:
                return (
                    None,
                    None,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    None,
                    None,
                    "",
                    "",
                    "",
                    "",
                    "Please upload or record a lead vocal track.",
                )

            selection = SegmentSelection(
                start_sec=_safe_float(selection_start_sec),
                end_sec=_safe_float(selection_end_sec),
                enabled=str(selection_enabled).strip().lower() == "true",
            )
            autotune_cfg = AutotuneConfig(
                mode=(autotune_mode or "Off").strip().lower(),
                strength=max(0.0, min(1.0, _safe_float(autotune_strength) / 100.0)),
                tonic=autotune_tonic,
                scale=autotune_scale,
                preserve_formants=bool(preserve_formants),
            )
            effects_cfg = SingingEffectConfig(
                pitch_shift_semitones=_safe_float(pitch_shift_semitones),
                highpass_hz=max(0.0, _safe_float(highpass_hz)),
                lowpass_hz=max(0.0, _safe_float(lowpass_hz)),
                compression_amount=max(0.0, min(1.0, _safe_float(compression_amount) / 100.0)),
                echo_amount=max(0.0, min(1.0, _safe_float(echo_amount) / 100.0)),
                reverb_amount=max(0.0, min(1.0, _safe_float(reverb_amount) / 100.0)),
            )
            mix_cfg = MixConfig(
                vocal_gain_db=_safe_float(vocal_gain_db),
                backing_gain_db=_safe_float(backing_gain_db),
            )

            tenant_paths = get_tenant_paths(request=request, strict=True)
            temp_dir = tenant_paths.temp_dir
            temp_dir.mkdir(parents=True, exist_ok=True)

            def _process_impl():
                result = process_singing_audio(
                    vocal_path=source_audio,
                    backing_path=backing_audio or None,
                    selection=selection,
                    autotune=autotune_cfg,
                    effects=effects_cfg,
                    mix=mix_cfg,
                    temp_dir=temp_dir,
                    progress_callback=progress,
                )

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_stem = sanitize_output_name(Path(str(source_audio)).stem) or "singing_take"
                vocal_name = f"singing_vocal_{safe_stem}"
                mix_name = f"singing_mix_{safe_stem}"

                vocal_metadata = _build_metadata_text(
                    timestamp=timestamp,
                    source_audio=source_audio,
                    backing_audio=backing_audio,
                    selection=result.selection,
                    autotune_result=result.autotune_result,
                    effects=effects_cfg,
                    mix=mix_cfg,
                    output_type="vocal",
                )
                mix_metadata = _build_metadata_text(
                    timestamp=timestamp,
                    source_audio=source_audio,
                    backing_audio=backing_audio,
                    selection=result.selection,
                    autotune_result=result.autotune_result,
                    effects=effects_cfg,
                    mix=mix_cfg,
                    output_type="mix",
                ) if result.mix_output_path else ""

                if play_completion_beep:
                    play_completion_beep()

                return (
                    result.vocal_output_path,
                    result.mix_output_path,
                    gr.update(interactive=True),
                    gr.update(interactive=bool(result.mix_output_path)),
                    result.vocal_output_path,
                    result.mix_output_path,
                    vocal_name,
                    mix_name if result.mix_output_path else "",
                    vocal_metadata,
                    mix_metadata,
                    result.status_message,
                )

            try:
                if run_heavy_job:
                    return run_heavy_job(
                        "singing_enhancements_process",
                        _process_impl,
                        request=request,
                    )
                return _process_impl()
            except MemoryAdmissionError as exc:
                return (
                    None,
                    None,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    None,
                    None,
                    "",
                    "",
                    "",
                    "",
                    f"⚠ Memory safety guard rejected request: {str(exc)}",
                )
            except Exception as exc:
                return (
                    None,
                    None,
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    None,
                    None,
                    "",
                    "",
                    "",
                    "",
                    f"❌ Error processing singing track: {str(exc)}",
                )

        def process_source_separation_handler(
            source_audio,
            model_filename,
            use_autocast,
            request: gr.Request = None,
            progress=gr.Progress(),
        ):
            if not source_audio:
                payload = build_source_separation_ui_payload(
                    None,
                    status_message="Please upload a mixed song or stem source first.",
                )
            elif not model_filename:
                payload = build_source_separation_ui_payload(
                    None,
                    status_message="Select a source-separation model first.",
                )
            else:
                print(
                    "[Source Separation UI] Request received: "
                    f"source_audio={source_audio} model={model_filename} use_autocast={bool(use_autocast)}",
                    flush=True,
                )
                try:
                    tenant_paths = get_tenant_paths(request=request, strict=True)
                    temp_dir = tenant_paths.temp_dir / "source_separation" / datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"[Source Separation UI] Working directory: {temp_dir}",
                        flush=True,
                    )

                    def _separate_impl():
                        result = source_separation_manager.separate_audio(
                            input_path=source_audio,
                            model_filename=model_filename,
                            output_dir=temp_dir,
                            use_autocast=bool(use_autocast),
                            normalization=0.9,
                            invert_spect=False,
                            progress_callback=progress,
                        )
                        if play_completion_beep:
                            play_completion_beep()
                        return build_source_separation_ui_payload(
                            result,
                            status_message=result.status_message,
                        )

                    if run_heavy_job:
                        payload = run_heavy_job(
                            "singing_source_separation_process",
                            _separate_impl,
                            request=request,
                        )
                    else:
                        payload = _separate_impl()
                except MemoryAdmissionError as exc:
                    print(
                        f"[Source Separation UI] Memory admission rejected: {exc}",
                        flush=True,
                    )
                    payload = build_source_separation_ui_payload(
                        None,
                        status_message=f"⚠ Memory safety guard rejected request: {str(exc)}",
                    )
                except Exception as exc:
                    print(
                        f"[Source Separation UI] Separation failed: {exc}\n{traceback.format_exc()}",
                        flush=True,
                    )
                    payload = build_source_separation_ui_payload(
                        None,
                        status_message=f"❌ Source separation failed: {str(exc)}",
                    )

            return (
                payload["vocals_audio"],
                payload["backing_audio"],
                payload["button_update"],
                payload["button_update"],
                payload["all_button_update"],
                payload["use_button_update"],
                payload["use_button_update"],
                payload["vocals_name"],
                payload["backing_name"],
                payload["all_name"],
                payload["result_json"],
                payload["status"],
                payload["detailed_updates"]["Drums"],
                payload["detailed_updates"]["Bass"],
                payload["detailed_updates"]["Other"],
                payload["detailed_updates"]["Guitar"],
                payload["detailed_updates"]["Piano"],
                payload["accordion_update"],
            )

        process_event = components["process_btn"].click(
            process_singing_handler,
            inputs=[
                components["source_audio"],
                components["backing_audio"],
                components["selection_start_sec"],
                components["selection_end_sec"],
                components["selection_enabled"],
                components["autotune_mode"],
                components["autotune_strength"],
                components["autotune_tonic"],
                components["autotune_scale"],
                components["preserve_formants"],
                components["pitch_shift_semitones"],
                components["highpass_hz"],
                components["lowpass_hz"],
                components["compression_amount"],
                components["echo_amount"],
                components["reverb_amount"],
                components["vocal_gain_db"],
                components["backing_gain_db"],
            ],
            outputs=[
                components["processed_vocal_audio"],
                components["final_mix_audio"],
                components["save_vocal_btn"],
                components["save_mix_btn"],
                components["temp_vocal_output_path"],
                components["temp_mix_output_path"],
                components["vocal_suggested_name"],
                components["mix_suggested_name"],
                components["vocal_metadata_text"],
                components["mix_metadata_text"],
                components["status"],
            ],
        )

        components["stop_btn"].click(
            lambda: (
                "Processing stopped.",
                gr.update(interactive=False),
                gr.update(interactive=False),
            ),
            inputs=[],
            outputs=[
                components["status"],
                components["save_vocal_btn"],
                components["save_mix_btn"],
            ],
            cancels=[process_event],
            queue=False,
        )

        sep_process_event = components["sep_process_btn"].click(
            process_source_separation_handler,
            inputs=[
                components["sep_source_audio"],
                components["sep_model_select"],
                components["sep_use_autocast"],
            ],
            outputs=[
                components["sep_vocals_audio"],
                components["sep_backing_audio"],
                components["save_sep_vocals_btn"],
                components["save_sep_backing_btn"],
                components["save_sep_all_btn"],
                components["use_sep_vocals_btn"],
                components["use_sep_backing_btn"],
                components["sep_vocals_suggested_name"],
                components["sep_backing_suggested_name"],
                components["sep_all_suggested_name"],
                components["sep_result_json"],
                components["sep_status"],
                components["sep_drums_audio"],
                components["sep_bass_audio"],
                components["sep_other_audio"],
                components["sep_guitar_audio"],
                components["sep_piano_audio"],
                components["sep_detailed_accordion"],
            ],
        )

        components["sep_stop_btn"].click(
            lambda: ("Source separation stopped.",),
            inputs=[],
            outputs=[components["sep_status"]],
            cancels=[sep_process_event],
            queue=False,
        )

        if audio_route_trigger is not None and main_tabs_component is not None:
            components["use_sep_vocals_btn"].click(
                lambda audio_path, destination: build_source_separation_vocals_route_update(
                    audio_path=audio_path,
                    destination=destination,
                    audio_route_build=audio_route_build,
                    main_tabs_component_available=True,
                ),
                inputs=[components["sep_vocals_audio"], components["sep_vocals_destination"]],
                outputs=[
                    components["source_audio"],
                    audio_route_trigger,
                    components["singing_sections"],
                    components["sep_status"],
                    main_tabs_component,
                ],
            )
        elif audio_route_trigger is not None:
            components["use_sep_vocals_btn"].click(
                lambda audio_path, destination: build_source_separation_vocals_route_update(
                    audio_path=audio_path,
                    destination=destination,
                    audio_route_build=audio_route_build,
                    main_tabs_component_available=False,
                )[:4],
                inputs=[components["sep_vocals_audio"], components["sep_vocals_destination"]],
                outputs=[
                    components["source_audio"],
                    audio_route_trigger,
                    components["singing_sections"],
                    components["sep_status"],
                ],
            )
        else:
            components["use_sep_vocals_btn"].click(
                lambda audio_path, destination: (
                    gr.update(value=audio_path) if destination != "Voice Changer" and audio_path else gr.update(),
                    gr.update(selected="singing_studio") if destination != "Voice Changer" and audio_path else gr.update(),
                    (
                        "Loaded vocals into Studio."
                        if destination != "Voice Changer" and audio_path
                        else "Voice Changer routing is unavailable in this session."
                    ),
                ),
                inputs=[components["sep_vocals_audio"], components["sep_vocals_destination"]],
                outputs=[components["source_audio"], components["singing_sections"], components["sep_status"]],
            )
        components["use_sep_backing_btn"].click(
            lambda audio_path: build_source_separation_studio_route_update(audio_path, "Backing"),
            inputs=[components["sep_backing_audio"]],
            outputs=[components["backing_audio"], components["singing_sections"], components["sep_status"]],
        )

        components["singing_source_separation_tab"].select(
            lambda goal, current_model: refresh_source_separation_model_choices(goal, current_model, force_refresh=True),
            inputs=[
                components["sep_goal_select"],
                components["sep_model_select"],
            ],
            outputs=[
                components["sep_model_select"],
                components["sep_goal_help"],
                components["sep_model_info"],
            ],
        )

        components["sep_goal_select"].change(
            refresh_source_separation_model_choices,
            inputs=[
                components["sep_goal_select"],
                components["sep_model_select"],
            ],
            outputs=[
                components["sep_model_select"],
                components["sep_goal_help"],
                components["sep_model_info"],
            ],
        )
        components["sep_model_select"].change(
            update_source_separation_model_info,
            inputs=[components["sep_model_select"]],
            outputs=[components["sep_model_info"]],
        )

        def get_existing_files(request: gr.Request):
            output_dir = get_tenant_output_dir(request=request, strict=True)
            return json.dumps(get_existing_wav_stems(output_dir))

        def build_save_modal_js(modal_js: str) -> str:
            return f"""
            (existingFilesJson, suggestedName) => {{
                try {{
                    window.inputModalExistingFiles = JSON.parse(existingFilesJson || '[]');
                }} catch (error) {{
                    window.inputModalExistingFiles = [];
                }}
                const openModal = {modal_js};
                openModal(suggestedName);
            }}
            """

        save_vocal_modal_js = show_input_modal_js(
            title="Save Processed Vocal",
            message="Enter a filename for the processed vocal track:",
            placeholder="e.g., singing_vocal_take_01",
            context="save_singing_vocal_",
        )
        save_mix_modal_js = show_input_modal_js(
            title="Save Final Mix",
            message="Enter a filename for the vocal + backing mix:",
            placeholder="e.g., singing_mix_take_01",
            context="save_singing_mix_",
        )
        save_sep_vocals_modal_js = show_input_modal_js(
            title="Save Separated Vocals",
            message="Enter a filename for the separated vocals:",
            placeholder="e.g., singing_sep_vocals_take_01",
            context="save_singing_sep_vocals_",
        )
        save_sep_backing_modal_js = show_input_modal_js(
            title="Save Backing / Instrumental",
            message="Enter a filename for the separated backing track:",
            placeholder="e.g., singing_sep_backing_take_01",
            context="save_singing_sep_backing_",
        )
        save_sep_all_modal_js = show_input_modal_js(
            title="Save All Stems",
            message="Enter a base filename for the separated stem bundle:",
            placeholder="e.g., singing_sep_song_take_01",
            context="save_singing_sep_all_",
        )

        components["save_vocal_btn"].click(
            fn=get_existing_files,
            inputs=[],
            outputs=[components["existing_files_json"]],
        ).then(
            fn=None,
            inputs=[components["existing_files_json"], components["vocal_suggested_name"]],
            js=build_save_modal_js(save_vocal_modal_js),
        )

        components["save_mix_btn"].click(
            fn=get_existing_files,
            inputs=[],
            outputs=[components["existing_files_json"]],
        ).then(
            fn=None,
            inputs=[components["existing_files_json"], components["mix_suggested_name"]],
            js=build_save_modal_js(save_mix_modal_js),
        )

        components["save_sep_vocals_btn"].click(
            fn=get_existing_files,
            inputs=[],
            outputs=[components["existing_files_json"]],
        ).then(
            fn=None,
            inputs=[components["existing_files_json"], components["sep_vocals_suggested_name"]],
            js=build_save_modal_js(save_sep_vocals_modal_js),
        )

        components["save_sep_backing_btn"].click(
            fn=get_existing_files,
            inputs=[],
            outputs=[components["existing_files_json"]],
        ).then(
            fn=None,
            inputs=[components["existing_files_json"], components["sep_backing_suggested_name"]],
            js=build_save_modal_js(save_sep_backing_modal_js),
        )

        components["save_sep_all_btn"].click(
            fn=get_existing_files,
            inputs=[],
            outputs=[components["existing_files_json"]],
        ).then(
            fn=None,
            inputs=[components["existing_files_json"], components["sep_all_suggested_name"]],
            js=build_save_modal_js(save_sep_all_modal_js),
        )

        def handle_save_modal(
            input_value,
            vocal_audio,
            mix_audio,
            temp_vocal_output_path,
            temp_mix_output_path,
            vocal_metadata_text,
            mix_metadata_text,
            sep_result_json,
            request: gr.Request,
        ):
            output_dir = get_tenant_output_dir(request=request, strict=True)

            def _no_update():
                return (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            matched, cancelled, chosen_name = parse_modal_submission(input_value, "save_singing_vocal_")
            if matched:
                if cancelled:
                    return _no_update()
                if not chosen_name or not chosen_name.strip():
                    return (
                        "❌ Please enter a filename for the vocal output.",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                try:
                    audio_to_save = vocal_audio or temp_vocal_output_path
                    if not audio_to_save:
                        return (
                            "❌ No processed vocal to save.",
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )
                    output_path = save_generated_output(
                        audio_value=audio_to_save,
                        output_dir=output_dir,
                        raw_name=chosen_name,
                        metadata_text=vocal_metadata_text,
                        default_sample_rate=44_100,
                    )
                    return (
                        f"Saved vocal output as {output_path.name}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                except Exception as exc:
                    return (
                        f"❌ Save failed: {str(exc)}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )

            matched, cancelled, chosen_name = parse_modal_submission(input_value, "save_singing_mix_")
            if matched:
                if cancelled:
                    return _no_update()
                if not chosen_name or not chosen_name.strip():
                    return (
                        "❌ Please enter a filename for the mix output.",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                try:
                    audio_to_save = mix_audio or temp_mix_output_path
                    if not audio_to_save:
                        return (
                            "❌ No final mix to save.",
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                        )
                    output_path = save_generated_output(
                        audio_value=audio_to_save,
                        output_dir=output_dir,
                        raw_name=chosen_name,
                        metadata_text=mix_metadata_text,
                        default_sample_rate=44_100,
                    )
                    return (
                        f"Saved mix output as {output_path.name}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                except Exception as exc:
                    return (
                        f"❌ Save failed: {str(exc)}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )

            sep_result = deserialize_source_separation_result(sep_result_json)

            matched, cancelled, chosen_name = parse_modal_submission(input_value, "save_singing_sep_vocals_")
            if matched:
                if cancelled:
                    return _no_update()
                if sep_result is None:
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        "❌ No source-separation result is available.",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                try:
                    output_path = save_source_separation_outputs(
                        sep_result,
                        output_dir,
                        chosen_name,
                        stem_name="Vocals",
                    )[0]
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        f"Saved separated vocals as {output_path.name}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                except Exception as exc:
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        f"❌ Save failed: {str(exc)}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )

            matched, cancelled, chosen_name = parse_modal_submission(input_value, "save_singing_sep_backing_")
            if matched:
                if cancelled:
                    return _no_update()
                if sep_result is None:
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        "❌ No source-separation result is available.",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                try:
                    output_path = save_source_separation_outputs(
                        sep_result,
                        output_dir,
                        chosen_name,
                        stem_name="Backing",
                    )[0]
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        f"Saved backing stem as {output_path.name}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                except Exception as exc:
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        f"❌ Save failed: {str(exc)}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )

            matched, cancelled, chosen_name = parse_modal_submission(input_value, "save_singing_sep_all_")
            if matched:
                if cancelled:
                    return _no_update()
                if sep_result is None:
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        "❌ No source-separation result is available.",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                try:
                    saved_paths = save_source_separation_outputs(
                        sep_result,
                        output_dir,
                        chosen_name,
                        stem_name=None,
                    )
                    names = ", ".join(path.name for path in saved_paths)
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        f"Saved stem bundle: {names}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )
                except Exception as exc:
                    return (
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        f"❌ Save failed: {str(exc)}",
                        gr.update(),
                        gr.update(),
                        gr.update(),
                    )

            return _no_update()

        input_trigger.change(
            handle_save_modal,
            inputs=[
                input_trigger,
                components["processed_vocal_audio"],
                components["final_mix_audio"],
                components["temp_vocal_output_path"],
                components["temp_mix_output_path"],
                components["vocal_metadata_text"],
                components["mix_metadata_text"],
                components["sep_result_json"],
            ],
            outputs=[
                components["status"],
                components["save_vocal_btn"],
                components["save_mix_btn"],
                components["sep_status"],
                components["save_sep_vocals_btn"],
                components["save_sep_backing_btn"],
                components["save_sep_all_btn"],
            ],
        )


get_tool_class = lambda: SingingEnhancementsTool


if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone

    run_tool_standalone(
        SingingEnhancementsTool,
        port=7871,
        title="Singing Enhancements - Standalone",
    )
