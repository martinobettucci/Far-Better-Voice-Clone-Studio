"""
Conversation Tab

Create conversations using VibeVoice, Qwen Base, or Qwen Speakers.
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "modules"))
import gradio as gr
import soundfile as sf
import torch
import numpy as np
import random
import re
import json
from datetime import datetime
from pathlib import Path
from modules.core_components.ai_models.model_utils import (
    set_seed, get_device, resolve_model_source
)
from textwrap import dedent

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager
from modules.core_components.ui_components.prompt_assistant import (
    create_prompt_assistant,
    wire_prompt_assistant_events,
    wire_prompt_apply_listener,
)
from modules.core_components.ui_components.audio_routing import (
    create_audio_route_controls,
    wire_audio_route_dropdown_refresh,
    wire_audio_route_source,
)
from modules.core_components.tools.generated_output_save import (
    get_existing_wav_stems,
    parse_modal_submission,
    save_generated_output,
)
from modules.core_components.tools.output_audio_pipeline import (
    OutputAudioPipelineConfig,
    apply_generation_output_pipeline,
)
from modules.core_components.tools.live_audio_streaming import LiveAudioChunkWriter, normalize_audio_chunk
from modules.core_components.tools.live_stream_policy import (
    conversation_stream_supported,
    prefix_non_stream_status,
)
from modules.core_components.runtime import MemoryAdmissionError


class ConversationTool(Tool):
    """Conversation tool implementation."""

    config = ToolConfig(
        name="Conversation",
        module_name="tool_conversation",
        description="Create multi-speaker conversations",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Conversation tool UI."""
        components = {}

        # Get helper functions and config
        format_help_html = shared_state['format_help_html']
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        create_vibevoice_advanced_params = shared_state['create_vibevoice_advanced_params']
        create_qwen_advanced_params = shared_state['create_qwen_advanced_params']
        create_luxtts_advanced_params = shared_state['create_luxtts_advanced_params']
        create_emotion_intensity_slider = shared_state['create_emotion_intensity_slider']
        _user_config = shared_state['_user_config']
        LANGUAGES = shared_state['LANGUAGES']
        MODEL_SIZES_CUSTOM = shared_state['MODEL_SIZES_CUSTOM']
        MODEL_SIZES_BASE = shared_state['MODEL_SIZES_BASE']
        MODEL_SIZES_VIBEVOICE = shared_state['MODEL_SIZES_VIBEVOICE']
        CUSTOM_VOICE_SPEAKERS = shared_state['CUSTOM_VOICE_SPEAKERS']
        _active_emotions = shared_state.get('_active_emotions', {})
        TTS_ENGINES = shared_state.get('TTS_ENGINES', {})
        expert_visible = bool(_user_config.get("conversation_show_expert_params", False))
        initial_cb_min_p = float(_user_config.get("chatterbox_min_p", 0.05))
        initial_cb_max_new_tokens = int(_user_config.get("chatterbox_max_new_tokens", 2048))
        initial_audio_route_choices = shared_state.get("audio_route_get_initial_targets", lambda: [])()

        # Mapping: conversation radio label -> engine key in TTS_ENGINES
        CONV_ENGINE_MAP = {
            "VibeVoice": "VibeVoice",
            "Qwen Base": "Qwen3",
            "Qwen Speakers": "Qwen3",
            "Chatterbox": "Chatterbox",
            "LuxTTS": "LuxTTS",
        }
        ALL_CONV_CHOICES = ["VibeVoice", "Qwen Base", "Qwen Speakers", "Chatterbox", "LuxTTS"]

        # Filter conversation choices based on enabled engines
        engine_settings = _user_config.get("enabled_engines", {})
        visible_choices = []
        for choice in ALL_CONV_CHOICES:
            engine_key = CONV_ENGINE_MAP.get(choice)
            if engine_key:
                engine_info = TTS_ENGINES.get(engine_key, {})
                if engine_settings.get(engine_key, engine_info.get("default_enabled", True)):
                    visible_choices.append(choice)
        if not visible_choices:
            visible_choices = ALL_CONV_CHOICES  # Safety fallback

        # Model selector at top
        initial_conv_model = _user_config.get("conv_model_type", "VibeVoice")
        if initial_conv_model not in visible_choices:
            initial_conv_model = visible_choices[0]
        is_vibevoice = initial_conv_model == "VibeVoice"
        is_qwen_base = initial_conv_model == "Qwen Base"
        is_qwen_custom = initial_conv_model == "Qwen Speakers"
        is_luxtts = initial_conv_model == "LuxTTS"
        is_chatterbox = initial_conv_model == "Chatterbox"
        deepfilter_available = bool(shared_state.get("DEEPFILTER_AVAILABLE", False))

        with gr.TabItem("Conversation", id="tab_conversation") as conv_tab:
            components['conv_tab'] = conv_tab
            gr.Markdown("Choose a model and create multi-speaker conversations with your custom voices")
            components['conv_model_type'] = gr.Radio(
                choices=visible_choices,
                value=initial_conv_model,
                show_label=False,
                container=False
            )

            # Get sample choices once for all dropdowns
            conversation_available_samples = get_sample_choices()
            conversation_first_sample = conversation_available_samples[0] if conversation_available_samples else None

            with gr.Row():
                # Left - Script input and model-specific controls
                with gr.Column(scale=2):
                    gr.Markdown("### Conversation Script")

                    components['conversation_script'] = gr.Textbox(
                        label="Script:",
                        placeholder=dedent("""\
                            Use [N]: format for speaker labels (1-4 for VibeVoice, 1-8 for Base/LuxTTS, 1-9 for CustomVoice).
                            Qwen also supports (style) for emotions:

                            [1]: (cheerful) Hey, how's it going?
                            [2]: (excited) I'm doing great, thanks for asking!
                            [1]: That's wonderful to hear.
                            [3]: (curious) Mind if I join this conversation?

                            VibeVoice: Natural long-form generation.
                            Base: Your custom voice clips with advanced pause control, with hacked Style control.
                            CustomVoice: Qwen Preset speakers with style control and Pause Controls.
                            LuxTTS: Voice cloning with custom samples, sequential generation."""),
                        lines=18
                    )
                    components['prompt_assistant'] = create_prompt_assistant(
                        shared_state=shared_state,
                        target_ids=["conversation.script"],
                        default_target_id="conversation.script",
                    )

                    # Qwen speaker mapping
                    speaker_guide = dedent("""\
                        **Qwen Speaker Numbers → Preset Voices:**

                        | # | Speaker | Voice | Language |   | # | Speaker | Voice | Language |
                        |---|---------|-------|----------|---|---|---------|-------|----------|
                        | 1 | Vivian | Bright young female | 🇨🇳 Chinese |   | 6 | Ryan | Dynamic male | 🇺🇸 English |
                        | 2 | Serena | Warm gentle female | 🇨🇳 Chinese |   | 7 | Aiden | Sunny American male | 🇺🇸 English |
                        | 3 | Uncle_Fu | Seasoned mellow male | 🇨🇳 Chinese |   | 8 | Ono_Anna | Playful female | 🇯🇵 Japanese |
                        | 4 | Dylan | Youthful Beijing male | 🇨🇳 Chinese |   | 9 | Sohee | Warm female | 🇰🇷 Korean |
                        | 5 | Eric | Lively Chengdu male | 🇨🇳 Chinese |  |  |  |  |  |

                        *Each speaker works best in their native language.*
                        """)

                    components['qwen_speaker_table'] = gr.HTML(
                        value=format_help_html(speaker_guide),
                        container=True,
                        padding=True,
                        visible=is_qwen_custom
                    )

                    # Shared voice sample selectors (used by Qwen Base, LuxTTS, Chatterbox, VibeVoice)
                    uses_samples = is_qwen_base or is_luxtts or is_chatterbox or is_vibevoice
                    components['shared_voices_section'] = gr.Column(visible=uses_samples)
                    with components['shared_voices_section']:
                        gr.Markdown("### Voice Samples (Up to 8 Speakers)")

                        with gr.Row():
                            with gr.Column():
                                components['conv_voice_1'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[1] Voice Sample",
                                    info="Select from your prepared samples"
                                )
                            with gr.Column():
                                components['conv_voice_2'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[2] Voice Sample",
                                    info="Select from your prepared samples"
                                )

                        with gr.Row():
                            with gr.Column():
                                components['conv_voice_3'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[3] Voice Sample",
                                    info="Select from your prepared samples"
                                )
                            with gr.Column():
                                components['conv_voice_4'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[4] Voice Sample",
                                    info="Select from your prepared samples"
                                )

                        # Rows 5-8 hidden for VibeVoice (only supports 4 speakers)
                        components['conv_voices_row_5_6'] = gr.Row(visible=not is_vibevoice)
                        with components['conv_voices_row_5_6']:
                            with gr.Column():
                                components['conv_voice_5'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[5] Voice Sample",
                                    info="Select from your prepared samples"
                                )
                            with gr.Column():
                                components['conv_voice_6'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[6] Voice Sample",
                                    info="Select from your prepared samples"
                                )

                        components['conv_voices_row_7_8'] = gr.Row(visible=not is_vibevoice)
                        with components['conv_voices_row_7_8']:
                            with gr.Column():
                                components['conv_voice_7'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[7] Voice Sample",
                                    info="Select from your prepared samples"
                                )
                            with gr.Column():
                                components['conv_voice_8'] = gr.Dropdown(
                                    choices=conversation_available_samples,
                                    value=conversation_first_sample,
                                    label="[8] Voice Sample",
                                    info="Select from your prepared samples"
                                )

                # Right - Settings and output
                with gr.Column(scale=1):
                    gr.Markdown("### Settings")

                    # Qwen Speakers settings
                    components['qwen_custom_settings'] = gr.Column(visible=is_qwen_custom)
                    with components['qwen_custom_settings']:
                        components['conv_model_size'] = gr.Dropdown(
                            choices=MODEL_SIZES_CUSTOM,
                            value=_user_config.get("conv_model_size", "Large"),
                            label="Model Size",
                            info="Small = Faster, Large = Better Quality"
                        )

                    # Qwen Base settings
                    components['qwen_base_settings'] = gr.Column(visible=is_qwen_base)
                    with components['qwen_base_settings']:
                        components['conv_base_model_size'] = gr.Dropdown(
                            choices=MODEL_SIZES_BASE,
                            value=_user_config.get("conv_base_model_size", "Small"),
                            label="Model Size",
                            info="Small = Faster, Large = Better Quality"
                        )

                    # Shared Language and Seed
                    is_qwen_initial = is_qwen_base or is_qwen_custom
                    from modules.core_components.constants import CHATTERBOX_LANGUAGES
                    if is_qwen_initial:
                        initial_lang_choices = LANGUAGES
                        initial_lang_value = _user_config.get("language", "Auto")
                    elif is_chatterbox:
                        initial_lang_choices = CHATTERBOX_LANGUAGES
                        initial_lang_value = "English"
                    else:
                        initial_lang_choices = ["Auto"]
                        initial_lang_value = "Auto"
                    with gr.Column():
                        with gr.Row():
                            components['conv_language'] = gr.Dropdown(
                                scale=5,
                                choices=initial_lang_choices,
                                value=initial_lang_value,
                                label="Language",
                                info="Language for all lines (Auto recommended)"
                            )
                            components['conv_seed'] = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                                info="(-1 for random)"
                            )

                    components['conv_show_expert_params'] = gr.Checkbox(
                        label="Show Expert Parameters",
                        value=expert_visible,
                        info="Reveal advanced controls for selected engines",
                    )
                    components['conv_expert_note'] = gr.Markdown(
                        "Expert options are currently available for Chatterbox mode only.",
                        visible=(expert_visible and not is_chatterbox),
                    )

                    # Shared Pause Controls
                    components['qwen_pause_controls'] = gr.Accordion("Pause Controls", open=False, visible=(is_qwen_custom or is_qwen_base))
                    with components['qwen_pause_controls']:
                        with gr.Column():
                            components['conv_pause_linebreak'] = gr.Slider(
                                minimum=0.0,
                                maximum=3.0,
                                value=_user_config.get("conv_pause_linebreak", 0.25),
                                step=0.05,
                                label="Pause Between Lines",
                                info="Silence between each speaker turn"
                            )

                            with gr.Row():
                                components['conv_pause_period'] = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=_user_config.get("conv_pause_period", 0.4),
                                    step=0.1,
                                    label="After Period (.)",
                                    info="Pause after periods"
                                )
                                components['conv_pause_comma'] = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=_user_config.get("conv_pause_comma", 0.2),
                                    step=0.1,
                                    label="After Comma (,)",
                                    info="Pause after commas"
                                )

                            with gr.Row():
                                components['conv_pause_question'] = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=_user_config.get("conv_pause_question", 0.6),
                                    step=0.1,
                                    label="After Question (?)",
                                    info="Pause after questions"
                                )
                                components['conv_pause_hyphen'] = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=_user_config.get("conv_pause_hyphen", 0.3),
                                    step=0.1,
                                    label="After Hyphen (-)",
                                    info="Pause after hyphens"
                                )

                    # Chatterbox-specific settings
                    components['cb_settings'] = gr.Column(visible=is_chatterbox)
                    with components['cb_settings']:
                        # Pause between lines
                        components['cb_pause_linebreak'] = gr.Slider(
                            minimum=0.0,
                            maximum=3.0,
                            value=0.25,
                            step=0.05,
                            label="Pause Between Lines",
                            info="Silence between each speaker turn"
                        )

                        # Chatterbox Advanced Parameters
                        create_chatterbox_advanced_params = shared_state['create_chatterbox_advanced_params']
                        cb_conv_params = create_chatterbox_advanced_params(
                            visible=True,
                            include_expert=True,
                            initial_min_p=initial_cb_min_p,
                            initial_max_new_tokens=initial_cb_max_new_tokens,
                            expert_visible=(expert_visible and is_chatterbox),
                        )
                        components['cb_conv_exaggeration'] = cb_conv_params['exaggeration']
                        components['cb_conv_cfg_weight'] = cb_conv_params['cfg_weight']
                        components['cb_conv_temperature'] = cb_conv_params['temperature']
                        components['cb_conv_repetition_penalty'] = cb_conv_params['repetition_penalty']
                        components['cb_conv_top_p'] = cb_conv_params['top_p']
                        components['cb_conv_min_p'] = cb_conv_params['min_p']
                        components['cb_conv_max_new_tokens'] = cb_conv_params['max_new_tokens']
                        components['cb_conv_expert_row'] = cb_conv_params['expert_row']
                        components['cb_conv_accordion'] = cb_conv_params['accordion']

                    # LuxTTS-specific settings
                    components['luxtts_settings'] = gr.Column(visible=is_luxtts)
                    with components['luxtts_settings']:
                        # Pause between lines
                        components['luxtts_pause_linebreak'] = gr.Slider(
                            minimum=0.0,
                            maximum=3.0,
                            value=0.25,
                            step=0.05,
                            label="Pause Between Lines",
                            info="Silence between each speaker turn"
                        )

                        # LuxTTS Advanced Parameters
                        lux_conv_params = create_luxtts_advanced_params(visible=True)
                        components['lux_conv_num_steps'] = lux_conv_params['num_steps']
                        components['lux_conv_t_shift'] = lux_conv_params['t_shift']
                        components['lux_conv_speed'] = lux_conv_params['speed']
                        components['lux_conv_guidance_scale'] = lux_conv_params['guidance_scale']
                        components['lux_conv_rms'] = lux_conv_params['rms']
                        components['lux_conv_ref_duration'] = lux_conv_params['ref_duration']
                        components['lux_conv_return_smooth'] = lux_conv_params['return_smooth']

                    # VibeVoice-specific settings
                    components['vibevoice_settings'] = gr.Column(visible=is_vibevoice)
                    with components['vibevoice_settings']:
                        components['longform_model_size'] = gr.Dropdown(
                            choices=MODEL_SIZES_VIBEVOICE,
                            value=_user_config.get("vibevoice_model_size", "Large"),
                            label="Model Size",
                            info="Small = Faster, Large = Better Quality"
                        )

                        # VibeVoice Advanced Parameters
                        vv_conv_params = create_vibevoice_advanced_params(
                            initial_num_steps=20,
                            initial_cfg_scale=3.0,
                            include_sentences_per_chunk=True,
                            visible=True
                        )
                        components['vv_conv_num_steps'] = vv_conv_params['num_steps']
                        components['longform_cfg_scale'] = vv_conv_params['cfg_scale']
                        components['vv_conv_do_sample'] = vv_conv_params['do_sample']
                        components['vv_conv_sentences_per_chunk'] = vv_conv_params['sentences_per_chunk']
                        components['vv_conv_repetition_penalty'] = vv_conv_params['repetition_penalty']
                        components['vv_conv_temperature'] = vv_conv_params['temperature']
                        components['vv_conv_top_k'] = vv_conv_params['top_k']
                        components['vv_conv_top_p'] = vv_conv_params['top_p']

                    # Qwen Advanced Parameters
                    components['qwen_conv_advanced'] = gr.Column(visible=(is_qwen_custom or is_qwen_base))
                    with components['qwen_conv_advanced']:
                        # Emotion intensity slider
                        components['conv_emotion_intensity_row'] = gr.Row(visible=is_qwen_base)
                        with components['conv_emotion_intensity_row']:
                            components['conv_emotion_intensity'] = create_emotion_intensity_slider(
                                initial_intensity=1.0,
                                label="Emotion Intensity",
                                visible=is_qwen_base
                            )

                        # Qwen advanced parameters
                        qwen_conv_params = create_qwen_advanced_params(
                            include_emotion=False,
                            visible=(is_qwen_custom or is_qwen_base)
                        )
                        components['qwen_conv_do_sample'] = qwen_conv_params['do_sample']
                        components['qwen_conv_temperature'] = qwen_conv_params['temperature']
                        components['qwen_conv_top_k'] = qwen_conv_params['top_k']
                        components['qwen_conv_top_p'] = qwen_conv_params['top_p']
                        components['qwen_conv_repetition_penalty'] = qwen_conv_params['repetition_penalty']
                        components['qwen_conv_max_new_tokens'] = qwen_conv_params['max_new_tokens']

                    # Shared settings
                    with gr.Row():
                        components['conv_generate_btn'] = gr.Button("Generate Conversation", variant="primary", size="lg")
                        components['conv_stop_btn'] = gr.Button("Stop", variant="stop", size="lg")

                    components['conv_output_audio'] = gr.Audio(
                        label="Generated Conversation",
                        type="filepath",
                        interactive=False,
                        streaming=True,
                        autoplay=True,
                    )
                    with gr.Row():
                        components['conv_output_enable_denoise'] = gr.Checkbox(
                            label="Enable Denoise",
                            value=False,
                            visible=deepfilter_available,
                        )
                        components['conv_output_enable_remove_silences'] = gr.Checkbox(
                            label="Remove Silences",
                            value=False,
                        )
                        components['conv_output_enable_normalize'] = gr.Checkbox(
                            label="Enable Normalize",
                            value=False,
                        )
                        components['conv_output_enable_mono'] = gr.Checkbox(
                            label="Enable Mono",
                            value=False,
                        )
                        components['conv_output_apply_pipeline_btn'] = gr.Button(
                            "Apply Pipeline",
                            variant="secondary",
                            size="sm",
                        )
                    components['conv_save_btn'] = gr.Button("Save to Output", variant="primary", interactive=False)
                    route_controls = create_audio_route_controls(initial_audio_route_choices)
                    components['conv_route_target'] = route_controls['target_dropdown']
                    components['conv_route_btn'] = route_controls['send_button']
                    components['conv_suggested_name'] = gr.State(value="")
                    components['conv_metadata_text'] = gr.State(value="")
                    components['conv_output_audio_path'] = gr.State(value="")
                    components['conv_existing_files_json'] = gr.State(value="[]")
                    components['conv_status'] = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=5)

                    # Model-specific tips
                    qwen_custom_tips_text = dedent("""\
                    **Qwen Speakers Tips:**
                    - Fast generation with preset voices
                    - Up to 9 different speakers
                    - Tip: Use `[break=1.5]` inline for custom pauses
                    - Each voice optimized for their native language
                    - Style instructions: (cheerful), (sad), (excited), etc.
                    """)

                    qwen_base_tips_text = dedent("""\
                    **Qwen Base Tips:**
                    - Use your own custom voice samples
                    - Up to 8 different speakers
                    - Tip: Use `[break=1.5]` inline for custom pauses
                    - Advanced pause control (periods, commas, questions, hyphens)
                    - Prepare 3-10 second voice samples in Library Manager
                    """)

                    vibevoice_tips_text = dedent("""\
                    **VibeVoice Tips:**
                    - Up to 90 minutes continuous generation
                    - Up to 4 speakers with custom voices
                    - May spontaneously add background music/sounds
                    - Longer scripts work best with Large model
                    - Natural conversation flow (no manual pause control)
                    """)

                    luxtts_tips_text = dedent("""\
                    **LuxTTS Tips:**
                    - Up to 8 speakers with custom voice samples
                    - Sequential generation: each line generated individually then stitched
                    - Voice prompts are cached per speaker for fast subsequent lines
                    - 48kHz output for high quality audio
                    - No style/emotion control (use Qwen for that)
                    - Adjust ref_duration if voice quality is poor
                    """)

                    components['qwen_custom_tips'] = gr.HTML(
                        value=format_help_html(qwen_custom_tips_text),
                        container=True,
                        padding=True,
                        visible=is_qwen_custom
                    )

                    components['qwen_base_tips'] = gr.HTML(
                        value=format_help_html(qwen_base_tips_text),
                        container=True,
                        padding=True,
                        visible=is_qwen_base
                    )

                    components['vibevoice_tips'] = gr.HTML(
                        value=format_help_html(vibevoice_tips_text),
                        container=True,
                        padding=True,
                        visible=is_vibevoice
                    )

                    components['luxtts_tips'] = gr.HTML(
                        value=format_help_html(luxtts_tips_text),
                        container=True,
                        padding=True,
                        visible=is_luxtts
                    )

                    chatterbox_tips_text = dedent("""\
                    **Chatterbox Tips:**
                    - Up to 8 speakers with custom voice samples
                    - English uses fast default TTS model
                    - Other languages use Multilingual model (23 languages)
                    - Sequential generation: each line generated then stitched
                    - Exaggeration controls emotion intensity (0=neutral, 2=max)
                    - No text transcript needed for voice samples
                    """)

                    components['cb_tips'] = gr.HTML(
                        value=format_help_html(chatterbox_tips_text),
                        container=True,
                        padding=True,
                        visible=is_chatterbox
                    )

            return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Conversation tab events."""

        # Get helper functions
        get_available_samples = shared_state['get_available_samples']
        get_sample_choices = shared_state['get_sample_choices']
        save_preference = shared_state['save_preference']
        get_tenant_output_dir = shared_state['get_tenant_output_dir']
        get_tenant_paths = shared_state['get_tenant_paths']
        show_input_modal_js = shared_state['show_input_modal_js']
        input_trigger = shared_state['input_trigger']
        configure_tts_manager_for_tenant = shared_state.get('configure_tts_manager_for_tenant')
        CUSTOM_VOICE_SPEAKERS = shared_state['CUSTOM_VOICE_SPEAKERS']
        LANGUAGES = shared_state['LANGUAGES']
        _active_emotions = shared_state.get('_active_emotions', {})
        play_completion_beep = shared_state.get('play_completion_beep')
        get_or_create_voice_prompt = shared_state.get('get_or_create_voice_prompt')
        _user_config = shared_state.get('_user_config', {})
        prompt_apply_trigger = shared_state.get('prompt_apply_trigger')
        run_heavy_job = shared_state.get('run_heavy_job')
        run_heavy_stream_job = shared_state.get('run_heavy_stream_job')
        normalize_audio = shared_state['normalize_audio']
        remove_silences = shared_state['remove_silences']
        convert_to_mono = shared_state['convert_to_mono']
        clean_audio = shared_state['clean_audio']
        deepfilter_available = bool(shared_state.get("DEEPFILTER_AVAILABLE", False))

        # Get TTS manager (singleton)
        tts_manager = get_tts_manager(_user_config)

        def prepare_voice_samples_dict(v1, v2=None, v3=None, v4=None, v5=None, v6=None, v7=None, v8=None, request: gr.Request = None):
            """Prepare voice samples dictionary for generation."""
            from modules.core_components.tools import strip_sample_extension
            samples = {}
            available_samples = get_available_samples(request=request, strict=True)

            voice_inputs = [("Speaker1", v1), ("Speaker2", v2), ("Speaker3", v3), ("Speaker4", v4),
                            ("Speaker5", v5), ("Speaker6", v6), ("Speaker7", v7), ("Speaker8", v8)]

            for speaker_num, sample_name in voice_inputs:
                if sample_name:
                    bare_name = strip_sample_extension(sample_name)
                    for s in available_samples:
                        if s["name"] == bare_name:
                            samples[speaker_num] = {
                                "wav_path": s["wav_path"],
                                "ref_text": s["ref_text"],
                                "name": s["name"]
                            }
                            break
            return samples

        def validate_samples_have_transcripts(voice_samples_dict):
            """Check that all voice samples have transcripts. Returns error message or None."""
            missing = []
            for speaker_key, data in voice_samples_dict.items():
                ref_text = data.get("ref_text", "").strip()
                if not ref_text:
                    name = data.get("name", speaker_key)
                    missing.append(name)
            if missing:
                names = ", ".join(missing)
                return (
                    f"No transcript found for: {names}.\n\n"
                    "Please transcribe these samples first in the **Library Manager** tab "
                    "(using Whisper or VibeVoice ASR), then try again."
                )
            return None

        def preprocess_conversation_script(script):
            """Add [1]: to lines without speaker labels."""
            lines = script.strip().split('\n')
            processed_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Check if line already has a speaker label [N]: or [Speaker N]:
                if line.startswith('[') and ']:' in line:
                    processed_lines.append(line)
                else:
                    # Add default [1]: label
                    processed_lines.append(f"[1]: {line}")
            return '\n'.join(processed_lines)

        def extract_style_instructions(text):
            """Extract style instructions from parentheses."""
            import re
            instructions = re.findall(r'\(([^)]+)\)', text)
            clean_text = re.sub(r'\s*\([^)]+\)\s*', ' ', text)
            clean_text = ' '.join(clean_text.split())
            combined_instruct = ', '.join(instructions) if instructions else ''
            return clean_text, combined_instruct

        def _error_result(message: str):
            return None, message, "", ""

        def _create_preview_path(filename_prefix: str, request: gr.Request):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tenant_paths = get_tenant_paths(request=request, strict=True)
            temp_dir = tenant_paths.temp_dir
            temp_dir.mkdir(parents=True, exist_ok=True)
            safe_prefix = "".join(c if c.isalnum() or c in "_-" else "_" for c in filename_prefix)
            return temp_dir / f"{safe_prefix}_{timestamp}.wav", safe_prefix

        def _coerce_mono_float32(audio_data):
            if isinstance(audio_data, torch.Tensor):
                arr = audio_data.detach().cpu().numpy()
            else:
                arr = np.asarray(audio_data)
            arr = np.squeeze(arr)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            elif arr.ndim > 1:
                arr = arr.reshape(-1)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32, copy=False)
            return arr

        def _split_segments_with_pauses(text_with_breaks: str, pause_pattern):
            """Split text containing [break=x] markers into (segment, pause_after) pairs."""
            parts = pause_pattern.split(text_with_breaks)
            segments = []
            for j in range(0, len(parts), 2):
                segment_text = parts[j].strip()
                if not segment_text:
                    continue
                segment_text = pause_pattern.sub("", segment_text).strip()
                if not segment_text:
                    continue

                segment_pause = 0.0
                if j + 1 < len(parts):
                    try:
                        segment_pause = float(parts[j + 1])
                    except ValueError:
                        segment_pause = 0.0
                segments.append((segment_text, segment_pause))
            return segments

        def _unwrap_cached_voice_prompt(prompt_value):
            """Normalize prompt cache return value to prompt items only."""
            if isinstance(prompt_value, tuple) and len(prompt_value) >= 1:
                return prompt_value[0]
            return prompt_value

        class _StreamingPreviewWriter:
            """Incrementally write generated segments to disk to avoid large in-memory concatenations."""

            def __init__(self, filename_prefix: str, request: gr.Request):
                self.path, self.safe_prefix = _create_preview_path(filename_prefix, request=request)
                self._file = None
                self.sample_rate = None
                self.total_samples = 0

            def _ensure_file(self, sr: int):
                sr = int(sr)
                if self._file is None:
                    self.sample_rate = sr
                    self._file = sf.SoundFile(
                        str(self.path),
                        mode="w",
                        samplerate=sr,
                        channels=1,
                        subtype="PCM_16",
                    )
                elif sr != self.sample_rate:
                    raise ValueError(f"Mismatched sample rates while stitching ({self.sample_rate} vs {sr})")

            def add_silence(self, seconds: float):
                if self._file is None or self.sample_rate is None:
                    return
                samples = int(max(0.0, float(seconds)) * self.sample_rate)
                if samples <= 0:
                    return
                chunk_size = self.sample_rate
                while samples > 0:
                    n = min(samples, chunk_size)
                    self._file.write(np.zeros(n, dtype=np.float32))
                    self.total_samples += n
                    samples -= n

            def write_segment(self, audio_data, sr: int, pause_after: float = 0.0):
                arr = _coerce_mono_float32(audio_data)
                self._ensure_file(sr)
                self._file.write(arr)
                self.total_samples += int(arr.shape[0])
                if pause_after > 0:
                    self.add_silence(pause_after)

            def finalize(self):
                if self._file is not None:
                    self._file.close()
                    self._file = None
                if self.sample_rate is None or self.total_samples <= 0:
                    raise RuntimeError("No audio segments generated.")
                duration = self.total_samples / self.sample_rate
                return str(self.path), self.safe_prefix, self.sample_rate, duration

            def cleanup(self):
                if self._file is not None:
                    self._file.close()
                    self._file = None
                try:
                    if self.path.exists():
                        self.path.unlink()
                except Exception:
                    pass

        def generate_conversation_handler(conversation_data, pause_linebreak, pause_period, pause_comma,
                                          pause_question, pause_hyphen, language, seed, model_size,
                                          do_sample, temperature, top_k, top_p, repetition_penalty, max_new_tokens,
                                          request: gr.Request = None,
                                          progress=gr.Progress()):
            """Generate multi-speaker conversation with Qwen Speakers preset speakers."""
            if not conversation_data or not conversation_data.strip():
                return _error_result("❌ Please enter conversation lines.")

            conversation_data = preprocess_conversation_script(conversation_data)

            try:
                # Parse conversation lines
                lines = []
                for line in conversation_data.strip().split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        bracket_content = line[1:bracket_end].strip()
                        text = line[bracket_end + 1:].lstrip(':').strip()

                        if bracket_content.isdigit():
                            speaker_num = int(bracket_content)
                            if 1 <= speaker_num <= len(CUSTOM_VOICE_SPEAKERS):
                                speaker = CUSTOM_VOICE_SPEAKERS[speaker_num - 1]
                                if text:
                                    lines.append((speaker, text))

                if not lines:
                    return _error_result("❌ No valid conversation lines found. Use format: [N]: Text")

                # Set seed
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                set_seed(seed)
                if configure_tts_manager_for_tenant:
                    configure_tts_manager_for_tenant(request=request, strict=True)

                progress(0.1, desc=f"Loading CustomVoice model ({model_size})...")
                model = tts_manager.get_qwen3_custom_voice(model_size)

                # Generate all lines with pause control using streaming stitcher
                segment_count = 0
                pause_pattern = re.compile(r'\[break=([\d\.]+)\]')
                writer = _StreamingPreviewWriter("conversation_qwen3", request=request)

                try:
                    for i, (speaker, text) in enumerate(lines):
                        progress_val = 0.1 + (0.8 * i / len(lines))
                        clean_text, style_instruct = extract_style_instructions(text)

                        # Insert pause markers
                        if pause_period > 0:
                            clean_text = re.sub(r'\.(?!\d)', f'. [break={pause_period}]', clean_text)
                        if pause_comma > 0:
                            clean_text = re.sub(r',(?!\d)', f', [break={pause_comma}]', clean_text)
                        if pause_question > 0:
                            clean_text = re.sub(r'\?(?!\d)', f'? [break={pause_question}]', clean_text)
                        if pause_hyphen > 0:
                            clean_text = re.sub(r'-(?!\d)', f'- [break={pause_hyphen}]', clean_text)

                        line_segments = _split_segments_with_pauses(clean_text, pause_pattern)
                        if not line_segments:
                            continue

                        if style_instruct:
                            progress(progress_val, desc=f"Line {i + 1}/{len(lines)} [{style_instruct[:15]}...]")
                        else:
                            progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker})")

                        # Generate and stream segments
                        for j, (segment_text, segment_pause) in enumerate(line_segments):
                            kwargs = {
                                "text": segment_text,
                                "language": language if language != "Auto" else "Auto",
                                "speaker": speaker,
                                "do_sample": do_sample,
                                "temperature": temperature,
                                "top_k": top_k,
                                "top_p": top_p,
                                "repetition_penalty": repetition_penalty,
                                "max_new_tokens": max_new_tokens
                            }
                            if style_instruct:
                                kwargs["instruct"] = style_instruct

                            wavs, sr = model.generate_custom_voice(**kwargs)

                            pause_after = float(segment_pause)
                            if j == len(line_segments) - 1 and i < len(lines) - 1:
                                pause_after += float(pause_linebreak)

                            writer.write_segment(wavs[0], sr, pause_after=pause_after)
                            segment_count += 1

                    if segment_count <= 0:
                        return _error_result("❌ No audio segments generated.")

                    progress(0.9, desc="Stitching conversation...")
                    preview_path, safe_prefix, sr, duration = writer.finalize()
                except Exception:
                    writer.cleanup()
                    raise

                speakers_used = list(set(s for s, _ in lines))
                metadata = dedent(f"""\
                    Generated: {datetime.now().strftime("%Y%m%d_%H%M%S")}
                    Type: Qwen3-TTS Conversation
                    Model: CustomVoice {model_size}
                    Language: {language}
                    Seed: {seed}
                    Pause Settings:
                      - Linebreak: {pause_linebreak}s
                      - Period: {pause_period}s
                      - Comma: {pause_comma}s
                      - Question: {pause_question}s
                      - Hyphen: {pause_hyphen}s
                    Speakers: {', '.join(speakers_used)}
                    Lines: {len(lines)}
                    Segments: {segment_count}

                    --- Script ---
                    {conversation_data.strip()}
                    """)
                # Strip leading blank lines and left-strip all lines for flush-left output
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                status = f"Ready to save | {len(lines)} lines | {duration:.1f}s | Seed: {seed} | {model_size}"
                return preview_path, status, safe_prefix, metadata_out

            except Exception as e:
                return _error_result(f"❌ Error generating conversation: {str(e)}")

        def generate_conversation_base_handler(conversation_data, voice_samples_dict, pause_linebreak,
                                               pause_period, pause_comma, pause_question, pause_hyphen,
                                               language, seed, model_size, do_sample, temperature, top_k,
                                               top_p, repetition_penalty, max_new_tokens, emotion_intensity,
                                               request: gr.Request = None,
                                               progress=gr.Progress()):
            """Generate multi-speaker conversation with Qwen Base + custom voice samples."""
            if not conversation_data or not conversation_data.strip():
                return _error_result("❌ Please enter conversation lines.")

            if not voice_samples_dict:
                return _error_result("❌ Please select at least one voice sample.")

            conversation_data = preprocess_conversation_script(conversation_data)

            try:
                # Parse lines
                lines = []
                for line in conversation_data.strip().split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        bracket_content = line[1:bracket_end].strip()
                        text = line[bracket_end + 1:].lstrip(':').strip()

                        if bracket_content.isdigit():
                            speaker_num = int(bracket_content)
                            if 1 <= speaker_num <= 8:
                                speaker_key = f"Speaker{speaker_num}"
                                if speaker_key in voice_samples_dict and text:
                                    sample_data = voice_samples_dict[speaker_key]
                                    lines.append((speaker_key, sample_data["wav_path"], sample_data["ref_text"], text))

                if not lines:
                    return _error_result("❌ No valid conversation lines found. Use format: [N]: Text (N=1-8)")

                # Set seed
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                set_seed(seed)
                if configure_tts_manager_for_tenant:
                    configure_tts_manager_for_tenant(request=request, strict=True)

                progress(0.1, desc=f"Loading Base model ({model_size})...")
                model = tts_manager.get_qwen3_base(model_size)

                # Generate all segments using streaming stitcher
                segment_count = 0
                pause_pattern = re.compile(r'\[break=([\d\.]+)\]')
                writer = _StreamingPreviewWriter("conversation_qwen_base", request=request)
                try:
                    for i, (speaker_key, voice_sample_path, ref_text, text) in enumerate(lines):
                        progress_val = 0.1 + (0.8 * i / len(lines))
                        clean_text, detected_emotion = extract_style_instructions(text)

                        # Apply emotion adjustments
                        emotion_key = detected_emotion.lower().replace(" ", "_").replace(",", "").strip() if detected_emotion else None
                        line_temp = temperature
                        line_top_p = top_p
                        line_rep_pen = repetition_penalty

                        if emotion_key and emotion_key in _active_emotions:
                            adjustments = _active_emotions[emotion_key]
                            line_temp = max(0.1, min(2.0, temperature + (adjustments["temp"] * emotion_intensity)))
                            line_top_p = max(0.0, min(1.0, top_p + (adjustments["top_p"] * emotion_intensity)))
                            line_rep_pen = max(1.0, min(2.0, repetition_penalty + (adjustments["penalty"] * emotion_intensity)))
                            progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker_key}) [{emotion_key}]")
                        else:
                            progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker_key})")

                        text = clean_text

                        # Insert pause markers
                        if pause_period > 0:
                            text = re.sub(r'\.(?!\d)', f'. [break={pause_period}]', text)
                        if pause_comma > 0:
                            text = re.sub(r',(?!\d)', f', [break={pause_comma}]', text)
                        if pause_question > 0:
                            text = re.sub(r'\?(?!\d)', f'? [break={pause_question}]', text)
                        if pause_hyphen > 0:
                            text = re.sub(r'-(?!\d)', f'- [break={pause_hyphen}]', text)

                        line_segments = _split_segments_with_pauses(text, pause_pattern)
                        if not line_segments:
                            continue

                        # Get voice prompt (cached if available)
                        voice_prompt = None
                        if get_or_create_voice_prompt:
                            voice_prompt = get_or_create_voice_prompt(
                                model, speaker_key, voice_sample_path, ref_text, model_size, request=request
                            )
                            voice_prompt = _unwrap_cached_voice_prompt(voice_prompt)

                        for j, (segment_text, segment_pause) in enumerate(line_segments):
                            wavs, sr = model.generate_voice_clone(
                                text=segment_text,
                                language=language if language != "Auto" else "auto",
                                ref_audio=voice_sample_path,
                                ref_text=ref_text,
                                voice_prompt=voice_prompt,
                                do_sample=do_sample,
                                temperature=line_temp,
                                top_k=top_k,
                                top_p=line_top_p,
                                repetition_penalty=line_rep_pen,
                                max_new_tokens=max_new_tokens
                            )

                            pause_after = float(segment_pause)
                            if j == len(line_segments) - 1 and i < len(lines) - 1:
                                pause_after += float(pause_linebreak)

                            writer.write_segment(wavs[0], sr, pause_after=pause_after)
                            segment_count += 1

                    if segment_count <= 0:
                        return _error_result("❌ No audio segments generated.")

                    progress(0.9, desc="Stitching conversation...")
                    preview_path, safe_prefix, sr, duration = writer.finalize()
                except Exception:
                    writer.cleanup()
                    raise

                speakers_used = list(set(k for k, _, _, _ in lines))
                metadata = dedent(f"""\
                    Generated: {datetime.now().strftime("%Y%m%d_%H%M%S")}
                    Type: Qwen3-TTS Conversation (Base Model + Custom Voices)
                    Model: Base {model_size}
                    Language: {language}
                    Seed: {seed}
                    Pause Settings:
                      - Linebreak: {pause_linebreak}s
                      - Period: {pause_period}s
                      - Comma: {pause_comma}s
                      - Question: {pause_question}s
                      - Hyphen: {pause_hyphen}s
                    Speakers: {', '.join(speakers_used)}
                    Lines: {len(lines)}
                    Segments: {segment_count}

                    --- Script ---
                    {conversation_data.strip()}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                status = f"Ready to save | {len(lines)} lines | {duration:.1f}s | Seed: {seed} | Base {model_size}"
                return preview_path, status, safe_prefix, metadata_out

            except Exception as e:
                import traceback
                print(f"Error in generate_conversation_base_handler:\n{traceback.format_exc()}")
                return _error_result(f"❌ Error generating conversation: {str(e)}")

        def generate_vibevoice_longform_handler(script_text, voice_samples_dict, model_size, cfg_scale, seed,
                                                num_steps, do_sample, temperature, top_k, top_p, repetition_penalty,
                                                sentences_per_chunk=0, request: gr.Request = None, progress=gr.Progress()):
            """Generate long-form multi-speaker audio with VibeVoice (up to 90 min)."""
            if not script_text or not script_text.strip():
                return _error_result("❌ Please enter a script.")

            script_text = preprocess_conversation_script(script_text)

            try:
                # Set seed
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                set_seed(seed)
                if configure_tts_manager_for_tenant:
                    configure_tts_manager_for_tenant(request=request, strict=True)

                progress(0.1, desc=f"Loading VibeVoice TTS ({model_size})...")
                model = tts_manager.get_vibevoice_tts(model_size)

                # Import processor
                from vibevoice_tts.processor.vibevoice_processor import VibeVoiceProcessor
                import warnings
                import logging

                # Map model size
                if model_size == "Large (4-bit)":
                    model_repo_id = "FranckyB/VibeVoice-Large-4bit"
                    download_label = "VibeVoice-Large (4-bit)"
                else:
                    model_repo_id = f"FranckyB/VibeVoice-{model_size}"
                    download_label = "VibeVoice-Large" if model_size == "Large" else "VibeVoice-1.5B"

                offline_mode = _user_config.get("offline_mode", False)
                processor_source = resolve_model_source(
                    model_repo_id,
                    offline_mode=offline_mode,
                    settings_download_name=download_label,
                    auto_download_when_online=True,
                )

                tokenizer_repo_id = "Qwen/Qwen2.5-1.5B"
                tokenizer_source = resolve_model_source(
                    tokenizer_repo_id,
                    offline_mode=offline_mode,
                    settings_download_name="Qwen2.5-1.5B (VibeVoice Tokenizer)",
                    auto_download_when_online=True,
                )

                # Suppress tokenizer warning
                prev_level = logging.getLogger("transformers.tokenization_utils_base").level
                logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    processor = VibeVoiceProcessor.from_pretrained(
                        processor_source,
                        local_files_only=offline_mode,
                        language_model_pretrained_name=tokenizer_source,
                    )

                logging.getLogger("transformers.tokenization_utils_base").setLevel(prev_level)

                # Parse script
                progress(0.3, desc="Processing script...")
                lines = []
                for line in script_text.strip().split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        bracket_content = line[1:bracket_end].strip()
                        text = line[bracket_end + 1:].lstrip(':').strip()

                        if bracket_content.isdigit():
                            speaker_num = int(bracket_content)
                            if text:
                                wrapped_num = ((speaker_num - 1) % 4) + 1
                                lines.append((f"Speaker{wrapped_num}", text, speaker_num))

                # Build voice samples list
                available_samples = []
                for i in range(1, 5):
                    speaker_key = f"Speaker{i}"
                    if speaker_key in voice_samples_dict and voice_samples_dict[speaker_key]:
                        sample_data = voice_samples_dict[speaker_key]
                        wav_path = sample_data["wav_path"] if isinstance(sample_data, dict) else sample_data
                        available_samples.append((speaker_key, wav_path))

                if not available_samples:
                    return _error_result("❌ Please provide at least one voice sample (Speaker1).")

                voice_samples = [sample for _, sample in available_samples]
                speaker_to_sample = {speaker: idx for idx, (speaker, _) in enumerate(available_samples)}

                # Format script for VibeVoice (0-based)
                formatted_lines = []
                for speaker, text, original_num in lines:
                    if speaker in speaker_to_sample:
                        vv_speaker_num = speaker_to_sample[speaker]
                        clean_text, _ = extract_style_instructions(text)
                        formatted_lines.append(f"Speaker {vv_speaker_num}: {clean_text}")

                formatted_script = '\n'.join(formatted_lines)

                # Process inputs
                inputs = processor(
                    text=[formatted_script],
                    voice_samples=[voice_samples],
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )

                # Move to device
                device = get_device()
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to(device)

                progress(0.5, desc="Generating audio...")

                # Build generation config
                gen_config = {'do_sample': do_sample}
                if do_sample:
                    gen_config['temperature'] = temperature
                    if top_k > 0:
                        gen_config['top_k'] = int(top_k)
                    if top_p < 1.0:
                        gen_config['top_p'] = top_p
                    if repetition_penalty != 1.0:
                        gen_config['repetition_penalty'] = repetition_penalty

                model.set_ddpm_inference_steps(num_steps=num_steps)
                sr = 24000

                sentences_per_chunk = int(sentences_per_chunk) if sentences_per_chunk else 0
                writer = _StreamingPreviewWriter("conversation_vibevoice", request=request)
                chunk_count = 0

                try:
                    # Chunked generation: process N lines at a time to prevent
                    # quality degradation (screaming/rushing) on long conversations.
                    if sentences_per_chunk > 0 and len(formatted_lines) > sentences_per_chunk:
                        chunks = []
                        for i in range(0, len(formatted_lines), sentences_per_chunk):
                            chunks.append(formatted_lines[i:i + sentences_per_chunk])

                        print(f"VibeVoice conversation chunking: {len(chunks)} chunks of ~{sentences_per_chunk} line(s)")

                        for idx, chunk_lines in enumerate(chunks):
                            chunk_script = '\n'.join(chunk_lines)
                            progress_val = 0.5 + (0.4 * idx / len(chunks))
                            progress(progress_val, desc=f"Generating chunk {idx + 1}/{len(chunks)}...")

                            chunk_inputs = processor(
                                text=[chunk_script],
                                voice_samples=[voice_samples],
                                padding=True,
                                return_tensors="pt",
                                return_attention_mask=True,
                            )

                            for k, v in chunk_inputs.items():
                                if torch.is_tensor(v):
                                    chunk_inputs[k] = v.to(device)

                            outputs = model.generate(
                                **chunk_inputs,
                                max_new_tokens=None,
                                cfg_scale=cfg_scale,
                                tokenizer=processor.tokenizer,
                                generation_config=gen_config,
                                verbose=False,
                            )

                            if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                                audio_tensor = outputs.speech_outputs[0].cpu().to(torch.float32)
                                writer.write_segment(audio_tensor.squeeze(), sr, pause_after=0.0)
                                chunk_count += 1
                                print(f"  Chunk {idx + 1}/{len(chunks)} done ({len(chunk_lines)} lines)")
                            else:
                                print(f"  Chunk {idx + 1}/{len(chunks)} produced no audio, skipping")
                    else:
                        # Standard single-pass generation
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=None,
                            cfg_scale=cfg_scale,
                            tokenizer=processor.tokenizer,
                            generation_config=gen_config,
                            verbose=False,
                        )

                        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
                            audio_tensor = outputs.speech_outputs[0].cpu().to(torch.float32)
                            writer.write_segment(audio_tensor.squeeze(), sr, pause_after=0.0)
                            chunk_count = 1

                    if chunk_count <= 0:
                        return _error_result("❌ VibeVoice failed to generate audio.")

                    progress(0.9, desc="Saving preview...")
                    preview_path, safe_prefix, sr, duration = writer.finalize()
                except Exception:
                    writer.cleanup()
                    raise

                chunk_info = f"Lines per Chunk: {sentences_per_chunk}" if sentences_per_chunk > 0 else "Chunking: Off"
                metadata = dedent(f"""\
                    Generated: {datetime.now().strftime("%Y%m%d_%H%M%S")}
                    Type: VibeVoice Conversation
                    Model: VibeVoice-{model_size}
                    Seed: {seed}
                    CFG Scale: {cfg_scale}
                    {chunk_info}
                    Lines: {len(lines)}
                    Speakers: {len(available_samples)}

                    --- Script ---
                    {script_text.strip()}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                status = f"Ready to save | {len(lines)} lines | {duration:.1f}s | Seed: {seed} | VibeVoice"
                return preview_path, status, safe_prefix, metadata_out

            except Exception as e:
                import traceback
                print(f"Error in generate_vibevoice_longform_handler:\n{traceback.format_exc()}")
                return _error_result(f"❌ Error generating conversation: {str(e)}")

        def generate_luxtts_conversation_handler(
            conversation_data, voice_samples_dict,
            pause_linebreak, seed,
            num_steps, t_shift, speed, guidance_scale,
            rms, ref_duration, return_smooth,
            request: gr.Request = None,
            progress=gr.Progress()
        ):
            """Generate multi-speaker conversation with LuxTTS voice cloning.

            Each line is generated sequentially with per-speaker voice prompts (cached),
            then stitched together with configurable pauses between speaker turns.
            """
            if not conversation_data or not conversation_data.strip():
                return _error_result("Error: Please enter conversation lines.")

            if not voice_samples_dict:
                return _error_result("Error: Please select at least one voice sample.")

            # Check all samples have transcripts
            transcript_error = validate_samples_have_transcripts(voice_samples_dict)
            if transcript_error:
                return _error_result(f"Error: {transcript_error}")

            conversation_data = preprocess_conversation_script(conversation_data)

            try:
                # Parse lines
                lines = []
                for line in conversation_data.strip().split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        bracket_content = line[1:bracket_end].strip()
                        text = line[bracket_end + 1:].lstrip(':').strip()

                        if bracket_content.isdigit():
                            speaker_num = int(bracket_content)
                            if 1 <= speaker_num <= 8:
                                speaker_key = f"Speaker{speaker_num}"
                                if speaker_key in voice_samples_dict and text:
                                    sample_data = voice_samples_dict[speaker_key]
                                    lines.append((speaker_key, sample_data["wav_path"], sample_data.get("name", speaker_key), text, sample_data.get("ref_text", "")))

                if not lines:
                    return _error_result("Error: No valid conversation lines found. Use format: [N]: Text (N=1-8)")

                # Set seed
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                set_seed(seed)
                if configure_tts_manager_for_tenant:
                    configure_tts_manager_for_tenant(request=request, strict=True)

                progress(0.05, desc="Loading LuxTTS model...")
                tts_manager.get_luxtts()

                # Generate all segments using streaming stitcher
                segment_count = 0
                writer = _StreamingPreviewWriter("conversation_luxtts", request=request)
                try:
                    for i, (speaker_key, wav_path, sample_name, text, ref_text) in enumerate(lines):
                        progress_val = 0.1 + (0.8 * i / len(lines))

                        # Strip any (style) markers — LuxTTS doesn't support them
                        clean_text, _ = extract_style_instructions(text)
                        if not clean_text.strip():
                            continue

                        progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker_key})")

                        audio_data, audio_sr, _was_cached = tts_manager.generate_voice_clone_luxtts(
                            text=clean_text,
                            voice_sample_path=wav_path,
                            sample_name=sample_name,
                            num_steps=num_steps,
                            t_shift=t_shift,
                            speed=speed,
                            return_smooth=return_smooth,
                            rms=rms,
                            ref_duration=ref_duration,
                            guidance_scale=guidance_scale,
                            seed=seed,
                            ref_text=ref_text or None,
                        )

                        # Add segment with linebreak pause after (except last line)
                        line_pause = pause_linebreak if i < len(lines) - 1 else 0.0
                        writer.write_segment(audio_data, audio_sr, pause_after=line_pause)
                        segment_count += 1

                    if segment_count <= 0:
                        return _error_result("Error: No audio segments generated.")

                    # Add short tail silence so the last utterance doesn't clip
                    writer.add_silence(0.15)

                    progress(0.9, desc="Stitching conversation...")
                    preview_path, safe_prefix, sr, duration = writer.finalize()
                except Exception:
                    writer.cleanup()
                    raise

                speakers_used = list(set(k for k, _, _, _, _ in lines))
                metadata = dedent(f"""\
                    Generated: {datetime.now().strftime("%Y%m%d_%H%M%S")}
                    Type: LuxTTS Conversation
                    Seed: {seed}
                    Sample Rate: {sr}Hz
                    Pause Between Lines: {pause_linebreak}s
                    Steps: {num_steps} | t_shift: {t_shift} | Speed: {speed}
                    Guidance Scale: {guidance_scale} | RMS: {rms} | Ref Duration: {ref_duration}s
                    Return Smooth: {return_smooth}
                    Speakers: {', '.join(speakers_used)}
                    Lines: {len(lines)}
                    Segments: {segment_count}

                    --- Script ---
                    {conversation_data.strip()}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                status = f"Ready to save | {len(lines)} lines | {duration:.1f}s | Seed: {seed} | LuxTTS"
                return preview_path, status, safe_prefix, metadata_out

            except Exception as e:
                import traceback
                print(f"Error in generate_luxtts_conversation_handler:\n{traceback.format_exc()}")
                return _error_result(f"Error generating LuxTTS conversation: {str(e)}")

        def generate_chatterbox_conversation_handler(
            conversation_data, voice_samples_dict,
            pause_linebreak, language, seed,
            exaggeration, cfg_weight, temperature, repetition_penalty, top_p, min_p, max_new_tokens,
            request: gr.Request = None,
            progress=gr.Progress()
        ):
            """Generate multi-speaker conversation with Chatterbox voice cloning.

            Each line is generated sequentially with per-speaker voice samples,
            then stitched together with configurable pauses between speaker turns.
            English uses the fast default model; other languages use Multilingual.
            """
            if not conversation_data or not conversation_data.strip():
                return _error_result("Error: Please enter conversation lines.")

            if not voice_samples_dict:
                return _error_result("Error: Please select at least one voice sample.")

            conversation_data = preprocess_conversation_script(conversation_data)

            try:
                from modules.core_components.constants import CHATTERBOX_LANG_TO_CODE

                # Parse lines
                lines = []
                for line in conversation_data.strip().split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        bracket_content = line[1:bracket_end].strip()
                        text = line[bracket_end + 1:].lstrip(':').strip()

                        if bracket_content.isdigit():
                            speaker_num = int(bracket_content)
                            if 1 <= speaker_num <= 8:
                                speaker_key = f"Speaker{speaker_num}"
                                if speaker_key in voice_samples_dict and text:
                                    sample_data = voice_samples_dict[speaker_key]
                                    lines.append((speaker_key, sample_data["wav_path"], sample_data.get("name", speaker_key), text))

                if not lines:
                    return _error_result("Error: No valid conversation lines found. Use format: [N]: Text (N=1-8)")

                # Set seed
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                set_seed(seed)
                if configure_tts_manager_for_tenant:
                    configure_tts_manager_for_tenant(request=request, strict=True)

                # Determine if multilingual
                lang_code = CHATTERBOX_LANG_TO_CODE.get(language, "en")
                use_multilingual = lang_code != "en"
                model_label = "Multilingual" if use_multilingual else "Default"
                min_p = max(0.0, min(0.30, float(min_p)))
                max_new_tokens = max(256, min(4096, int(max_new_tokens)))

                if use_multilingual:
                    progress(0.05, desc="Loading Chatterbox Multilingual...")
                    tts_manager.get_chatterbox_multilingual()
                else:
                    progress(0.05, desc="Loading Chatterbox TTS...")
                    tts_manager.get_chatterbox_tts()

                # Generate all segments using streaming stitcher
                segment_count = 0
                writer = _StreamingPreviewWriter("conversation_chatterbox", request=request)
                try:
                    for i, (speaker_key, wav_path, sample_name, text) in enumerate(lines):
                        progress_val = 0.1 + (0.8 * i / len(lines))

                        # Strip any (style) markers
                        clean_text, _ = extract_style_instructions(text)
                        if not clean_text.strip():
                            continue

                        progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker_key})")

                        if use_multilingual:
                            audio_data, audio_sr = tts_manager.generate_voice_clone_chatterbox_multilingual(
                                text=clean_text,
                                language_code=lang_code,
                                voice_sample_path=wav_path,
                                seed=seed,
                                exaggeration=exaggeration,
                                cfg_weight=cfg_weight,
                                temperature=temperature,
                                repetition_penalty=repetition_penalty,
                                top_p=top_p,
                                min_p=min_p,
                                max_new_tokens=max_new_tokens,
                            )
                        else:
                            audio_data, audio_sr = tts_manager.generate_voice_clone_chatterbox(
                                text=clean_text,
                                voice_sample_path=wav_path,
                                seed=seed,
                                exaggeration=exaggeration,
                                cfg_weight=cfg_weight,
                                temperature=temperature,
                                repetition_penalty=repetition_penalty,
                                top_p=top_p,
                                min_p=min_p,
                                max_new_tokens=max_new_tokens,
                            )

                        # Add segment with linebreak pause after (except last line)
                        line_pause = pause_linebreak if i < len(lines) - 1 else 0.0
                        writer.write_segment(audio_data, audio_sr, pause_after=line_pause)
                        segment_count += 1

                    if segment_count <= 0:
                        return _error_result("Error: No audio segments generated.")

                    # Add short tail silence
                    writer.add_silence(0.15)

                    progress(0.9, desc="Stitching conversation...")
                    preview_path, safe_prefix, sr, duration = writer.finalize()
                except Exception:
                    writer.cleanup()
                    raise

                speakers_used = list(set(k for k, _, _, _ in lines))
                metadata = dedent(f"""\
                    Generated: {datetime.now().strftime("%Y%m%d_%H%M%S")}
                    Type: Chatterbox Conversation
                    Model: {model_label}
                    Language: {language}
                    Seed: {seed}
                    Sample Rate: {sr}Hz
                    Pause Between Lines: {pause_linebreak}s
                    Exaggeration: {exaggeration} | CFG Weight: {cfg_weight}
                    Temperature: {temperature} | Repetition Penalty: {repetition_penalty} | Top-p: {top_p}
                    Min-p: {min_p} | Max New Tokens: {max_new_tokens}
                    Speakers: {', '.join(speakers_used)}
                    Lines: {len(lines)}
                    Segments: {segment_count}

                    --- Script ---
                    {conversation_data.strip()}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                status = f"Ready to save | {len(lines)} lines | {duration:.1f}s | Seed: {seed} | Chatterbox {model_label}"
                return preview_path, status, safe_prefix, metadata_out

            except Exception as e:
                import traceback
                print(f"Error in generate_chatterbox_conversation_handler:\n{traceback.format_exc()}")
                return _error_result(f"Error generating Chatterbox conversation: {str(e)}")

        def generate_conversation_base_stream_handler(conversation_data, voice_samples_dict, pause_linebreak,
                                                      pause_period, pause_comma, pause_question, pause_hyphen,
                                                      language, seed, model_size, do_sample, temperature, top_k,
                                                      top_p, repetition_penalty, max_new_tokens, emotion_intensity,
                                                      request: gr.Request = None,
                                                      progress=gr.Progress()):
            """Stream multi-speaker conversation with Qwen Base + custom voice samples."""
            if not conversation_data or not conversation_data.strip():
                yield None, "❌ Please enter conversation lines.", gr.update(interactive=False), "", "", ""
                return

            if not voice_samples_dict:
                yield None, "❌ Please select at least one voice sample.", gr.update(interactive=False), "", "", ""
                return

            conversation_data = preprocess_conversation_script(conversation_data)
            writer = None
            finalized = False

            try:
                # Parse lines
                lines = []
                for line in conversation_data.strip().split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        bracket_content = line[1:bracket_end].strip()
                        text = line[bracket_end + 1:].lstrip(':').strip()

                        if bracket_content.isdigit():
                            speaker_num = int(bracket_content)
                            if 1 <= speaker_num <= 8:
                                speaker_key = f"Speaker{speaker_num}"
                                if speaker_key in voice_samples_dict and text:
                                    sample_data = voice_samples_dict[speaker_key]
                                    lines.append((speaker_key, sample_data["wav_path"], sample_data["ref_text"], text))

                if not lines:
                    yield None, "❌ No valid conversation lines found. Use format: [N]: Text (N=1-8)", gr.update(interactive=False), "", "", ""
                    return

                # Set seed and configure
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                set_seed(seed)
                if configure_tts_manager_for_tenant:
                    configure_tts_manager_for_tenant(request=request, strict=True)

                progress(0.1, desc=f"Loading Base model ({model_size})...")
                model = tts_manager.get_qwen3_base(model_size)

                preview_path, safe_prefix = _create_preview_path("conversation_qwen_base_stream", request=request)
                writer = LiveAudioChunkWriter(preview_path)
                pause_pattern = re.compile(r'\[break=([\d\.]+)\]')
                streamed_chunks = 0

                # Reset audio output for a fresh live stream session.
                yield gr.update(value=None), "Streaming Qwen Base...", gr.update(interactive=False), "", "", ""

                for i, (speaker_key, voice_sample_path, ref_text, text) in enumerate(lines):
                    progress_val = 0.1 + (0.8 * i / len(lines))
                    clean_text, detected_emotion = extract_style_instructions(text)

                    # Apply emotion adjustments
                    emotion_key = detected_emotion.lower().replace(" ", "_").replace(",", "").strip() if detected_emotion else None
                    line_temp = temperature
                    line_top_p = top_p
                    line_rep_pen = repetition_penalty

                    if emotion_key and emotion_key in _active_emotions:
                        adjustments = _active_emotions[emotion_key]
                        line_temp = max(0.1, min(2.0, temperature + (adjustments["temp"] * emotion_intensity)))
                        line_top_p = max(0.0, min(1.0, top_p + (adjustments["top_p"] * emotion_intensity)))
                        line_rep_pen = max(1.0, min(2.0, repetition_penalty + (adjustments["penalty"] * emotion_intensity)))
                        progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker_key}) [{emotion_key}]")
                    else:
                        progress(progress_val, desc=f"Line {i + 1}/{len(lines)} ({speaker_key})")

                    line_text = clean_text

                    if pause_period > 0:
                        line_text = re.sub(r'\.(?!\d)', f'. [break={pause_period}]', line_text)
                    if pause_comma > 0:
                        line_text = re.sub(r',(?!\d)', f', [break={pause_comma}]', line_text)
                    if pause_question > 0:
                        line_text = re.sub(r'\?(?!\d)', f'? [break={pause_question}]', line_text)
                    if pause_hyphen > 0:
                        line_text = re.sub(r'-(?!\d)', f'- [break={pause_hyphen}]', line_text)

                    line_segments = _split_segments_with_pauses(line_text, pause_pattern)
                    if not line_segments:
                        continue

                    voice_prompt = None
                    if get_or_create_voice_prompt:
                        voice_prompt = get_or_create_voice_prompt(
                            model, speaker_key, voice_sample_path, ref_text, model_size, request=request
                        )
                        voice_prompt = _unwrap_cached_voice_prompt(voice_prompt)

                    for j, (segment_text, segment_pause) in enumerate(line_segments):
                        stream_iter = tts_manager.stream_voice_clone_qwen(
                            text=segment_text,
                            language=language if language != "Auto" else "auto",
                            prompt_items=voice_prompt,
                            seed=seed,
                            do_sample=do_sample,
                            temperature=line_temp,
                            top_k=top_k,
                            top_p=line_top_p,
                            repetition_penalty=line_rep_pen,
                            max_new_tokens=max_new_tokens,
                            model_size=model_size,
                        )

                        emitted_segment_chunk = False
                        for chunk_audio, chunk_sr in stream_iter:
                            chunk_arr = normalize_audio_chunk(chunk_audio)
                            chunk_path = writer.write_chunk(chunk_arr, chunk_sr)
                            if not chunk_path:
                                continue
                            emitted_segment_chunk = True
                            streamed_chunks += 1
                            ui_chunk = np.clip(chunk_arr, -1.0, 1.0)
                            ui_chunk = (ui_chunk * 32767.0).astype(np.int16, copy=False)
                            yield (
                                (int(chunk_sr), ui_chunk),
                                f"Streaming Qwen Base... line {i + 1}/{len(lines)}",
                                gr.update(interactive=False),
                                "",
                                "",
                                "",
                            )

                        if emitted_segment_chunk:
                            pause_after = float(segment_pause)
                            if j == len(line_segments) - 1 and i < len(lines) - 1:
                                pause_after += float(pause_linebreak)
                            if pause_after > 0:
                                writer.add_silence(pause_after)

                if streamed_chunks <= 0:
                    raise RuntimeError("Qwen Base did not emit any streamed audio chunks.")

                result = writer.finalize()
                finalized = True
                speakers_used = list(set(k for k, _, _, _ in lines))
                metadata = dedent(f"""\
                    Generated: {datetime.now().strftime("%Y%m%d_%H%M%S")}
                    Type: Qwen3-TTS Conversation (Base Model + Custom Voices)
                    Model: Base {model_size}
                    Language: {language}
                    Seed: {seed}
                    Stream Chunks: {streamed_chunks}
                    Pause Settings:
                      - Linebreak: {pause_linebreak}s
                      - Period: {pause_period}s
                      - Comma: {pause_comma}s
                      - Question: {pause_question}s
                      - Hyphen: {pause_hyphen}s
                    Speakers: {', '.join(speakers_used)}
                    Lines: {len(lines)}

                    --- Script ---
                    {conversation_data.strip()}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())
                status = f"Ready to save | {len(lines)} lines | {result.duration_seconds:.1f}s | Seed: {seed} | Base {model_size}"
                if play_completion_beep:
                    play_completion_beep()
                # Keep the streamed audio in the player; only publish finalized filepath via state.
                yield gr.update(), status, gr.update(interactive=True), safe_prefix, metadata_out, result.path
            except Exception as e:
                import traceback
                print(f"Error in generate_conversation_base_stream_handler:\n{traceback.format_exc()}")
                yield None, f"❌ Error generating conversation: {str(e)}", gr.update(interactive=False), "", "", ""
            finally:
                if writer is not None and not finalized:
                    writer.cleanup_final_file()

        def generate_vibevoice_longform_stream_handler(script_text, voice_samples_dict, model_size, cfg_scale, seed,
                                                       num_steps, do_sample, temperature, top_k, top_p, repetition_penalty,
                                                       sentences_per_chunk=0, request: gr.Request = None,
                                                       progress=gr.Progress()):
            """Stream long-form multi-speaker audio with VibeVoice native audio streamer."""
            if not script_text or not script_text.strip():
                yield None, "❌ Please enter a script.", gr.update(interactive=False), "", "", ""
                return

            script_text = preprocess_conversation_script(script_text)
            writer = None
            finalized = False

            try:
                import warnings
                import logging
                import threading
                from modules.vibevoice_tts.modular.streamer import AudioStreamer
                from vibevoice_tts.processor.vibevoice_processor import VibeVoiceProcessor

                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)
                set_seed(seed)
                if configure_tts_manager_for_tenant:
                    configure_tts_manager_for_tenant(request=request, strict=True)

                progress(0.1, desc=f"Loading VibeVoice TTS ({model_size})...")
                model = tts_manager.get_vibevoice_tts(model_size)

                if model_size == "Large (4-bit)":
                    model_repo_id = "FranckyB/VibeVoice-Large-4bit"
                    download_label = "VibeVoice-Large (4-bit)"
                else:
                    model_repo_id = f"FranckyB/VibeVoice-{model_size}"
                    download_label = "VibeVoice-Large" if model_size == "Large" else "VibeVoice-1.5B"

                offline_mode = _user_config.get("offline_mode", False)
                processor_source = resolve_model_source(
                    model_repo_id,
                    offline_mode=offline_mode,
                    settings_download_name=download_label,
                    auto_download_when_online=True,
                )
                tokenizer_source = resolve_model_source(
                    "Qwen/Qwen2.5-1.5B",
                    offline_mode=offline_mode,
                    settings_download_name="Qwen2.5-1.5B (VibeVoice Tokenizer)",
                    auto_download_when_online=True,
                )

                prev_level = logging.getLogger("transformers.tokenization_utils_base").level
                logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    processor = VibeVoiceProcessor.from_pretrained(
                        processor_source,
                        local_files_only=offline_mode,
                        language_model_pretrained_name=tokenizer_source,
                    )
                logging.getLogger("transformers.tokenization_utils_base").setLevel(prev_level)

                # Parse script
                lines = []
                for line in script_text.strip().split('\n'):
                    line = line.strip()
                    if not line or ':' not in line:
                        continue

                    if line.startswith('[') and ']' in line:
                        bracket_end = line.index(']')
                        bracket_content = line[1:bracket_end].strip()
                        text = line[bracket_end + 1:].lstrip(':').strip()
                        if bracket_content.isdigit():
                            speaker_num = int(bracket_content)
                            if text:
                                wrapped_num = ((speaker_num - 1) % 4) + 1
                                lines.append((f"Speaker{wrapped_num}", text, speaker_num))

                available_samples = []
                for i in range(1, 5):
                    speaker_key = f"Speaker{i}"
                    if speaker_key in voice_samples_dict and voice_samples_dict[speaker_key]:
                        sample_data = voice_samples_dict[speaker_key]
                        wav_path = sample_data["wav_path"] if isinstance(sample_data, dict) else sample_data
                        available_samples.append((speaker_key, wav_path))

                if not available_samples:
                    yield None, "❌ Please provide at least one voice sample (Speaker1).", gr.update(interactive=False), "", "", ""
                    return

                voice_samples = [sample for _, sample in available_samples]
                speaker_to_sample = {speaker: idx for idx, (speaker, _) in enumerate(available_samples)}

                formatted_lines = []
                for speaker, text, _original_num in lines:
                    if speaker in speaker_to_sample:
                        vv_speaker_num = speaker_to_sample[speaker]
                        clean_text, _ = extract_style_instructions(text)
                        formatted_lines.append(f"Speaker {vv_speaker_num}: {clean_text}")

                if not formatted_lines:
                    yield None, "❌ No valid conversation lines found.", gr.update(interactive=False), "", "", ""
                    return

                sentences_per_chunk = int(sentences_per_chunk) if sentences_per_chunk else 0
                if sentences_per_chunk > 0 and len(formatted_lines) > sentences_per_chunk:
                    script_chunks = [
                        '\n'.join(formatted_lines[i:i + sentences_per_chunk])
                        for i in range(0, len(formatted_lines), sentences_per_chunk)
                    ]
                else:
                    script_chunks = ['\n'.join(formatted_lines)]

                gen_config = {'do_sample': do_sample}
                if do_sample:
                    gen_config['temperature'] = temperature
                    if top_k > 0:
                        gen_config['top_k'] = int(top_k)
                    if top_p < 1.0:
                        gen_config['top_p'] = top_p
                    if repetition_penalty != 1.0:
                        gen_config['repetition_penalty'] = repetition_penalty

                model.set_ddpm_inference_steps(num_steps=num_steps)
                device = get_device()
                preview_path, safe_prefix = _create_preview_path("conversation_vibevoice_stream", request=request)
                writer = LiveAudioChunkWriter(preview_path)
                streamed_chunks = 0
                sr = 24000
                min_emit_samples = int(sr * 0.20)
                pending_chunks = []
                pending_samples = 0

                def _take_pending_chunk():
                    nonlocal pending_chunks, pending_samples
                    if pending_samples <= 0:
                        return None
                    if len(pending_chunks) == 1:
                        merged = pending_chunks[0]
                    else:
                        merged = np.concatenate(pending_chunks, axis=0)
                    pending_chunks = []
                    pending_samples = 0
                    return merged

                # Reset audio output for a fresh live stream session.
                yield gr.update(value=None), "Streaming VibeVoice...", gr.update(interactive=False), "", "", ""

                for idx, chunk_script in enumerate(script_chunks):
                    progress_val = 0.2 + (0.6 * idx / len(script_chunks))
                    progress(progress_val, desc=f"Streaming chunk {idx + 1}/{len(script_chunks)}...")

                    chunk_inputs = processor(
                        text=[chunk_script],
                        voice_samples=[voice_samples],
                        padding=True,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                    for k, v in chunk_inputs.items():
                        if torch.is_tensor(v):
                            chunk_inputs[k] = v.to(device)

                    streamer = AudioStreamer(batch_size=1, timeout=None)
                    generation_errors = []

                    def _run_generation():
                        try:
                            model.generate(
                                **chunk_inputs,
                                max_new_tokens=None,
                                cfg_scale=cfg_scale,
                                tokenizer=processor.tokenizer,
                                generation_config=gen_config,
                                audio_streamer=streamer,
                                verbose=False,
                            )
                        except Exception as exc:
                            generation_errors.append(exc)
                            try:
                                streamer.end()
                            except Exception:
                                pass

                    worker = threading.Thread(target=_run_generation, daemon=True, name=f"vv-conv-stream-{idx}")
                    worker.start()

                    for audio_chunk in streamer.get_stream(0):
                        chunk_arr = normalize_audio_chunk(audio_chunk)
                        if chunk_arr.size <= 0:
                            continue
                        pending_chunks.append(chunk_arr)
                        pending_samples += int(chunk_arr.shape[0])
                        if pending_samples < min_emit_samples:
                            continue
                        chunk_arr = _take_pending_chunk()
                        if chunk_arr is None or chunk_arr.size <= 0:
                            continue
                        chunk_path = writer.write_chunk(chunk_arr, sr)
                        if not chunk_path:
                            continue
                        streamed_chunks += 1
                        ui_chunk = np.clip(chunk_arr, -1.0, 1.0)
                        ui_chunk = (ui_chunk * 32767.0).astype(np.int16, copy=False)
                        yield (
                            (int(sr), ui_chunk),
                            f"Streaming VibeVoice... chunk {idx + 1}/{len(script_chunks)}",
                            gr.update(interactive=False),
                            "",
                            "",
                            "",
                        )

                    worker.join()
                    tail_arr = _take_pending_chunk()
                    if tail_arr is not None and tail_arr.size > 0:
                        tail_path = writer.write_chunk(tail_arr, sr)
                        if tail_path:
                            streamed_chunks += 1
                            ui_tail = np.clip(tail_arr, -1.0, 1.0)
                            ui_tail = (ui_tail * 32767.0).astype(np.int16, copy=False)
                            yield (
                                (int(sr), ui_tail),
                                f"Streaming VibeVoice... chunk {idx + 1}/{len(script_chunks)}",
                                gr.update(interactive=False),
                                "",
                                "",
                                "",
                            )
                    if generation_errors:
                        raise generation_errors[0]

                if streamed_chunks <= 0:
                    raise RuntimeError("VibeVoice failed to emit streamed audio chunks.")

                result = writer.finalize()
                finalized = True
                chunk_info = f"Lines per Chunk: {sentences_per_chunk}" if sentences_per_chunk > 0 else "Chunking: Off"
                metadata = dedent(f"""\
                    Generated: {datetime.now().strftime("%Y%m%d_%H%M%S")}
                    Type: VibeVoice Conversation
                    Model: VibeVoice-{model_size}
                    Seed: {seed}
                    CFG Scale: {cfg_scale}
                    Stream Chunks: {streamed_chunks}
                    {chunk_info}
                    Lines: {len(lines)}
                    Speakers: {len(available_samples)}

                    --- Script ---
                    {script_text.strip()}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())
                status = f"Ready to save | {len(lines)} lines | {result.duration_seconds:.1f}s | Seed: {seed} | VibeVoice"
                if play_completion_beep:
                    play_completion_beep()
                # Keep the streamed audio in the player; only publish finalized filepath via state.
                yield gr.update(), status, gr.update(interactive=True), safe_prefix, metadata_out, result.path
            except Exception as e:
                import traceback
                print(f"Error in generate_vibevoice_longform_stream_handler:\n{traceback.format_exc()}")
                yield None, f"❌ Error generating conversation: {str(e)}", gr.update(interactive=False), "", "", ""
            finally:
                if writer is not None and not finalized:
                    writer.cleanup_final_file()

        def unified_conversation_generate(
            model_type, script,
            # Shared voice samples (used by all sample-based engines)
            sv1, sv2, sv3, sv4, sv5, sv6, sv7, sv8,
            # Qwen Speakers params
            qwen_custom_pause_linebreak, qwen_custom_pause_period, qwen_custom_pause_comma,
            qwen_custom_pause_question, qwen_custom_pause_hyphen, qwen_custom_model_size,
            # Qwen Base params
            qwen_base_pause_linebreak, qwen_base_pause_period, qwen_base_pause_comma, qwen_base_pause_question,
            qwen_base_pause_hyphen, qwen_base_model_size,
            # Shared Qwen params
            qwen_lang, qwen_seed, emotion_intensity,
            # Qwen advanced params
            qwen_do_sample, qwen_temperature, qwen_top_k, qwen_top_p, qwen_repetition_penalty, qwen_max_new_tokens,
            # VibeVoice params
            vv_model_size, vv_cfg,
            # VibeVoice advanced params
            vv_num_steps, vv_do_sample, vv_temperature, vv_top_k, vv_top_p, vv_repetition_penalty,
            vv_sentences_per_chunk,
            # LuxTTS params
            lux_pause_linebreak,
            lux_num_steps, lux_t_shift, lux_speed, lux_guidance_scale,
            lux_rms, lux_ref_duration, lux_return_smooth,
            # Chatterbox params
            cb_pause_linebreak,
            cb_exaggeration, cb_cfg_weight, cb_temperature, cb_repetition_penalty, cb_top_p, cb_min_p, cb_max_new_tokens,
            # Shared
            seed,
            request: gr.Request = None,
            progress=gr.Progress()
        ):
            """Route to appropriate generation function based on model type."""
            yield (
                gr.update(value=None),
                "Generating...",
                gr.update(interactive=False),
                "",
                "",
                "",
            )
            voice_samples = prepare_voice_samples_dict(sv1, sv2, sv3, sv4, sv5, sv6, sv7, sv8, request=request)
            stream_selected = conversation_stream_supported(model_type, tts_manager)
            if stream_selected and model_type == "Qwen Base":
                qwen_size = "1.7B" if qwen_base_model_size == "Large" else "0.6B"
                stream_selected = tts_manager.qwen_runtime_streaming_available(model_size=qwen_size)

            def _generate_stream_impl():
                if model_type == "Qwen Base":
                    qwen_size = "1.7B" if qwen_base_model_size == "Large" else "0.6B"
                    yield from generate_conversation_base_stream_handler(
                        script,
                        voice_samples,
                        qwen_base_pause_linebreak,
                        qwen_base_pause_period,
                        qwen_base_pause_comma,
                        qwen_base_pause_question,
                        qwen_base_pause_hyphen,
                        qwen_lang,
                        qwen_seed,
                        qwen_size,
                        qwen_do_sample,
                        qwen_temperature,
                        qwen_top_k,
                        qwen_top_p,
                        qwen_repetition_penalty,
                        qwen_max_new_tokens,
                        emotion_intensity,
                        request,
                        progress,
                    )
                else:
                    if vv_model_size == "Small":
                        vv_size = "1.5B"
                    elif vv_model_size == "Large (4-bit)":
                        vv_size = "Large (4-bit)"
                    else:
                        vv_size = "Large"
                    yield from generate_vibevoice_longform_stream_handler(
                        script,
                        voice_samples,
                        vv_size,
                        vv_cfg,
                        seed,
                        vv_num_steps,
                        vv_do_sample,
                        vv_temperature,
                        vv_top_k,
                        vv_top_p,
                        vv_repetition_penalty,
                        vv_sentences_per_chunk,
                        request,
                        progress,
                    )

            def _generate_impl():
                if model_type == "Qwen Speakers":
                    qwen_size = "1.7B" if qwen_custom_model_size == "Large" else "0.6B"
                    result = generate_conversation_handler(script, qwen_custom_pause_linebreak, qwen_custom_pause_period,
                                                           qwen_custom_pause_comma, qwen_custom_pause_question,
                                                           qwen_custom_pause_hyphen, qwen_lang, qwen_seed, qwen_size,
                                                           qwen_do_sample, qwen_temperature, qwen_top_k, qwen_top_p,
                                                           qwen_repetition_penalty, qwen_max_new_tokens, request, progress)
                elif model_type == "Qwen Base":
                    qwen_size = "1.7B" if qwen_base_model_size == "Large" else "0.6B"
                    result = generate_conversation_base_handler(script, voice_samples, qwen_base_pause_linebreak,
                                                                qwen_base_pause_period, qwen_base_pause_comma,
                                                                qwen_base_pause_question, qwen_base_pause_hyphen,
                                                                qwen_lang, qwen_seed, qwen_size,
                                                                qwen_do_sample, qwen_temperature, qwen_top_k, qwen_top_p,
                                                                qwen_repetition_penalty, qwen_max_new_tokens,
                                                                emotion_intensity, request, progress)
                elif model_type == "LuxTTS":
                    result = generate_luxtts_conversation_handler(script, voice_samples, lux_pause_linebreak,
                                                                  seed, lux_num_steps, lux_t_shift, lux_speed,
                                                                  lux_guidance_scale, lux_rms, lux_ref_duration,
                                                                  lux_return_smooth, request, progress)
                elif model_type == "Chatterbox":
                    result = generate_chatterbox_conversation_handler(script, voice_samples, cb_pause_linebreak,
                                                                      qwen_lang, seed,
                                                                      cb_exaggeration, cb_cfg_weight, cb_temperature,
                                                                      cb_repetition_penalty, cb_top_p, cb_min_p, cb_max_new_tokens,
                                                                      request, progress)
                else:  # VibeVoice
                    if vv_model_size == "Small":
                        vv_size = "1.5B"
                    elif vv_model_size == "Large (4-bit)":
                        vv_size = "Large (4-bit)"
                    else:
                        vv_size = "Large"
                    result = generate_vibevoice_longform_handler(script, voice_samples, vv_size, vv_cfg, seed,
                                                                 vv_num_steps, vv_do_sample, vv_temperature, vv_top_k,
                                                                 vv_top_p, vv_repetition_penalty,
                                                                 vv_sentences_per_chunk, request, progress)

                audio_value, status, suggested_name, metadata_text = result
                if not stream_selected:
                    status = prefix_non_stream_status(status)
                return (
                    audio_value,
                    status,
                    gr.update(interactive=bool(audio_value)),
                    suggested_name,
                    metadata_text,
                    audio_value,
                )

            try:
                if stream_selected:
                    if run_heavy_stream_job:
                        yield from run_heavy_stream_job("conversation_generate_stream", _generate_stream_impl, request=request)
                        return
                    yield from _generate_stream_impl()
                    return
                if run_heavy_job:
                    yield run_heavy_job("conversation_generate", _generate_impl, request=request)
                    return
                yield _generate_impl()
            except MemoryAdmissionError as exc:
                yield (
                    None,
                    f"⚠ Memory safety guard rejected request: {str(exc)}",
                    gr.update(interactive=False),
                    "",
                    "",
                    "",
                )

        # Event handlers
        if components.get('prompt_assistant'):
            wire_prompt_assistant_events(
                assistant=components['prompt_assistant'],
                target_components={"conversation.script": components['conversation_script']},
                status_component=components['conv_status'],
                shared_state=shared_state,
            )

        if prompt_apply_trigger is not None:
            wire_prompt_apply_listener(
                prompt_apply_trigger=prompt_apply_trigger,
                target_components={"conversation.script": components['conversation_script']},
                status_component=components['conv_status'],
            )

        wire_audio_route_dropdown_refresh(
            components['conv_tab'],
            components['conv_route_target'],
            shared_state,
        )
        wire_audio_route_source(
            send_button=components['conv_route_btn'],
            target_dropdown=components['conv_route_target'],
            audio_value_component=components['conv_output_audio_path'],
            status_component=components['conv_status'],
            shared_state=shared_state,
            source_label="Conversation",
        )

        conv_generate_event = components['conv_generate_btn'].click(
            unified_conversation_generate,
            inputs=[
                components['conv_model_type'], components['conversation_script'],
                # Shared voice samples
                components['conv_voice_1'], components['conv_voice_2'], components['conv_voice_3'], components['conv_voice_4'],
                components['conv_voice_5'], components['conv_voice_6'], components['conv_voice_7'], components['conv_voice_8'],
                # Qwen Speakers
                components['conv_pause_linebreak'], components['conv_pause_period'], components['conv_pause_comma'],
                components['conv_pause_question'], components['conv_pause_hyphen'], components['conv_model_size'],
                # Qwen Base
                components['conv_pause_linebreak'], components['conv_pause_period'], components['conv_pause_comma'],
                components['conv_pause_question'], components['conv_pause_hyphen'], components['conv_base_model_size'],
                # Shared Qwen
                components['conv_language'], components['conv_seed'], components['conv_emotion_intensity'],
                # Qwen advanced params
                components['qwen_conv_do_sample'], components['qwen_conv_temperature'], components['qwen_conv_top_k'], components['qwen_conv_top_p'],
                components['qwen_conv_repetition_penalty'], components['qwen_conv_max_new_tokens'],
                # VibeVoice
                components['longform_model_size'], components['longform_cfg_scale'],
                # VibeVoice advanced params
                components['vv_conv_num_steps'], components['vv_conv_do_sample'], components['vv_conv_temperature'], components['vv_conv_top_k'],
                components['vv_conv_top_p'], components['vv_conv_repetition_penalty'],
                components['vv_conv_sentences_per_chunk'],
                # LuxTTS
                components['luxtts_pause_linebreak'],
                components['lux_conv_num_steps'], components['lux_conv_t_shift'], components['lux_conv_speed'], components['lux_conv_guidance_scale'],
                components['lux_conv_rms'], components['lux_conv_ref_duration'], components['lux_conv_return_smooth'],
                # Chatterbox
                components['cb_pause_linebreak'],
                components['cb_conv_exaggeration'], components['cb_conv_cfg_weight'], components['cb_conv_temperature'],
                components['cb_conv_repetition_penalty'], components['cb_conv_top_p'],
                components['cb_conv_min_p'], components['cb_conv_max_new_tokens'],
                # Shared
                components['conv_seed']
            ],
            outputs=[
                components['conv_output_audio'],
                components['conv_status'],
                components['conv_save_btn'],
                components['conv_suggested_name'],
                components['conv_metadata_text'],
                components['conv_output_audio_path'],
            ]
        )

        components['conv_stop_btn'].click(
            lambda: ("Generation stopped.", gr.update(interactive=False)),
            inputs=[],
            outputs=[components['conv_status'], components['conv_save_btn']],
            cancels=[conv_generate_event],
            queue=False,
        )

        def apply_conversation_output_pipeline(audio_path, enable_denoise, enable_remove_silences, enable_normalize, enable_mono, request: gr.Request):
            pipeline = OutputAudioPipelineConfig(
                enable_denoise=bool(enable_denoise),
                enable_remove_silences=bool(enable_remove_silences),
                enable_normalize=bool(enable_normalize),
                enable_mono=bool(enable_mono),
            )
            updated_audio, status = apply_generation_output_pipeline(
                audio_path,
                pipeline,
                deepfilter_available=deepfilter_available,
                denoise_step=lambda path: clean_audio(path),
                remove_silences_step=lambda path: remove_silences(path, request=request),
                normalize_step=lambda path: normalize_audio(path, request=request),
                mono_step=lambda path: convert_to_mono(path, request=request),
            )
            if not updated_audio:
                return gr.update(), status, audio_path
            return gr.update(value=updated_audio), status, updated_audio

        components['conv_output_apply_pipeline_btn'].click(
            apply_conversation_output_pipeline,
            inputs=[
                components['conv_output_audio_path'],
                components['conv_output_enable_denoise'],
                components['conv_output_enable_remove_silences'],
                components['conv_output_enable_normalize'],
                components['conv_output_enable_mono'],
            ],
            outputs=[components['conv_output_audio'], components['conv_status'], components['conv_output_audio_path']],
        )

        save_conv_modal_js = show_input_modal_js(
            title="Save Conversation Output",
            message="Enter a filename for this generated conversation:",
            placeholder="e.g., my_conversation_take",
            context="save_conv_"
        )

        def get_conv_existing_files(request: gr.Request):
            output_dir = get_tenant_output_dir(request=request, strict=True)
            return json.dumps(get_existing_wav_stems(output_dir))

        save_conv_js = f"""
        (existingFilesJson, suggestedName) => {{
            try {{
                window.inputModalExistingFiles = JSON.parse(existingFilesJson || '[]');
            }} catch(e) {{
                window.inputModalExistingFiles = [];
            }}
            const openModal = {save_conv_modal_js};
            openModal(suggestedName);
        }}
        """

        components['conv_save_btn'].click(
            fn=get_conv_existing_files,
            inputs=[],
            outputs=[components['conv_existing_files_json']],
        ).then(
            fn=None,
            inputs=[components['conv_existing_files_json'], components['conv_suggested_name']],
            js=save_conv_js,
        )

        def handle_conv_save_input(input_value, output_audio_path, metadata_text, request: gr.Request):
            matched, cancelled, chosen_name = parse_modal_submission(input_value, "save_conv_")
            if not matched:
                return gr.update(), gr.update()
            if cancelled:
                return gr.update(), gr.update()
            if not chosen_name or not chosen_name.strip():
                return "❌ Please enter a filename.", gr.update()
            if not output_audio_path:
                return "❌ No generated audio to save.", gr.update(interactive=False)

            try:
                output_dir = get_tenant_output_dir(request=request, strict=True)
                output_path = save_generated_output(
                    audio_value=output_audio_path,
                    output_dir=output_dir,
                    raw_name=chosen_name,
                    metadata_text=metadata_text,
                    default_sample_rate=24000,
                )
                return f"Saved: {output_path.name}", gr.update(interactive=False)
            except Exception as e:
                return f"❌ Save failed: {str(e)}", gr.update()

        input_trigger.change(
            handle_conv_save_input,
            inputs=[input_trigger, components['conv_output_audio_path'], components['conv_metadata_text']],
            outputs=[components['conv_status'], components['conv_save_btn']]
        )

        # Toggle UI based on model selection
        def toggle_conv_ui(model_type, show_expert):
            from modules.core_components.constants import CHATTERBOX_LANGUAGES
            is_qwen_custom = model_type == "Qwen Speakers"
            is_qwen_base = model_type == "Qwen Base"
            is_vibevoice = model_type == "VibeVoice"
            is_luxtts = model_type == "LuxTTS"
            is_chatterbox = model_type == "Chatterbox"
            is_qwen = is_qwen_custom or is_qwen_base
            uses_samples = is_qwen_base or is_luxtts or is_chatterbox or is_vibevoice
            show_cb_expert = bool(show_expert) and is_chatterbox

            # Language dropdown: full list for Qwen, Chatterbox languages for Chatterbox, Auto-only for VV/LuxTTS
            if is_qwen:
                lang_update = gr.update(choices=LANGUAGES, value=LANGUAGES[0])
            elif is_chatterbox:
                lang_update = gr.update(choices=CHATTERBOX_LANGUAGES, value="English")
            else:
                lang_update = gr.update(choices=["Auto"], value="Auto")

            return {
                components['qwen_speaker_table']: gr.update(visible=is_qwen_custom),
                components['shared_voices_section']: gr.update(visible=uses_samples),
                components['conv_voices_row_5_6']: gr.update(visible=uses_samples and not is_vibevoice),
                components['conv_voices_row_7_8']: gr.update(visible=uses_samples and not is_vibevoice),
                components['qwen_custom_settings']: gr.update(visible=is_qwen_custom),
                components['qwen_base_settings']: gr.update(visible=is_qwen_base),
                components['qwen_pause_controls']: gr.update(visible=is_qwen),
                components['conv_emotion_intensity_row']: gr.update(visible=is_qwen_base),
                components['qwen_conv_advanced']: gr.update(visible=is_qwen),
                components['luxtts_settings']: gr.update(visible=is_luxtts),
                components['vibevoice_settings']: gr.update(visible=is_vibevoice),
                components['cb_settings']: gr.update(visible=is_chatterbox),
                components['qwen_custom_tips']: gr.update(visible=is_qwen_custom),
                components['qwen_base_tips']: gr.update(visible=is_qwen_base),
                components['vibevoice_tips']: gr.update(visible=is_vibevoice),
                components['luxtts_tips']: gr.update(visible=is_luxtts),
                components['cb_tips']: gr.update(visible=is_chatterbox),
                components['cb_conv_expert_row']: gr.update(visible=show_cb_expert),
                components['cb_conv_accordion']: gr.update(open=show_cb_expert),
                components['conv_expert_note']: gr.update(visible=(bool(show_expert) and not is_chatterbox)),
                components['conv_language']: lang_update,
            }

        components['conv_model_type'].change(
            toggle_conv_ui,
            inputs=[components['conv_model_type'], components['conv_show_expert_params']],
            outputs=[components['qwen_speaker_table'],
                     components['shared_voices_section'], components['conv_voices_row_5_6'], components['conv_voices_row_7_8'],
                     components['qwen_custom_settings'], components['qwen_base_settings'], components['qwen_pause_controls'],
                     components['conv_emotion_intensity_row'], components['qwen_conv_advanced'],
                     components['luxtts_settings'],
                     components['vibevoice_settings'],
                     components['cb_settings'],
                     components['qwen_custom_tips'], components['qwen_base_tips'], components['vibevoice_tips'],
                     components['luxtts_tips'], components['cb_tips'], components['cb_conv_expert_row'],
                     components['cb_conv_accordion'], components['conv_expert_note'],
                     components['conv_language']]
        )

        components['conv_show_expert_params'].change(
            lambda enabled: save_preference("conversation_show_expert_params", bool(enabled)),
            inputs=[components['conv_show_expert_params']],
            outputs=[],
        )

        components['conv_show_expert_params'].change(
            lambda model_type, enabled: (
                gr.update(visible=(model_type == "Chatterbox" and bool(enabled))),
                gr.update(open=(model_type == "Chatterbox" and bool(enabled))),
                gr.update(visible=(bool(enabled) and model_type != "Chatterbox")),
            ),
            inputs=[components['conv_model_type'], components['conv_show_expert_params']],
            outputs=[components['cb_conv_expert_row'], components['cb_conv_accordion'], components['conv_expert_note']],
        )

        # Refresh voice samples handler
        def refresh_all_voice_samples(request: gr.Request):
            """Refresh all shared voice sample dropdowns."""
            updated_samples = get_sample_choices(request=request, strict=True)
            update = gr.update(choices=updated_samples)
            return [update] * 8

        # Auto-refresh all voice sample dropdowns when tab is selected
        components['conv_tab'].select(
            refresh_all_voice_samples,
            outputs=[
                components['conv_voice_1'], components['conv_voice_2'],
                components['conv_voice_3'], components['conv_voice_4'],
                components['conv_voice_5'], components['conv_voice_6'],
                components['conv_voice_7'], components['conv_voice_8'],
            ]
        )

        # Save preferences
        components['conv_model_type'].change(
            lambda x: save_preference("conv_model_type", x),
            inputs=[components['conv_model_type']],
            outputs=[]
        )

        components['conv_model_size'].change(
            lambda x: save_preference("conv_model_size", x),
            inputs=[components['conv_model_size']],
            outputs=[]
        )

        components['conv_base_model_size'].change(
            lambda x: save_preference("conv_base_model_size", x),
            inputs=[components['conv_base_model_size']],
            outputs=[]
        )

        components['longform_model_size'].change(
            lambda x: save_preference("vibevoice_model_size", x),
            inputs=[components['longform_model_size']],
            outputs=[]
        )

        components['conv_language'].change(
            lambda x: save_preference("language", x),
            inputs=[components['conv_language']],
            outputs=[]
        )

        # Save conversation pause preferences
        components['conv_pause_linebreak'].change(
            lambda x: save_preference("conv_pause_linebreak", x),
            inputs=[components['conv_pause_linebreak']],
            outputs=[]
        )

        components['conv_pause_period'].change(
            lambda x: save_preference("conv_pause_period", x),
            inputs=[components['conv_pause_period']],
            outputs=[]
        )

        components['conv_pause_comma'].change(
            lambda x: save_preference("conv_pause_comma", x),
            inputs=[components['conv_pause_comma']],
            outputs=[]
        )

        components['conv_pause_question'].change(
            lambda x: save_preference("conv_pause_question", x),
            inputs=[components['conv_pause_question']],
            outputs=[]
        )

        components['conv_pause_hyphen'].change(
            lambda x: save_preference("conv_pause_hyphen", x),
            inputs=[components['conv_pause_hyphen']],
            outputs=[]
        )

        components['cb_conv_min_p'].change(
            lambda v: save_preference("chatterbox_min_p", float(v) if v is not None else 0.05),
            inputs=[components['cb_conv_min_p']],
            outputs=[],
        )

        components['cb_conv_max_new_tokens'].change(
            lambda v: save_preference("chatterbox_max_new_tokens", int(v) if v is not None else 2048),
            inputs=[components['cb_conv_max_new_tokens']],
            outputs=[],
        )


# Export for tab registry
get_tool_class = lambda: ConversationTool


if __name__ == "__main__":
    """Standalone testing of Conversation tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(ConversationTool, port=7864, title="Conversation - Standalone")
