"""
Voice Clone Tab

Clone voices from samples using Qwen3-TTS, VibeVoice, LuxTTS, or Chatterbox.
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    # Also add modules directory for vibevoice_tts imports
    sys.path.insert(0, str(project_root / "modules"))

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import random
import json
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from modules.core_components.ai_models.model_utils import set_seed
from modules.core_components.runtime import MemoryAdmissionError

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
    prefix_non_stream_status,
    voice_clone_stream_supported,
)
from gradio_filelister import FileLister

class VoiceCloneTool(Tool):
    """Voice Clone tool implementation."""

    config = ToolConfig(
        name="Voice Clone",
        module_name="tool_voice_clone",
        description="Clone voices from voice samples",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Voice Clone tool UI."""
        components = {}

        # Get helper functions and config
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        load_sample_details = shared_state['load_sample_details']
        get_emotion_choices = shared_state['get_emotion_choices']
        get_prompt_cache_path = shared_state['get_prompt_cache_path']
        LANGUAGES = shared_state['LANGUAGES']
        VOICE_CLONE_OPTIONS = shared_state['VOICE_CLONE_OPTIONS']
        TTS_ENGINES = shared_state.get('TTS_ENGINES', {})
        _user_config = shared_state['_user_config']
        _active_emotions = shared_state['_active_emotions']
        deepfilter_available = bool(shared_state.get("DEEPFILTER_AVAILABLE", False))
        expert_visible = bool(_user_config.get("voice_clone_show_expert_params", False))
        initial_cb_min_p = float(_user_config.get("chatterbox_min_p", 0.05))
        initial_cb_max_new_tokens = int(_user_config.get("chatterbox_max_new_tokens", 2048))
        initial_audio_route_choices = shared_state.get("audio_route_get_initial_targets", lambda: [])()

        # Filter voice clone options based on enabled engines
        engine_settings = _user_config.get("enabled_engines", {})
        visible_options = []
        for engine_key, engine_info in TTS_ENGINES.items():
            if engine_settings.get(engine_key, engine_info.get("default_enabled", True)):
                visible_options.extend(engine_info["choices"])
        # Fall back to all options if nothing is enabled (safety)
        if not visible_options:
            visible_options = VOICE_CLONE_OPTIONS

        # Resolve preferred/default model based on engine preferences and visibility
        from modules.core_components.constants import resolve_preferred_tts_engine_and_model, coerce_choice_value
        _preferred_engine, preferred_model = resolve_preferred_tts_engine_and_model(_user_config)
        saved_model = coerce_choice_value(_user_config.get("voice_clone_model", preferred_model))
        saved_model = str(saved_model) if saved_model is not None else preferred_model
        if saved_model not in visible_options:
            saved_model = visible_options[0]
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_emotion_handler = shared_state['save_emotion_handler']
        delete_emotion_handler = shared_state['delete_emotion_handler']
        save_preference = shared_state['save_preference']
        refresh_samples = shared_state['refresh_samples']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']

        with gr.TabItem("Voice Clone", id="tab_voice_clone") as voice_clone_tab:
            components['voice_clone_tab'] = voice_clone_tab
            gr.Markdown("Clone voices from tenant samples. <small>(Create/manage them in Library Manager)</small>")
            with gr.Row():
                # Left column - Sample selection (1/3 width)
                with gr.Column(scale=1):
                    gr.Markdown("### Voice Sample")

                    components['sample_lister'] = FileLister(
                        value=get_sample_choices(),
                        height=200,
                        show_footer=False,
                        interactive=True,
                    )

                    with gr.Row():
                        pass

                    components['sample_audio'] = gr.Audio(
                        label="Sample Preview",
                        type="filepath",
                        interactive=False,
                        visible=True,
                        value=None,
                        elem_id="voice-clone-sample-audio"
                    )

                    components['sample_text'] = gr.Textbox(
                        label="Sample Text",
                        interactive=False,
                        max_lines=10,
                        value=None
                    )

                    components['sample_info'] = gr.Textbox(
                        label="Info",
                        interactive=False,
                        max_lines=10,
                        value=None
                    )

                # Right column - Generation (2/3 width)
                with gr.Column(scale=3):
                    gr.Markdown("### Generate Speech")

                    components['text_input'] = gr.Textbox(
                        label="Text to Generate",
                        placeholder="Enter the text you want to speak in the cloned voice...",
                        lines=6
                    )
                    components['prompt_assistant'] = create_prompt_assistant(
                        shared_state=shared_state,
                        target_ids=["voice_clone.text"],
                        default_target_id="voice_clone.text",
                    )

                    with gr.Row():
                        components['clone_model_dropdown'] = gr.Dropdown(
                            choices=visible_options,
                            value=saved_model,
                            label="Engine & Model",
                            scale=4
                        )
                        components['seed_input'] = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0,
                            scale=1
                        )

                    # Determine which model is initially selected to conditionally show/hide parameters
                    is_qwen_initial = "Qwen" in saved_model

                    # Language dropdown (hidden for VibeVoice models)
                    components['language_row'] = gr.Row(visible=is_qwen_initial)
                    with components['language_row']:
                        components['language_dropdown'] = gr.Dropdown(
                            choices=LANGUAGES,
                            value=_user_config.get("language", "Auto"),
                            label="Language",
                        )

                    # Qwen3 Advanced Parameters (create_qwen_advanced_params includes its own accordion)
                    create_qwen_advanced_params = shared_state['create_qwen_advanced_params']

                    qwen_params = create_qwen_advanced_params(
                        emotions_dict=_active_emotions,
                        include_emotion=True,
                        initial_emotion="(None)",
                        initial_intensity=1.0,
                        visible=is_qwen_initial,
                        emotion_visible=True,
                        shared_state=shared_state
                    )

                    # Store the accordion reference for toggling
                    components['qwen_params_accordion'] = qwen_params.get('accordion')

                    # Store references
                    components['qwen_params'] = qwen_params
                    components['qwen_emotion_preset'] = qwen_params['emotion_preset']
                    components['qwen_emotion_intensity'] = qwen_params['emotion_intensity']
                    components['qwen_save_emotion_btn'] = qwen_params.get('save_emotion_btn')
                    components['qwen_delete_emotion_btn'] = qwen_params.get('delete_emotion_btn')
                    components['qwen_emotion_save_name'] = qwen_params.get('emotion_save_name')
                    components['qwen_do_sample'] = qwen_params['do_sample']
                    components['qwen_temperature'] = qwen_params['temperature']
                    components['qwen_top_k'] = qwen_params['top_k']
                    components['qwen_top_p'] = qwen_params['top_p']
                    components['qwen_repetition_penalty'] = qwen_params['repetition_penalty']
                    components['qwen_max_new_tokens'] = qwen_params['max_new_tokens']

                    # VibeVoice Advanced Parameters
                    is_vv_initial = "VibeVoice" in saved_model
                    create_vibevoice_advanced_params = shared_state['create_vibevoice_advanced_params']
                    vv_params = create_vibevoice_advanced_params(
                        include_sentences_per_chunk=True,
                        visible=is_vv_initial
                    )
                    components['vv_params_accordion'] = vv_params['accordion']
                    components['vv_cfg_scale'] = vv_params['cfg_scale']
                    components['vv_num_steps'] = vv_params['num_steps']
                    components['vv_do_sample'] = vv_params['do_sample']
                    components['vv_sentences_per_chunk'] = vv_params['sentences_per_chunk']
                    components['vv_repetition_penalty'] = vv_params['repetition_penalty']
                    components['vv_temperature'] = vv_params['temperature']
                    components['vv_top_k'] = vv_params['top_k']
                    components['vv_top_p'] = vv_params['top_p']

                    # LuxTTS Advanced Parameters
                    is_lux_initial = "LuxTTS" in saved_model
                    create_luxtts_advanced_params = shared_state['create_luxtts_advanced_params']
                    luxtts_params = create_luxtts_advanced_params(visible=is_lux_initial)
                    components['luxtts_params_accordion'] = luxtts_params.get('accordion')
                    components['luxtts_num_steps'] = luxtts_params['num_steps']
                    components['luxtts_t_shift'] = luxtts_params['t_shift']
                    components['luxtts_speed'] = luxtts_params['speed']
                    components['luxtts_return_smooth'] = luxtts_params['return_smooth']
                    components['luxtts_rms'] = luxtts_params['rms']
                    components['luxtts_ref_duration'] = luxtts_params['ref_duration']
                    components['luxtts_guidance_scale'] = luxtts_params['guidance_scale']

                    # Chatterbox Advanced Parameters
                    is_cb_initial = "Chatterbox" in saved_model
                    create_chatterbox_advanced_params = shared_state['create_chatterbox_advanced_params']
                    cb_params = create_chatterbox_advanced_params(
                        visible=is_cb_initial,
                        include_expert=True,
                        initial_min_p=initial_cb_min_p,
                        initial_max_new_tokens=initial_cb_max_new_tokens,
                        expert_visible=(expert_visible and is_cb_initial),
                    )
                    components['cb_params_accordion'] = cb_params['accordion']
                    components['cb_exaggeration'] = cb_params['exaggeration']
                    components['cb_cfg_weight'] = cb_params['cfg_weight']
                    components['cb_temperature'] = cb_params['temperature']
                    components['cb_repetition_penalty'] = cb_params['repetition_penalty']
                    components['cb_top_p'] = cb_params['top_p']
                    components['cb_min_p'] = cb_params['min_p']
                    components['cb_max_new_tokens'] = cb_params['max_new_tokens']
                    components['cb_expert_row'] = cb_params['expert_row']

                    # Chatterbox Multilingual language dropdown (only shown for Chatterbox - Multilingual)
                    is_cb_mtl_initial = "Multilingual" in saved_model
                    components['cb_language_row'] = gr.Row(visible=is_cb_mtl_initial)
                    with components['cb_language_row']:
                        from modules.core_components.constants import CHATTERBOX_LANGUAGES
                        components['cb_language_dropdown'] = gr.Dropdown(
                            choices=CHATTERBOX_LANGUAGES,
                            value="English",
                            label="Language (Chatterbox Multilingual)",
                        )

                    components['vc_show_expert_params'] = gr.Checkbox(
                        label="Show Expert Parameters",
                        value=expert_visible,
                        info="Reveal advanced controls for selected engines",
                    )

                    with gr.Row():
                        components['generate_btn'] = gr.Button("Generate Audio", variant="primary", size="lg")
                        components['stop_btn'] = gr.Button("Stop", variant="stop", size="lg")

                    components['output_audio'] = gr.Audio(
                        label="Generated Audio",
                        type="filepath",
                        interactive=False,
                        streaming=True,
                        autoplay=True,
                    )
                    with gr.Row():
                        components['output_enable_denoise'] = gr.Checkbox(
                            label="Enable Denoise",
                            value=False,
                            visible=deepfilter_available,
                        )
                        components['output_enable_remove_silences'] = gr.Checkbox(
                            label="Remove Silences",
                            value=False,
                        )
                        components['output_enable_normalize'] = gr.Checkbox(
                            label="Enable Normalize",
                            value=False,
                        )
                        components['output_enable_mono'] = gr.Checkbox(
                            label="Enable Mono",
                            value=False,
                        )
                        components['output_apply_pipeline_btn'] = gr.Button(
                            "Apply Pipeline",
                            variant="secondary",
                            size="sm",
                        )
                    components['save_btn'] = gr.Button("Save to Output", variant="primary", interactive=False)
                    route_controls = create_audio_route_controls(initial_audio_route_choices)
                    components['route_target'] = route_controls['target_dropdown']
                    components['route_btn'] = route_controls['send_button']
                    components['suggested_name'] = gr.State(value="")
                    components['metadata_text'] = gr.State(value="")
                    components['output_audio_path'] = gr.State(value="")
                    components['existing_files_json'] = gr.State(value="[]")

                    components['clone_status'] = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=5)

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Clone tab events."""

        # Get helper functions and directories
        get_sample_choices = shared_state['get_sample_choices']
        get_available_samples = shared_state['get_available_samples']
        load_sample_details = shared_state['load_sample_details']
        get_prompt_cache_path = shared_state['get_prompt_cache_path']
        get_or_create_voice_prompt = shared_state['get_or_create_voice_prompt']
        configure_tts_manager_for_tenant = shared_state.get('configure_tts_manager_for_tenant')
        refresh_samples = shared_state['refresh_samples']
        show_input_modal_js = shared_state['show_input_modal_js']
        show_confirmation_modal_js = shared_state['show_confirmation_modal_js']
        save_emotion_handler = shared_state['save_emotion_handler']
        delete_emotion_handler = shared_state['delete_emotion_handler']
        save_preference = shared_state['save_preference']
        confirm_trigger = shared_state['confirm_trigger']
        input_trigger = shared_state['input_trigger']
        prompt_apply_trigger = shared_state.get('prompt_apply_trigger')
        get_tenant_output_dir = shared_state['get_tenant_output_dir']
        get_tenant_paths = shared_state['get_tenant_paths']
        play_completion_beep = shared_state.get('play_completion_beep')
        run_heavy_job = shared_state.get('run_heavy_job')
        run_heavy_stream_job = shared_state.get('run_heavy_stream_job')
        normalize_audio = shared_state['normalize_audio']
        remove_silences = shared_state['remove_silences']
        convert_to_mono = shared_state['convert_to_mono']
        clean_audio = shared_state['clean_audio']
        deepfilter_available = bool(shared_state.get("DEEPFILTER_AVAILABLE", False))

        # Get TTS manager (singleton)
        tts_manager = get_tts_manager()

        def get_selected_sample_name(lister_value):
            """Extract selected sample name from FileLister value (strips .wav extension)."""
            if not lister_value:
                return None
            selected = lister_value.get("selected", [])
            if len(selected) == 1:
                from modules.core_components.tools import strip_sample_extension
                return strip_sample_extension(selected[0])
            return None

        def generate_audio_handler(sample_name, text_to_generate, language, seed, model_selection="Qwen3 - Small",
                                   qwen_do_sample=True, qwen_temperature=0.9, qwen_top_k=50, qwen_top_p=1.0, qwen_repetition_penalty=1.05,
                                   qwen_max_new_tokens=2048,
                                   vv_do_sample=False, vv_temperature=1.0, vv_top_k=50, vv_top_p=1.0, vv_repetition_penalty=1.1,
                                   vv_cfg_scale=3.0, vv_num_steps=20, vv_sentences_per_chunk=0,
                                   lux_num_steps=4, lux_t_shift=0.5, lux_speed=1.0, lux_return_smooth=False,
                                   lux_rms=0.01, lux_ref_duration=30, lux_guidance_scale=3.0,
                                   cb_exaggeration=0.5, cb_cfg_weight=0.5, cb_temperature=0.8,
                                   cb_repetition_penalty=1.2, cb_top_p=1.0, cb_min_p=0.05, cb_max_new_tokens=2048,
                                   cb_language="English",
                                   request: gr.Request = None,
                                   progress=gr.Progress()):
            """Generate audio using voice cloning - supports Qwen, VibeVoice, and LuxTTS engines."""
            if not sample_name:
                return None, "Please select a voice sample first.", gr.update(interactive=False), "", "", ""

            if not text_to_generate or not text_to_generate.strip():
                return None, "Please enter text to generate.", gr.update(interactive=False), "", "", ""

            # Parse model selection to determine engine and size
            if "LuxTTS" in model_selection:
                engine = "luxtts"
                model_size = "Default"
            elif "VibeVoice" in model_selection:
                engine = "vibevoice"
                if "Small" in model_selection:
                    model_size = "1.5B"
                elif "4-bit" in model_selection:
                    model_size = "Large (4-bit)"
                else:  # Large
                    model_size = "Large"
            elif "Chatterbox" in model_selection:
                engine = "chatterbox"
                if "Multilingual" in model_selection:
                    model_size = "Multilingual"
                else:
                    model_size = "Default"
            else:  # Qwen3
                engine = "qwen"
                if "Small" in model_selection:
                    model_size = "0.6B"
                else:  # Large
                    model_size = "1.7B"

            # Find the selected sample
            samples = get_available_samples(request=request, strict=True)
            sample = None
            for s in samples:
                if s["name"] == sample_name:
                    sample = s
                    break

            if not sample:
                return None, f"❌ Sample '{sample_name}' not found.", gr.update(interactive=False), "", "", ""

            # Check that sample has a transcript (required for all engines)
            sample_ref_text = sample.get("ref_text") or sample.get("meta", {}).get("Text", "")
            if not sample_ref_text.strip():
                return None, (
                    f"❌ No transcript found for sample '{sample_name}'.\n\n"
                    "Please transcribe this sample first in the **Library Manager** tab "
                    "(using Whisper or VibeVoice ASR), then try again."
                ), gr.update(interactive=False), "", "", ""

            stream_supported = voice_clone_stream_supported(model_selection, tts_manager)
            if engine == "qwen" and stream_supported:
                stream_supported = tts_manager.qwen_runtime_streaming_available(model_size=model_size)
            should_stream = engine in {"qwen", "vibevoice"} and stream_supported
            add_non_stream_notice = not should_stream

            def _generate_impl():
                # Set the seed for reproducibility
                resolved_seed = int(seed) if seed is not None else -1
                if resolved_seed < 0:
                    resolved_seed = random.randint(0, 2147483647)

                set_seed(resolved_seed)
                seed_msg = f"Seed: {resolved_seed}"
                if configure_tts_manager_for_tenant:
                    configure_tts_manager_for_tenant(request=request, strict=True)

                if engine == "qwen":
                    # Qwen engine - uses cached prompts
                    progress(0.1, desc=f"Loading Qwen3 model ({model_size})...")

                    # Get or create the voice prompt (with caching)
                    model = tts_manager.get_qwen3_base(model_size)
                    prompt_items, was_cached = get_or_create_voice_prompt(
                        model=model,
                        sample_name=sample_name,
                        wav_path=sample["wav_path"],
                        ref_text=sample["ref_text"],
                        model_size=model_size,
                        progress_callback=progress,
                        request=request,
                    )

                    cache_status = "cached" if was_cached else "newly processed"
                    progress(0.6, desc=f"Generating audio ({cache_status} prompt)...")

                    # Generate using manager method
                    audio_data, sr = tts_manager.generate_voice_clone_qwen(
                        text=text_to_generate,
                        language=language,
                        prompt_items=prompt_items,
                        seed=resolved_seed,
                        do_sample=qwen_do_sample,
                        temperature=qwen_temperature,
                        top_k=qwen_top_k,
                        top_p=qwen_top_p,
                        repetition_penalty=qwen_repetition_penalty,
                        max_new_tokens=qwen_max_new_tokens,
                        model_size=model_size
                    )
                    wavs = [audio_data]

                    engine_display = f"Qwen3-{model_size}"

                elif engine == "vibevoice":
                    progress(0.1, desc=f"Loading VibeVoice model ({model_size})...")

                    # Generate using manager method
                    audio_data, sr = tts_manager.generate_voice_clone_vibevoice(
                        text=text_to_generate,
                        voice_sample_path=sample["wav_path"],
                        seed=resolved_seed,
                        do_sample=vv_do_sample,
                        temperature=vv_temperature,
                        top_k=vv_top_k,
                        top_p=vv_top_p,
                        repetition_penalty=vv_repetition_penalty,
                        cfg_scale=vv_cfg_scale,
                        num_steps=vv_num_steps,
                        sentences_per_chunk=int(vv_sentences_per_chunk),
                        model_size=model_size,
                        user_config=shared_state.get('_user_config', {})
                    )
                    wavs = [audio_data]

                    engine_display = f"VibeVoice-{model_size}"
                    cache_status = "no caching (VibeVoice)"

                elif engine == "luxtts":
                    progress(0.05, desc="Loading LuxTTS model...")

                    # Generate using manager method (with prompt caching)
                    audio_data, sr, was_cached = tts_manager.generate_voice_clone_luxtts(
                        text=text_to_generate,
                        voice_sample_path=sample["wav_path"],
                        sample_name=sample_name,
                        num_steps=int(lux_num_steps),
                        t_shift=float(lux_t_shift),
                        speed=float(lux_speed),
                        return_smooth=bool(lux_return_smooth),
                        rms=float(lux_rms),
                        ref_duration=int(lux_ref_duration),
                        guidance_scale=float(lux_guidance_scale),
                        seed=resolved_seed,
                        ref_text=sample.get("ref_text") or sample.get("meta", {}).get("Text"),
                        progress_callback=progress,
                    )
                    wavs = [audio_data]

                    engine_display = "LuxTTS"
                    cache_status = "cached" if was_cached else "newly processed"

                elif engine == "chatterbox":
                    cb_min_p = max(0.0, min(0.30, float(cb_min_p)))
                    cb_max_new_tokens = max(256, min(4096, int(cb_max_new_tokens)))
                    if model_size == "Multilingual":
                        progress(0.1, desc="Loading Chatterbox Multilingual model...")
                        from modules.core_components.constants import CHATTERBOX_LANG_TO_CODE
                        lang_code = CHATTERBOX_LANG_TO_CODE.get(cb_language, "en")
                        audio_data, sr = tts_manager.generate_voice_clone_chatterbox_multilingual(
                            text=text_to_generate,
                            language_code=lang_code,
                            voice_sample_path=sample["wav_path"],
                            seed=resolved_seed,
                            exaggeration=float(cb_exaggeration),
                            cfg_weight=float(cb_cfg_weight),
                            temperature=float(cb_temperature),
                            repetition_penalty=float(cb_repetition_penalty),
                            top_p=float(cb_top_p),
                            min_p=cb_min_p,
                            max_new_tokens=cb_max_new_tokens,
                        )
                        engine_display = "Chatterbox Multilingual"
                    else:
                        progress(0.1, desc="Loading Chatterbox TTS model...")
                        audio_data, sr = tts_manager.generate_voice_clone_chatterbox(
                            text=text_to_generate,
                            voice_sample_path=sample["wav_path"],
                            seed=resolved_seed,
                            exaggeration=float(cb_exaggeration),
                            cfg_weight=float(cb_cfg_weight),
                            temperature=float(cb_temperature),
                            repetition_penalty=float(cb_repetition_penalty),
                            top_p=float(cb_top_p),
                            min_p=cb_min_p,
                            max_new_tokens=cb_max_new_tokens,
                        )
                        engine_display = "Chatterbox"

                    wavs = [audio_data]
                    cache_status = "no caching (Chatterbox)"

                progress(0.8, desc="Saving preview...")
                # Generate preview filename in tenant temp directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = "".join(c if c.isalnum() else "_" for c in sample_name)
                tenant_paths = get_tenant_paths(request=request, strict=True)
                temp_dir = tenant_paths.temp_dir
                temp_dir.mkdir(parents=True, exist_ok=True)
                output_file = temp_dir / f"voice_clone_{safe_name}_{timestamp}.wav"

                sf.write(str(output_file), wavs[0], sr)

                chatterbox_extra = ""
                if engine == "chatterbox":
                    chatterbox_extra = f"Min-p: {cb_min_p} | Max New Tokens: {cb_max_new_tokens}"

                metadata = dedent(f"""\
                    Generated: {timestamp}
                    Sample: {sample_name}
                    Engine: {engine_display}
                    Language: {language}
                    Seed: {resolved_seed}
                    {chatterbox_extra}
                    Text: {' '.join(text_to_generate.split())}
                    """)
                metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                status_msg = f"Generated using {engine_display}. {cache_status}\n{seed_msg}\nReady to save."
                if add_non_stream_notice:
                    status_msg = prefix_non_stream_status(status_msg)
                return (
                    str(output_file),
                    status_msg,
                    gr.update(interactive=True),
                    f"{safe_name}_{engine_display.lower().replace(' ', '_').replace('-', '_')}",
                    metadata_out,
                    str(output_file),
                )

            def _stream_generate_impl():
                writer = None
                finalized = False
                try:
                    resolved_seed = int(seed) if seed is not None else -1
                    if resolved_seed < 0:
                        resolved_seed = random.randint(0, 2147483647)
                    set_seed(resolved_seed)
                    seed_msg = f"Seed: {resolved_seed}"

                    if configure_tts_manager_for_tenant:
                        configure_tts_manager_for_tenant(request=request, strict=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    safe_name = "".join(c if c.isalnum() else "_" for c in sample_name)
                    tenant_paths = get_tenant_paths(request=request, strict=True)
                    temp_dir = tenant_paths.temp_dir
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    final_output_file = temp_dir / f"voice_clone_{safe_name}_{timestamp}.wav"
                    writer = LiveAudioChunkWriter(final_output_file)

                    if engine == "qwen":
                        progress(0.1, desc=f"Loading Qwen3 model ({model_size})...")
                        model = tts_manager.get_qwen3_base(model_size)
                        prompt_items, was_cached = get_or_create_voice_prompt(
                            model=model,
                            sample_name=sample_name,
                            wav_path=sample["wav_path"],
                            ref_text=sample["ref_text"],
                            model_size=model_size,
                            progress_callback=progress,
                            request=request,
                        )
                        cache_status = "cached prompt" if was_cached else "new prompt"
                        stream_iter = tts_manager.stream_voice_clone_qwen(
                            text=text_to_generate,
                            language=language,
                            prompt_items=prompt_items,
                            seed=resolved_seed,
                            do_sample=qwen_do_sample,
                            temperature=qwen_temperature,
                            top_k=qwen_top_k,
                            top_p=qwen_top_p,
                            repetition_penalty=qwen_repetition_penalty,
                            max_new_tokens=qwen_max_new_tokens,
                            model_size=model_size,
                        )
                        engine_display = f"Qwen3-{model_size}"
                    else:
                        progress(0.1, desc=f"Loading VibeVoice model ({model_size})...")
                        cache_status = "native streaming"
                        stream_iter = tts_manager.stream_voice_clone_vibevoice(
                            text=text_to_generate,
                            voice_sample_path=sample["wav_path"],
                            seed=resolved_seed,
                            do_sample=vv_do_sample,
                            temperature=vv_temperature,
                            top_k=vv_top_k,
                            top_p=vv_top_p,
                            repetition_penalty=vv_repetition_penalty,
                            cfg_scale=vv_cfg_scale,
                            num_steps=vv_num_steps,
                            model_size=model_size,
                            user_config=shared_state.get("_user_config", {}),
                        )
                        engine_display = f"VibeVoice-{model_size}"

                    chunk_count = 0
                    for chunk_audio, chunk_sr in stream_iter:
                        chunk_arr = normalize_audio_chunk(chunk_audio)
                        chunk_path = writer.write_chunk(chunk_arr, chunk_sr)
                        if not chunk_path:
                            continue
                        chunk_count += 1
                        ui_chunk = np.clip(chunk_arr, -1.0, 1.0)
                        ui_chunk = (ui_chunk * 32767.0).astype(np.int16, copy=False)
                        yield (
                            (int(chunk_sr), ui_chunk),
                            f"Streaming {engine_display}... chunk {chunk_count}",
                            gr.update(interactive=False),
                            "",
                            "",
                            "",
                        )

                    if chunk_count <= 0:
                        raise RuntimeError(f"{engine_display} did not emit any streamed audio chunks.")

                    result = writer.finalize()
                    finalized = True
                    metadata = dedent(f"""\
                        Generated: {timestamp}
                        Sample: {sample_name}
                        Engine: {engine_display}
                        Language: {language}
                        Seed: {resolved_seed}
                        Stream Chunks: {chunk_count}
                        Text: {' '.join(text_to_generate.split())}
                        """)
                    metadata_out = '\n'.join(line.lstrip() for line in metadata.lstrip().splitlines())

                    progress(1.0, desc="Done!")
                    if play_completion_beep:
                        play_completion_beep()
                    yield (
                        gr.update(),
                        f"Generated using {engine_display}. {cache_status}\n{seed_msg}\nReady to save.",
                        gr.update(interactive=True),
                        f"{safe_name}_{engine_display.lower().replace(' ', '_').replace('-', '_')}",
                        metadata_out,
                        result.path,
                    )
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    yield None, f"❌ Error generating audio: {str(e)}", gr.update(interactive=False), "", "", ""
                finally:
                    if writer is not None and not finalized:
                        writer.cleanup_final_file()

            try:
                if should_stream:
                    if run_heavy_stream_job:
                        return run_heavy_stream_job("voice_clone_generate_stream", _stream_generate_impl, request=request)
                    return _stream_generate_impl()
                if run_heavy_job:
                    return run_heavy_job("voice_clone_generate", _generate_impl, request=request)
                return _generate_impl()
            except MemoryAdmissionError as exc:
                return None, f"⚠ Memory safety guard rejected request: {str(exc)}", gr.update(interactive=False), "", "", ""
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"❌ Error generating audio: {str(e)}", gr.update(interactive=False), "", "", ""

        def load_sample_from_lister(lister_value, request: gr.Request):
            """Load audio, text, and info for the selected sample from FileLister."""
            sample_name = get_selected_sample_name(lister_value)
            if not sample_name:
                return None, "", ""
            return load_sample_details(sample_name, request=request, strict=True)

        # Connect event handlers for Voice Clone tab
        components['sample_lister'].change(
            load_sample_from_lister,
            inputs=[components['sample_lister']],
            outputs=[components['sample_audio'], components['sample_text'], components['sample_info']]
        )

        if components.get('prompt_assistant'):
            wire_prompt_assistant_events(
                assistant=components['prompt_assistant'],
                target_components={"voice_clone.text": components['text_input']},
                status_component=components['clone_status'],
                shared_state=shared_state,
            )

        if prompt_apply_trigger is not None:
            wire_prompt_apply_listener(
                prompt_apply_trigger=prompt_apply_trigger,
                target_components={"voice_clone.text": components['text_input']},
                status_component=components['clone_status'],
            )

        wire_audio_route_dropdown_refresh(
            components['voice_clone_tab'],
            components['route_target'],
            shared_state,
        )
        wire_audio_route_source(
            send_button=components['route_btn'],
            target_dropdown=components['route_target'],
            audio_value_component=components['output_audio_path'],
            status_component=components['clone_status'],
            shared_state=shared_state,
            source_label="Voice Clone",
        )

        # Double-click = play sample audio
        components['sample_lister'].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#voice-clone-sample-audio .play-pause-button'); if (btn) btn.click(); }, 150); }"
        )

        # Auto-refresh samples when tab is selected
        def refresh_samples_for_tab(request: gr.Request):
            return get_sample_choices(request=request, strict=True)

        components['voice_clone_tab'].select(
            refresh_samples_for_tab,
            outputs=[components['sample_lister']]
        )

        # Wire up emotion preset handlers (same pattern as original)
        if 'update_from_emotion' in components.get('qwen_params', {}):
            components['qwen_emotion_preset'].change(
                components['qwen_params']['update_from_emotion'],
                inputs=[components['qwen_emotion_preset'], components['qwen_emotion_intensity']],
                outputs=[components['qwen_temperature'], components['qwen_top_p'], components['qwen_repetition_penalty']]
            )

            components['qwen_emotion_intensity'].change(
                components['qwen_params']['update_from_emotion'],
                inputs=[components['qwen_emotion_preset'], components['qwen_emotion_intensity']],
                outputs=[components['qwen_temperature'], components['qwen_top_p'], components['qwen_repetition_penalty']]
            )

        # Emotion save button
        components['qwen_save_emotion_btn'].click(
            fn=None,
            inputs=[components['qwen_emotion_preset']],
            outputs=None,
            js=show_input_modal_js(
                title="Save Emotion Preset",
                message="Enter a name for this emotion preset:",
                placeholder="e.g., Happy, Sad, Excited",
                context="qwen_emotion_"
            )
        )

        # Emotion delete button
        components['qwen_delete_emotion_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_confirmation_modal_js(
                title="Delete Emotion Preset?",
                message="This will permanently delete this emotion preset from your configuration.",
                confirm_button_text="Delete",
                context="qwen_emotion_"
            )
        )

        # Handler for emotion save from input modal
        def handle_qwen_emotion_input(input_value, intensity, temp, rep_pen, top_p):
            """Process input modal submission for Voice Clone emotion save."""
            if not input_value or not input_value.startswith("qwen_emotion_"):
                return gr.update(), gr.update()

            parts = input_value.split("_")
            if len(parts) >= 3:
                if parts[2] == "cancel":
                    return gr.update(), ""
                emotion_name = "_".join(parts[2:-1])

                # Use shared helper to process save result
                from modules.core_components.emotion_manager import process_save_emotion_result
                save_result = save_emotion_handler(emotion_name, intensity, temp, rep_pen, top_p)
                return process_save_emotion_result(save_result, shared_state)

            return gr.update(), gr.update()

        input_trigger.change(
            handle_qwen_emotion_input,
            inputs=[input_trigger, components['qwen_emotion_intensity'], components['qwen_temperature'],
                    components['qwen_repetition_penalty'], components['qwen_top_p']],
            outputs=[components['qwen_emotion_preset'], components['clone_status']]
        )

        # Handler for emotion delete from confirmation modal
        def delete_qwen_emotion_wrapper(confirm_value, emotion_name):
            """Only process if context matches qwen_emotion_."""
            if not confirm_value or not confirm_value.startswith("qwen_emotion_"):
                return gr.update(), gr.update()

            # Call the delete handler and discard the clear_trigger (3rd value)
            from modules.core_components.emotion_manager import process_delete_emotion_result
            delete_result = delete_emotion_handler(confirm_value, emotion_name)
            dropdown_update, status_msg, _clear = process_delete_emotion_result(delete_result, shared_state)
            return dropdown_update, status_msg

        confirm_trigger.change(
            delete_qwen_emotion_wrapper,
            inputs=[confirm_trigger, components['qwen_emotion_preset']],
            outputs=[components['qwen_emotion_preset'], components['clone_status']]
        )

        def generate_from_lister(lister_value, *args, request: gr.Request = None):
            """Extract sample name from lister and pass to generate."""
            yield (
                gr.update(value=None),
                "Generating...",
                gr.update(interactive=False),
                "",
                "",
                "",
            )
            result = generate_audio_handler(get_selected_sample_name(lister_value), *args, request=request)
            if hasattr(result, "__next__"):
                yield from result
                return
            yield result

        generate_event = components['generate_btn'].click(
            generate_from_lister,
            inputs=[components['sample_lister'], components['text_input'], components['language_dropdown'], components['seed_input'], components['clone_model_dropdown'],
                    components['qwen_do_sample'], components['qwen_temperature'], components['qwen_top_k'], components['qwen_top_p'], components['qwen_repetition_penalty'],
                    components['qwen_max_new_tokens'],
                    components['vv_do_sample'], components['vv_temperature'], components['vv_top_k'], components['vv_top_p'], components['vv_repetition_penalty'],
                    components['vv_cfg_scale'], components['vv_num_steps'], components['vv_sentences_per_chunk'],
                    components['luxtts_num_steps'], components['luxtts_t_shift'], components['luxtts_speed'], components['luxtts_return_smooth'],
                    components['luxtts_rms'], components['luxtts_ref_duration'], components['luxtts_guidance_scale'],
                    components['cb_exaggeration'], components['cb_cfg_weight'], components['cb_temperature'],
                    components['cb_repetition_penalty'], components['cb_top_p'], components['cb_min_p'],
                    components['cb_max_new_tokens'], components['cb_language_dropdown']],

            outputs=[
                components['output_audio'],
                components['clone_status'],
                components['save_btn'],
                components['suggested_name'],
                components['metadata_text'],
                components['output_audio_path'],
            ]
        )

        components['stop_btn'].click(
            lambda: ("Generation stopped.", gr.update(interactive=False)),
            inputs=[],
            outputs=[components['clone_status'], components['save_btn']],
            cancels=[generate_event],
            queue=False,
        )

        def apply_clone_output_pipeline(audio_path, enable_denoise, enable_remove_silences, enable_normalize, enable_mono, request: gr.Request):
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

        components['output_apply_pipeline_btn'].click(
            apply_clone_output_pipeline,
            inputs=[
                components['output_audio_path'],
                components['output_enable_denoise'],
                components['output_enable_remove_silences'],
                components['output_enable_normalize'],
                components['output_enable_mono'],
            ],
            outputs=[components['output_audio'], components['clone_status'], components['output_audio_path']],
        )

        save_clone_modal_js = show_input_modal_js(
            title="Save Voice Clone Output",
            message="Enter a filename for this generated audio:",
            placeholder="e.g., my_clone_line",
            context="save_vc_clone_"
        )

        def get_clone_existing_files(request: gr.Request):
            output_dir = get_tenant_output_dir(request=request, strict=True)
            return json.dumps(get_existing_wav_stems(output_dir))

        save_clone_js = f"""
        (existingFilesJson, suggestedName) => {{
            try {{
                window.inputModalExistingFiles = JSON.parse(existingFilesJson || '[]');
            }} catch(e) {{
                window.inputModalExistingFiles = [];
            }}
            const openModal = {save_clone_modal_js};
            openModal(suggestedName);
        }}
        """

        components['save_btn'].click(
            fn=get_clone_existing_files,
            inputs=[],
            outputs=[components['existing_files_json']],
        ).then(
            fn=None,
            inputs=[components['existing_files_json'], components['suggested_name']],
            js=save_clone_js,
        )

        def handle_clone_save_input(input_value, output_audio_path, metadata_text, request: gr.Request):
            matched, cancelled, chosen_name = parse_modal_submission(input_value, "save_vc_clone_")
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
            handle_clone_save_input,
            inputs=[input_trigger, components['output_audio_path'], components['metadata_text']],
            outputs=[components['clone_status'], components['save_btn']]
        )

        # Toggle language visibility based on model selection
        def toggle_language_visibility(model_selection):
            is_qwen = "Qwen" in model_selection
            is_cb_mtl = model_selection == "Chatterbox - Multilingual"
            return gr.update(visible=is_qwen), gr.update(visible=is_cb_mtl)

        components['clone_model_dropdown'].change(
            toggle_language_visibility,
            inputs=[components['clone_model_dropdown']],
            outputs=[components['language_row'], components['cb_language_row']]
        )

        # Toggle accordion visibility based on engine
        def toggle_engine_params(model_selection):
            is_qwen = "Qwen" in model_selection
            is_vv = "VibeVoice" in model_selection
            is_lux = "LuxTTS" in model_selection
            is_cb = "Chatterbox" in model_selection
            return (
                gr.update(visible=is_qwen),
                gr.update(visible=is_vv),
                gr.update(visible=is_lux),
                gr.update(visible=is_cb),
            )

        def toggle_cb_expert_row(model_selection, show_expert):
            return gr.update(visible=("Chatterbox" in model_selection and bool(show_expert)))

        components['clone_model_dropdown'].change(
            toggle_engine_params,
            inputs=[components['clone_model_dropdown']],
            outputs=[components['qwen_params_accordion'], components['vv_params_accordion'], components['luxtts_params_accordion'], components['cb_params_accordion']]
        )

        components['clone_model_dropdown'].change(
            toggle_cb_expert_row,
            inputs=[components['clone_model_dropdown'], components['vc_show_expert_params']],
            outputs=[components['cb_expert_row']],
        )

        components['vc_show_expert_params'].change(
            lambda enabled: save_preference("voice_clone_show_expert_params", bool(enabled)),
            inputs=[components['vc_show_expert_params']],
            outputs=[],
        )

        components['vc_show_expert_params'].change(
            toggle_cb_expert_row,
            inputs=[components['clone_model_dropdown'], components['vc_show_expert_params']],
            outputs=[components['cb_expert_row']],
        )

        components['cb_min_p'].change(
            lambda v: save_preference("chatterbox_min_p", float(v) if v is not None else 0.05),
            inputs=[components['cb_min_p']],
            outputs=[],
        )

        components['cb_max_new_tokens'].change(
            lambda v: save_preference("chatterbox_max_new_tokens", int(v) if v is not None else 2048),
            inputs=[components['cb_max_new_tokens']],
            outputs=[],
        )

        # Save voice clone model selection
        components['clone_model_dropdown'].change(
            lambda x: save_preference("voice_clone_model", x),
            inputs=[components['clone_model_dropdown']],
            outputs=[]
        )

        # Save language selection
        components['language_dropdown'].change(
            lambda x: save_preference("language", x),
            inputs=[components['language_dropdown']],
            outputs=[]
        )

        # Refresh emotion dropdowns and auto-load first sample when tab is selected
        def on_tab_select(lister_value, request: gr.Request):
            """When tab is selected, refresh emotions and auto-load sample if not loaded."""
            emotion_update = gr.update(choices=shared_state['get_emotion_choices'](shared_state['_active_emotions']))
            visible_models = []
            engine_settings = shared_state["_user_config"].get("enabled_engines", {})
            tts_engines = shared_state.get("TTS_ENGINES", {})
            for engine_key, engine_info in tts_engines.items():
                if engine_settings.get(engine_key, engine_info.get("default_enabled", True)):
                    visible_models.extend(engine_info.get("choices", []))
            if not visible_models:
                visible_models = shared_state.get("VOICE_CLONE_OPTIONS", [])
            preferred_model = shared_state["_user_config"].get(
                "voice_clone_model",
                (visible_models[0] if visible_models else "Qwen3 - Large"),
            )
            model_update = gr.update(value=(preferred_model if preferred_model in visible_models else (visible_models[0] if visible_models else preferred_model)))

            # Auto-load selected sample
            sample_name = get_selected_sample_name(lister_value)
            if sample_name:
                audio, text, info = load_sample_details(sample_name, request=request, strict=True)
                return emotion_update, audio, text, info, model_update

            return emotion_update, None, "", "", model_update

        components['voice_clone_tab'].select(
            on_tab_select,
            inputs=[components['sample_lister']],
            outputs=[components['qwen_emotion_preset'], components['sample_audio'],
                     components['sample_text'], components['sample_info'], components['clone_model_dropdown']]
        )


# Export for tab registry
get_tool_class = lambda: VoiceCloneTool


if __name__ == "__main__":
    """Standalone testing of Voice Clone tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(VoiceCloneTool, port=7862, title="Voice Clone - Standalone")
