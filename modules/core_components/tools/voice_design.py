"""
Voice Design Tab

Create new voices from natural language descriptions using Qwen3-TTS VoiceDesign.
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
import gradio as gr
import soundfile as sf
import json
import torch
import random
from datetime import datetime
from pathlib import Path
from modules.core_components.ai_models.model_utils import set_seed

from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.ai_models.tts_manager import get_tts_manager
from modules.core_components.tools.generated_output_save import (
    get_existing_wav_stems,
    parse_modal_submission,
    persist_audio_to_wav,
    sanitize_output_name,
)
from modules.core_components.tools.output_audio_pipeline import (
    OutputAudioPipelineConfig,
    apply_generation_output_pipeline,
)
from modules.core_components.tools.live_stream_policy import prefix_non_stream_status
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


class VoiceDesignTool(Tool):
    """Voice Design tool implementation."""

    config = ToolConfig(
        name="Voice Design",
        module_name="tool_voice_design",
        description="Create voices from natural language descriptions",
        enabled=True,
        category="generation"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Voice Design tool UI."""
        components = {}

        # Get shared utilities
        LANGUAGES = shared_state.get('LANGUAGES', ['Auto'])
        _user_config = shared_state.get('_user_config', {})
        create_qwen_advanced_params = shared_state.get('create_qwen_advanced_params')
        deepfilter_available = bool(shared_state.get("DEEPFILTER_AVAILABLE", False))
        initial_audio_route_choices = shared_state.get("audio_route_get_initial_targets", lambda: [])()

        with gr.TabItem("Voice Design", id="tab_voice_design") as voice_design_tab:
            components["voice_design_tab"] = voice_design_tab
            gr.Markdown("Create new voices from natural language descriptions")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Create Design")

                    components['design_text_input'] = gr.Textbox(
                        label="Reference Text",
                        placeholder="Enter the text for the voice design (this will be spoken in the designed voice)...",
                        lines=3,
                        value="Thank you for listening to this voice design sample. This sentence is intentionally a bit long so you can hear the full range and quality of the generated voice."
                    )

                    components['design_instruct_input'] = gr.Textbox(
                        label="Voice Design Instructions",
                        placeholder="Describe the voice: e.g., 'Young female voice, bright and cheerful, slightly breathy' or 'Deep male voice with a warm, comforting tone, speak slowly'",
                        lines=3
                    )
                    components['prompt_assistant'] = create_prompt_assistant(
                        shared_state=shared_state,
                        target_ids=["voice_design.reference", "voice_design.instructions"],
                        default_target_id="voice_design.reference",
                    )

                    with gr.Row():
                        components['design_language'] = gr.Dropdown(
                            choices=LANGUAGES,
                            value=_user_config.get("language", "Auto"),
                            label="Language",
                            scale=2
                        )
                        components['design_seed'] = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0,
                            scale=1
                        )

                    # Qwen Advanced Parameters
                    design_params = create_qwen_advanced_params(
                        include_emotion=False,
                        visible=True,
                        shared_state=shared_state
                    )
                    components.update(design_params)

                    with gr.Row():
                        components['design_generate_btn'] = gr.Button("Generate Voice", variant="primary", size="lg")
                        components['design_stop_btn'] = gr.Button("Stop", variant="stop", size="lg")
                    components['design_status'] = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=3)

                with gr.Column(scale=1):
                    gr.Markdown("### Preview & Save")
                    components['design_output_audio'] = gr.Audio(
                        label="Generated Audio",
                        type="filepath",
                        interactive=True,
                    )
                    with gr.Row():
                        components['design_output_enable_denoise'] = gr.Checkbox(
                            label="Enable Denoise",
                            value=False,
                            visible=deepfilter_available,
                        )
                        components['design_output_enable_remove_silences'] = gr.Checkbox(
                            label="Remove Silences",
                            value=False,
                        )
                        components['design_output_enable_normalize'] = gr.Checkbox(
                            label="Enable Normalize",
                            value=False,
                        )
                        components['design_output_enable_mono'] = gr.Checkbox(
                            label="Enable Mono",
                            value=False,
                        )
                        components['design_output_apply_pipeline_btn'] = gr.Button(
                            "Apply Pipeline",
                            variant="secondary",
                            size="sm",
                        )

                    components['design_save_btn'] = gr.Button("Save Sample", variant="primary", interactive=False)
                    route_controls = create_audio_route_controls(initial_audio_route_choices)
                    components['design_route_target'] = route_controls['target_dropdown']
                    components['design_route_btn'] = route_controls['send_button']
                    components['design_existing_files_json'] = gr.State(value="[]")

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Voice Design events."""

        # Get shared utilities and directories
        show_input_modal_js = shared_state.get('show_input_modal_js')
        save_preference = shared_state.get('save_preference')
        input_trigger = shared_state.get('input_trigger')
        prompt_apply_trigger = shared_state.get('prompt_apply_trigger')
        get_tenant_samples_dir = shared_state.get('get_tenant_samples_dir')
        get_tenant_paths = shared_state.get('get_tenant_paths')
        configure_tts_manager_for_tenant = shared_state.get('configure_tts_manager_for_tenant')
        play_completion_beep = shared_state.get('play_completion_beep')
        normalize_audio = shared_state['normalize_audio']
        remove_silences = shared_state['remove_silences']
        convert_to_mono = shared_state['convert_to_mono']
        clean_audio = shared_state['clean_audio']
        deepfilter_available = bool(shared_state.get("DEEPFILTER_AVAILABLE", False))

        # Get TTS manager (singleton)
        tts_manager = get_tts_manager()

        def generate_voice_design_handler(text_to_generate, language, instruct, seed,
                                          do_sample, temperature, top_k, top_p, repetition_penalty, max_new_tokens,
                                          request: gr.Request = None,
                                          progress=gr.Progress()):
            """Generate audio using voice design with natural language instructions."""
            if not text_to_generate or not text_to_generate.strip():
                return None, "❌ Please enter text to generate.", gr.update()

            if not instruct or not instruct.strip():
                return None, "❌ Please enter voice design instructions.", gr.update()

            try:
                # Set the seed for reproducibility
                seed = int(seed) if seed is not None else -1
                if seed < 0:
                    seed = random.randint(0, 2147483647)

                set_seed(seed)
                seed_msg = f"Seed: {seed}"
                if configure_tts_manager_for_tenant:
                    configure_tts_manager_for_tenant(request=request, strict=True)

                progress(0.1, desc="Loading VoiceDesign model...")

                # Call manager to generate
                audio_data, sr = tts_manager.generate_voice_design(
                    text=text_to_generate.strip(),
                    language=language if language != "Auto" else "Auto",
                    instruct=instruct.strip(),
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens
                )

                progress(0.8, desc="Saving preview...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                # Ensure audio is numpy array (move to CPU if tensor)
                if hasattr(audio_data, "cpu"):
                    audio_data = audio_data.cpu().numpy()
                elif hasattr(audio_data, "numpy"):
                    audio_data = audio_data.numpy()

                # Temporary previews must stay out of output history.
                tenant_paths = get_tenant_paths(request=request, strict=True)
                TEMP_DIR = tenant_paths.temp_dir
                out_file = TEMP_DIR / f"temp_voice_design_{timestamp}.wav"
                TEMP_DIR.mkdir(parents=True, exist_ok=True)

                sf.write(str(out_file), audio_data, sr)

                progress(1.0, desc="Done!")
                if play_completion_beep:
                    play_completion_beep()
                return (
                    str(out_file),
                    prefix_non_stream_status(f"Voice design generated. Ready to save.\n{seed_msg}"),
                    gr.update(interactive=True),
                )

            except Exception as e:
                return None, f"❌ Error generating audio: {str(e)}", gr.update(interactive=False)

        def save_designed_voice(audio, design_name, ref_text, instruct, seed, request: gr.Request = None):
            """Save a designed voice as a sample (tool-specific implementation)."""
            if not audio:
                return "❌ No audio to save"

            if not design_name or not design_name.strip():
                return "❌ Please enter a name"

            try:
                # Clean the name
                safe_name = sanitize_output_name(design_name)
                if not safe_name:
                    return "❌ Invalid name"

                # Copy audio file to samples folder
                SAMPLES_DIR = get_tenant_samples_dir(request=request, strict=True)
                SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
                wav_path = SAMPLES_DIR / f"{safe_name}.wav"
                json_path = SAMPLES_DIR / f"{safe_name}.json"

                persist_audio_to_wav(audio, wav_path, default_sample_rate=24000)

                # Create metadata
                metadata = {
                    "Name": safe_name,
                    "Text": ref_text if ref_text else "Voice design sample",
                    "Type": "Voice Design",
                    "Instruct": instruct if instruct else "",
                    "Seed": seed if seed is not None else -1,
                    "Created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

                return f"✓ Saved as sample: {safe_name}"

            except Exception as e:
                return f"❌ Error saving: {str(e)}"

        # Wire generate button
        design_generate_event = components['design_generate_btn'].click(
            generate_voice_design_handler,
            inputs=[components['design_text_input'], components['design_language'], components['design_instruct_input'],
                    components['design_seed'],
                    components['do_sample'], components['temperature'], components['top_k'],
                    components['top_p'], components['repetition_penalty'], components['max_new_tokens']],
            outputs=[components['design_output_audio'], components['design_status'], components['design_save_btn']]
        )

        components['design_stop_btn'].click(
            lambda: ("Generation stopped.", gr.update(interactive=False)),
            inputs=[],
            outputs=[components['design_status'], components['design_save_btn']],
            cancels=[design_generate_event],
            queue=False,
        )

        def apply_design_output_pipeline(audio_value, enable_denoise, enable_remove_silences, enable_normalize, enable_mono, request: gr.Request):
            pipeline = OutputAudioPipelineConfig(
                enable_denoise=bool(enable_denoise),
                enable_remove_silences=bool(enable_remove_silences),
                enable_normalize=bool(enable_normalize),
                enable_mono=bool(enable_mono),
            )
            updated_audio, status = apply_generation_output_pipeline(
                audio_value,
                pipeline,
                deepfilter_available=deepfilter_available,
                denoise_step=lambda path: clean_audio(path),
                remove_silences_step=lambda path: remove_silences(path, request=request),
                normalize_step=lambda path: normalize_audio(path, request=request),
                mono_step=lambda path: convert_to_mono(path, request=request),
            )
            if not updated_audio:
                return gr.update(), status
            return gr.update(value=updated_audio), status

        components['design_output_apply_pipeline_btn'].click(
            apply_design_output_pipeline,
            inputs=[
                components['design_output_audio'],
                components['design_output_enable_denoise'],
                components['design_output_enable_remove_silences'],
                components['design_output_enable_normalize'],
                components['design_output_enable_mono'],
            ],
            outputs=[components['design_output_audio'], components['design_status']],
        )

        if components.get('prompt_assistant'):
            wire_prompt_assistant_events(
                assistant=components['prompt_assistant'],
                target_components={
                    "voice_design.reference": components['design_text_input'],
                    "voice_design.instructions": components['design_instruct_input'],
                },
                status_component=components['design_status'],
                shared_state=shared_state,
            )

        if prompt_apply_trigger is not None:
            wire_prompt_apply_listener(
                prompt_apply_trigger=prompt_apply_trigger,
                target_components={
                    "voice_design.reference": components['design_text_input'],
                    "voice_design.instructions": components['design_instruct_input'],
                },
                status_component=components['design_status'],
            )

        wire_audio_route_dropdown_refresh(
            components['voice_design_tab'],
            components['design_route_target'],
            shared_state,
        )
        wire_audio_route_source(
            send_button=components['design_route_btn'],
            target_dropdown=components['design_route_target'],
            audio_value_component=components['design_output_audio'],
            status_component=components['design_status'],
            shared_state=shared_state,
            source_label="Voice Design",
        )

        # Save designed voice - show modal
        design_modal_js = show_input_modal_js(
            title="Save Designed Voice",
            message="Enter a name for this voice design:",
            placeholder="e.g., Bright-Female, Deep-Male, Cheerful-Voice",
            context="save_design_"
        )

        def get_existing_sample_names(request: gr.Request):
            import json as json_mod

            sample_dir = get_tenant_samples_dir(request=request, strict=True)
            return json_mod.dumps(get_existing_wav_stems(sample_dir))

        design_save_js = f"""
        (existingFilesJson) => {{
            try {{
                window.inputModalExistingFiles = JSON.parse(existingFilesJson || '[]');
            }} catch(e) {{
                window.inputModalExistingFiles = [];
            }}
            const openModal = {design_modal_js};
            openModal('');
        }}
        """

        components['design_save_btn'].click(
            fn=get_existing_sample_names,
            inputs=[],
            outputs=[components['design_existing_files_json']],
        ).then(
            fn=None,
            inputs=[components['design_existing_files_json']],
            js=design_save_js
        )

        # Handle save designed voice input modal submission
        if input_trigger:
            def handle_save_design_input(input_value, audio, ref_text, instruct, seed, request: gr.Request = None):
                """Process input modal submission for saving designed voice."""
                # Context filtering: only process if this is our context
                matched, cancelled, design_name = parse_modal_submission(input_value, "save_design_")
                if not matched:
                    return gr.update(), gr.update()
                if cancelled:
                    return gr.update(), gr.update()
                if not design_name or not design_name.strip():
                    return "❌ Please enter a name.", gr.update()

                status = save_designed_voice(audio, design_name, ref_text, instruct, seed, request=request)
                if status.startswith("✓"):
                    return status, gr.update(interactive=False)
                return status, gr.update()

            input_trigger.change(
                handle_save_design_input,
                inputs=[input_trigger, components['design_output_audio'], 
                        components['design_text_input'], components['design_instruct_input'],
                        components['design_seed']],
                outputs=[components['design_status'], components['design_save_btn']]
            )

        # Save language preference
        if save_preference:
            components['design_language'].change(
                lambda x: save_preference("language", x),
                inputs=[components['design_language']],
                outputs=[]
            )


# Export for registry
get_tool_class = lambda: VoiceDesignTool


if __name__ == "__main__":
    """Standalone testing of Voice Design tool."""
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(VoiceDesignTool, port=7861, title="Voice Design - Standalone")
