"""
Output History Tab

Browse and manage previously generated audio files.

Standalone testing:
    python -m modules.core_components.tools.output_history
"""
# Setup path for standalone testing BEFORE imports
if __name__ == "__main__":
    import sys
    from pathlib import Path as PathLib
    project_root = PathLib(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

import gradio as gr
from pathlib import Path
from datetime import datetime
from modules.core_components.tool_base import Tool, ToolConfig
from modules.core_components.tools.generated_output_save import convert_audio_file_to_mp3
from modules.core_components.ui_components.audio_routing import (
    create_audio_route_controls,
    wire_audio_route_dropdown_refresh,
    wire_audio_route_source,
)
from gradio_filelister import FileLister


class OutputHistoryTool(Tool):
    """Output History tool implementation."""

    config = ToolConfig(
        name="Output History",
        module_name="tool_output_history",
        description="Browse and manage generated audio files",
        enabled=True,
        category="utility"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create Output History tool UI."""
        components = {}
        initial_audio_route_choices = shared_state.get("audio_route_get_initial_targets", lambda: [])()

        with gr.TabItem("Output History") as tab:
            components['tab'] = tab
            gr.Markdown("Browse and manage previously generated audio files")
            with gr.Row():
                with gr.Column(scale=1):
                    components['file_lister'] = FileLister(
                        value=[],
                        height=400,
                        show_footer=False,
                        interactive=True,
                    )
                    with gr.Row():
                        components['convert_output_btn'] = gr.Button("Convert to MP3", size="sm")
                        components['delete_output_btn'] = gr.Button("Delete Selected", size="sm", variant="stop")

                with gr.Column(scale=1):
                    components['history_audio'] = gr.Audio(
                        label="Playback",
                        type="filepath",
                        autoplay=False,
                        interactive=False,
                        elem_id="output-history-audio"
                    )

                    components['history_metadata'] = gr.Textbox(
                        label="Generation Info",
                        interactive=False,
                        max_lines=15
                    )
                    components['history_download'] = gr.File(
                        label="Download Selected",
                        interactive=False
                    )
                    route_controls = create_audio_route_controls(initial_audio_route_choices)
                    components['route_target'] = route_controls['target_dropdown']
                    components['route_btn'] = route_controls['send_button']
                    components['delete_status'] = gr.Textbox(
                        label="Status",
                        interactive=False,
                        max_lines=2
                    )

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Output History events."""

        # Get required items from shared_state
        get_tenant_output_dir = shared_state.get('get_tenant_output_dir')
        show_confirmation_modal_js = shared_state.get('show_confirmation_modal_js')
        confirm_trigger = shared_state.get('confirm_trigger')

        def get_output_files_for_lister(output_dir):
            """Get list of generated output files for the FileLister widget."""
            if not output_dir or not output_dir.exists():
                return []
            files = sorted(
                [
                    f for f in output_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in {".wav", ".mp3"}
                ],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            result = []
            for f in files:
                try:
                    mtime = datetime.fromtimestamp(f.stat().st_mtime)
                    date_str = mtime.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date_str = ""
                result.append({"name": f.name, "date": date_str})
            return result

        def refresh_outputs(request: gr.Request):
            """Refresh the output file list."""
            try:
                output_dir = get_tenant_output_dir(request=request, strict=True)
                return get_output_files_for_lister(output_dir)
            except Exception as e:
                return [{"name": f"(Tenant Error) {str(e)}", "date": ""}]

        def on_file_selection_change(lister_value, request: gr.Request):
            """Handle file selection changes from FileLister.

            Single file selected: load audio + metadata.
            Multiple or no files: clear audio + metadata.
            """
            if not lister_value:
                return None, "", None

            selected = lister_value.get("selected", [])
            try:
                output_dir = get_tenant_output_dir(request=request, strict=True)
            except Exception as e:
                return None, f"Tenant error: {str(e)}", None

            if len(selected) == 1:
                file_path = output_dir / selected[0]
                if file_path.exists():
                    metadata_file = file_path.with_suffix(".txt")
                    if metadata_file.exists():
                        try:
                            metadata = metadata_file.read_text(encoding="utf-8")
                            return str(file_path), metadata, str(file_path)
                        except Exception:
                            pass
                    return str(file_path), "No metadata available", str(file_path)
            # Multiple or no selection - clear
            return None, "", None

        def delete_selected_files(action, lister_value, request: gr.Request):
            """Delete selected output files and their metadata."""
            # Ignore empty calls or wrong context
            if not action or not action.strip() or not action.startswith("output_"):
                return gr.update(), gr.update()

            # Only process confirm (cancel is handled purely in JS now)
            if "confirm" not in action:
                return gr.update(), gr.update()

            if not lister_value or not lister_value.get("selected"):
                return "[ERROR] No file(s) selected", gr.update()

            selected = lister_value["selected"]
            try:
                output_dir = get_tenant_output_dir(request=request, strict=True)
            except Exception as e:
                return f"Tenant error: {str(e)}", gr.update()
            deleted_count = 0
            errors = []

            for filename in selected:
                try:
                    audio_path = output_dir / filename
                    txt_path = audio_path.with_suffix(".txt")
                    stem = audio_path.stem

                    if audio_path.exists():
                        audio_path.unlink()

                    # Only remove metadata when no audio variants remain for this stem.
                    sibling_wav = output_dir / f"{stem}.wav"
                    sibling_mp3 = output_dir / f"{stem}.mp3"
                    if txt_path.exists() and not sibling_wav.exists() and not sibling_mp3.exists():
                        txt_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    errors.append(f"{filename}: {str(e)}")

            updated_files = get_output_files_for_lister(output_dir)

            if errors:
                msg = f"Deleted {deleted_count} file(s), {len(errors)} error(s): {'; '.join(errors)}"
            else:
                msg = f"Deleted {deleted_count} file(s)"

            return msg, updated_files

        def convert_selected_file_to_mp3(lister_value, request: gr.Request):
            """Convert a selected saved output WAV file to MP3."""
            if not lister_value or not lister_value.get("selected"):
                return "[ERROR] No file selected", gr.update(), gr.update(), gr.update()

            selected = lister_value["selected"]
            if len(selected) != 1:
                return "[ERROR] Select exactly one file to convert", gr.update(), gr.update(), gr.update()

            filename = selected[0]
            try:
                output_dir = get_tenant_output_dir(request=request, strict=True)
            except Exception as e:
                return f"Tenant error: {str(e)}", gr.update(), gr.update(), gr.update()

            source_path = output_dir / filename
            if not source_path.exists():
                return f"[ERROR] File not found: {filename}", gr.update(), gr.update(), gr.update()

            if source_path.suffix.lower() != ".wav":
                return "[ERROR] Only WAV outputs can be converted to MP3", gr.update(), gr.update(), gr.update()

            try:
                mp3_path = convert_audio_file_to_mp3(source_path, source_path.with_suffix(".mp3"))
            except Exception as e:
                return f"[ERROR] {str(e)}", gr.update(), gr.update(), gr.update()

            status = f"Converted to MP3: {mp3_path.name} ({mp3_path})"
            updated_files = get_output_files_for_lister(output_dir)
            return status, updated_files, str(mp3_path), str(mp3_path)

        # Show modal on delete button click
        if show_confirmation_modal_js and confirm_trigger:
            components['delete_output_btn'].click(
                fn=None,
                js=show_confirmation_modal_js(
                    title="Delete Output File(s)?",
                    message="This will permanently delete the selected audio file(s) and their metadata. This action cannot be undone.",
                    confirm_button_text="Delete",
                    context="output_"
                )
            )

            # Process confirmation — only outputs status + file list.
            # Audio/metadata clear naturally via file_lister.change cascade.
            confirm_trigger.change(
                delete_selected_files,
                inputs=[confirm_trigger, components['file_lister']],
                outputs=[components['delete_status'], components['file_lister']]
            )

        # Auto-refresh when tab is selected
        components['tab'].select(
            refresh_outputs,
            outputs=[components['file_lister']]
        )
        wire_audio_route_dropdown_refresh(
            components['tab'],
            components['route_target'],
            shared_state,
        )
        wire_audio_route_source(
            send_button=components['route_btn'],
            target_dropdown=components['route_target'],
            audio_value_component=components['history_audio'],
            status_component=components['delete_status'],
            shared_state=shared_state,
            source_label="Output History",
        )

        components['convert_output_btn'].click(
            convert_selected_file_to_mp3,
            inputs=[components['file_lister']],
            outputs=[
                components['delete_status'],
                components['file_lister'],
                components['history_audio'],
                components['history_download'],
            ]
        )

        # Load audio on selection change (click = display only, no autoplay)
        components['file_lister'].change(
            on_file_selection_change,
            inputs=[components['file_lister']],
            outputs=[components['history_audio'], components['history_metadata'], components['history_download']]
        )

        # Double-click = play audio by clicking the WaveSurfer play button via JS.
        # gr.update(autoplay=True) is broken in Gradio 6.x, so we use JS instead.
        # The .change handler already loads the file on first click of the dblclick;
        # this handler just triggers playback after a short delay for the waveform to load.
        components['file_lister'].double_click(
            fn=None,
            js="() => { setTimeout(() => { const btn = document.querySelector('#output-history-audio .play-pause-button'); if (btn) btn.click(); }, 150); }" 
        )


# Export for registry
get_tool_class = lambda: OutputHistoryTool


# Standalone testing
if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(OutputHistoryTool, port=7868, title="Output History - Standalone")
