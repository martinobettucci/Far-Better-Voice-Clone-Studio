import modules.core_components as core_components
import modules.core_components.ui_components as ui_components
import gradio as gr
from modules.core_components import help_page
from modules.core_components.tools import get_tool_registry
from modules.core_components.tools.singing_enhancements import SingingEnhancementsTool
from modules.core_components.tools.settings import SettingsTool, TOGGLEABLE_TOOLS


def test_singing_enhancements_is_exposed_in_registry_and_settings():
    registry = get_tool_registry()

    assert "singing_enhancements" in registry
    assert registry["singing_enhancements"].name == "Singing Enhancements"
    assert ("Singing Enhancements", "Singing Enhancements") in TOGGLEABLE_TOOLS


def test_help_guide_lists_singing_enhancements():
    assert "Singing Enhancements" in help_page.HELP_TOPICS
    text = help_page.get_help_text("Singing Enhancements")
    assert "segment selection" in text.lower()


def test_singing_waveform_assets_are_exported():
    assert "Waveform Selection" in ui_components.SINGING_WAVEFORM_HTML
    assert "singing-waveform-canvas" in ui_components.SINGING_WAVEFORM_CSS
    assert "refreshSingingWaveformStudio" in ui_components.SINGING_WAVEFORM_HEAD
    assert core_components.SINGING_WAVEFORM_HTML == ui_components.SINGING_WAVEFORM_HTML


def test_singing_enhancements_tool_builds_with_exported_waveform_widget():
    with gr.Blocks():
        components = SingingEnhancementsTool.create_tool(
            {
                "_user_config": {},
                "OUTPUT_DIR": "/tmp/output",
                "audio_route_get_initial_targets": lambda: [],
            }
        )

    assert "waveform_selector" in components
    assert isinstance(components["waveform_selector"], gr.HTML)
    assert "singing_sections" in components
    assert "sep_goal_select" in components
    assert "sep_model_select" in components
    assert "sep_source_audio" in components
    assert "sep_vocals_audio" in components


def test_settings_tool_builds_single_model_downloader_with_source_separation_entries():
    with gr.Blocks():
        components = SettingsTool.create_tool({"_user_config": {}})

    assert "download_all_models" in components
    assert "source_sep_model_select" not in components
    assert "Source Separation - Recommended Defaults" in components["ALL_MODEL_CHOICES"]
