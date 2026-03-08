from modules.core_components.library_processing import (
    build_routed_processing_context,
    resolve_processing_save_destination,
)
from modules.core_components.ui_components.audio_routing import (
    audio_route_build_payload,
    audio_route_get_available_targets,
    audio_route_get_target_tab_id,
    audio_route_parse_payload,
)


def test_audio_route_payload_round_trip():
    payload = audio_route_build_payload(
        "singing_enhancements.source",
        "/tmp/example.wav",
        source_label="Voice Clone",
    )

    parsed = audio_route_parse_payload(payload)

    assert parsed == {
        "target_id": "singing_enhancements.source",
        "audio_path": "/tmp/example.wav",
        "source_label": "Voice Clone",
    }


def test_audio_route_available_targets_only_include_instantiated_tools():
    tool_components = {
        "Voice Clone": {},
        "Library Manager": {},
        "Singing Enhancements": {},
        "Voice Changer": {},
    }

    targets = audio_route_get_available_targets(tool_components)

    assert targets == [
        ("Library Manager > Processing Studio", "library_manager.processing"),
        ("Singing Enhancements > Lead Vocal", "singing_enhancements.source"),
        ("Singing Enhancements > Backing Track", "singing_enhancements.backing"),
        ("Voice Changer > Source Audio", "voice_changer.source"),
    ]


def test_audio_route_target_tab_mapping():
    assert audio_route_get_target_tab_id("library_manager.processing") == "tab_library_manager"
    assert audio_route_get_target_tab_id("singing_enhancements.source") == "tab_singing_enhancements"
    assert audio_route_get_target_tab_id("voice_changer.source") == "tab_voice_changer"
    assert audio_route_get_target_tab_id("missing.target") == ""


def test_build_routed_processing_context_marks_external_source():
    context = build_routed_processing_context("/tmp/final_mix.wav")

    assert context.source_type == "External"
    assert context.source_identifier == "final_mix.wav"
    assert context.original_audio_path == "/tmp/final_mix.wav"
    assert context.source_dataset_folder == ""


def test_external_processing_save_destination_defaults_to_samples_then_dataset():
    assert resolve_processing_save_destination("primary", "external") == "sample"
    assert resolve_processing_save_destination("secondary", "external") == "dataset"
