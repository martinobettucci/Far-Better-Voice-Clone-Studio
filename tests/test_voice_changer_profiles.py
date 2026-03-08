from modules.core_components.tools.voice_changer import (
    build_voice_changer_metadata,
    resolve_voice_changer_conversion_settings,
)


def test_speech_profile_keeps_existing_behavior():
    settings = resolve_voice_changer_conversion_settings("Speech", 0)

    assert settings.profile == "speech"
    assert settings.effective_steps is None
    assert settings.chunk_seconds is None
    assert settings.overlap_seconds is None
    assert settings.used_profile_defaults is False


def test_singing_profile_applies_defaults_when_expert_steps_are_auto():
    settings = resolve_voice_changer_conversion_settings("Singing", 0)

    assert settings.profile == "singing"
    assert settings.effective_steps == 25
    assert settings.chunk_seconds == 6.0
    assert settings.overlap_seconds == 1.5
    assert settings.used_profile_defaults is True


def test_singing_profile_respects_explicit_expert_steps():
    settings = resolve_voice_changer_conversion_settings("Singing", 40)

    assert settings.profile == "singing"
    assert settings.effective_steps == 40
    assert settings.chunk_seconds == 6.0
    assert settings.overlap_seconds == 1.5
    assert settings.used_profile_defaults is False


def test_singing_profile_clamps_explicit_expert_steps_to_max():
    settings = resolve_voice_changer_conversion_settings("Singing", 999)

    assert settings.profile == "singing"
    assert settings.effective_steps == 100
    assert settings.chunk_seconds == 6.0
    assert settings.overlap_seconds == 1.5
    assert settings.used_profile_defaults is False


def test_voice_changer_metadata_mentions_profile_and_effective_parameters():
    settings = resolve_voice_changer_conversion_settings("Singing", 0)

    metadata = build_voice_changer_metadata(
        timestamp="20260308_120000",
        target_name="Lead Voice",
        settings=settings,
        effective_chunk_seconds=6.0,
        effective_overlap_seconds=1.5,
    )

    assert "Conversion Profile: Singing" in metadata
    assert "VC Diffusion Steps: 25" in metadata
    assert "Chunking: 6.00s chunks / 1.50s overlap" in metadata
    assert "Profile Note: Singing profile applied default diffusion and chunking tuned for sung input." in metadata
