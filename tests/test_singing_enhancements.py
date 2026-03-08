from modules.core_components.tools.singing_audio_pipeline import KeyEstimate
from modules.core_components.tools.singing_enhancements import detect_key_from_available_audio


def test_detect_key_prefers_first_confident_source_in_priority_order(monkeypatch):
    estimates_by_source = {
        "source_separation_original": KeyEstimate("G", "major", 0.12, "source_separation_original"),
        "source_separation_vocals": KeyEstimate("A", "minor", 0.20, "source_separation_vocals"),
        "source_separation_backing": KeyEstimate("C", "major", 0.30, "source_separation_backing"),
        "current_lead_vocal": KeyEstimate("D", "minor", 0.40, "current_lead_vocal"),
    }

    def fake_estimate(_audio_path, *, source_label, sample_rate=44100, max_duration_sec=90.0):
        return estimates_by_source[source_label]

    monkeypatch.setattr(
        "modules.core_components.tools.singing_enhancements.estimate_key_from_audio",
        fake_estimate,
    )

    result = detect_key_from_available_audio(
        source_audio="lead.wav",
        sep_source_audio="mix.wav",
        sep_vocals_audio="sep_vocals.wav",
        sep_backing_audio="sep_backing.wav",
    )

    assert result["mode"] == "Manual"
    assert result["tonic"] == "G"
    assert result["scale"] == "major"
    assert result["used_label"] == "Source Separation Original"


def test_detect_key_falls_through_priority_until_confident_source(monkeypatch):
    estimates_by_source = {
        "source_separation_original": KeyEstimate("F", "major", 0.01, "source_separation_original"),
        "source_separation_vocals": KeyEstimate("E", "minor", 0.04, "source_separation_vocals"),
        "source_separation_backing": KeyEstimate("D", "minor", 0.11, "source_separation_backing"),
    }

    def fake_estimate(_audio_path, *, source_label, sample_rate=44100, max_duration_sec=90.0):
        return estimates_by_source[source_label]

    monkeypatch.setattr(
        "modules.core_components.tools.singing_enhancements.estimate_key_from_audio",
        fake_estimate,
    )

    result = detect_key_from_available_audio(
        source_audio=None,
        sep_source_audio="mix.wav",
        sep_vocals_audio="sep_vocals.wav",
        sep_backing_audio="sep_backing.wav",
    )

    assert result["mode"] == "Manual"
    assert result["tonic"] == "D"
    assert result["scale"] == "minor"
    assert result["used_label"] == "Source Separation Backing"


def test_detect_key_uses_best_low_confidence_estimate_when_needed(monkeypatch):
    estimates_by_source = {
        "source_separation_original": KeyEstimate("F", "major", 0.01, "source_separation_original"),
        "current_lead_vocal": KeyEstimate("A", "minor", 0.05, "current_lead_vocal"),
    }

    def fake_estimate(_audio_path, *, source_label, sample_rate=44100, max_duration_sec=90.0):
        return estimates_by_source[source_label]

    monkeypatch.setattr(
        "modules.core_components.tools.singing_enhancements.estimate_key_from_audio",
        fake_estimate,
    )

    result = detect_key_from_available_audio(
        source_audio="lead.wav",
        sep_source_audio="mix.wav",
        sep_vocals_audio=None,
        sep_backing_audio=None,
    )

    assert result["mode"] == "Manual"
    assert result["tonic"] == "A"
    assert result["scale"] == "minor"
    assert result["used_label"] == "Current Lead Vocal"
    assert "Low-confidence detection" in result["status"]
