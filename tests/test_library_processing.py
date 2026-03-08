from modules.core_components.library_processing import (
    WordTimestampLike,
    build_speaker_separation_import_plan,
    build_speaker_separation_sample_metadata,
    build_speaker_separation_sample_name,
    clean_transcription_for_engine,
    language_to_code,
    parse_asr_model,
    split_into_segments,
)
from modules.core_components.tenant_storage import (
    BaseStoragePaths,
    TenantStorageService,
    ensure_tenant_dirs,
    get_tenant_paths,
)


def test_parse_asr_model_variants():
    assert parse_asr_model("Qwen3 ASR - Large") == ("Qwen3 ASR", "Large")
    assert parse_asr_model("VibeVoice ASR - Default") == ("VibeVoice ASR", None)
    assert parse_asr_model("Whisper - Medium") == ("Whisper", "Medium")


def test_language_to_code_handles_auto_and_known_values():
    assert language_to_code("Auto-detect") is None
    assert language_to_code("English") == "en"
    assert language_to_code("Spanish") == "es"
    assert language_to_code("Unknown") is None


def test_clean_transcription_for_vibevoice():
    raw = "[SPEAKER_1]: hello [noise] world"
    cleaned = clean_transcription_for_engine("VibeVoice ASR", raw)
    assert cleaned == "hello world"


def test_split_into_segments_creates_silence_cut_segments():
    words = [
        WordTimestampLike("Hello", 0.0, 0.4),
        WordTimestampLike("world.", 0.45, 0.9),
        WordTimestampLike("This", 3.0, 3.3),
        WordTimestampLike("is", 3.35, 3.6),
        WordTimestampLike("test.", 3.65, 4.0),
    ]
    segments = split_into_segments(
        full_text="Hello world. This is test.",
        word_timestamps=words,
        min_duration=0.2,
        max_duration=10.0,
        silence_trim=1.0,
        discard_under=0.1,
    )

    assert len(segments) >= 2
    for start, end, text in segments:
        assert end > start
        assert text.strip()


def test_speaker_separation_helpers_build_names_and_blank_metadata():
    name = build_speaker_separation_sample_name("Interview Mix", 2)
    metadata = build_speaker_separation_sample_metadata(
        name,
        source_identifier="sample/interview_mix.wav",
        expected_speakers=2,
        speaker_index=2,
    )

    assert name == "Interview_Mix_speaker_2"
    assert metadata["Name"] == name
    assert metadata["Type"] == "Sample"
    assert metadata["Text"] == ""
    assert metadata["SpeakerIndex"] == 2
    assert metadata["ExpectedSpeakers"] == 2


def test_speaker_separation_import_plan_is_deterministic_and_estimates_sizes():
    tracks = [
        type("Track", (), {"speaker_index": 1, "sample_rate": 16000, "audio_data": [0.0] * 1600})(),
        type("Track", (), {"speaker_index": 2, "sample_rate": 16000, "audio_data": [0.0] * 800})(),
    ]

    plans = build_speaker_separation_import_plan(
        "dialog_take",
        tracks,
        source_identifier="dataset/dialog_take.wav",
        expected_speakers=2,
    )

    assert [plan.sample_name for plan in plans] == [
        "dialog_take_speaker_1",
        "dialog_take_speaker_2",
    ]
    assert plans[0].estimated_size_bytes > plans[1].estimated_size_bytes > 0
    assert plans[0].metadata["Text"] == ""


def test_speaker_separation_import_plan_sizes_can_fail_quota_validation(tmp_path):
    base_paths = BaseStoragePaths(
        samples_dir=tmp_path / "samples",
        datasets_dir=tmp_path / "datasets",
        output_dir=tmp_path / "output",
        trained_models_dir=tmp_path / "trained",
        temp_dir=tmp_path / "temp",
    )
    paths = get_tenant_paths(base_paths, "tenant-a")
    ensure_tenant_dirs(paths)
    tenant_service = TenantStorageService(
        base_paths=base_paths,
        tenant_file_limit_mb=1,
        tenant_media_quota_gb=1,
    )
    large_track = type(
        "Track",
        (),
        {"speaker_index": 1, "sample_rate": 16000, "audio_data": [0.0] * 700_000},
    )()
    plans = build_speaker_separation_import_plan(
        "large_dialog",
        [large_track],
        source_identifier="sample/large_dialog.wav",
        expected_speakers=2,
    )

    ok, message = tenant_service.validate_generated_sizes(
        paths,
        [plan.estimated_size_bytes for plan in plans],
        label="Speaker separation import",
    )

    assert not ok
    assert "exceeds per-file limit" in message
