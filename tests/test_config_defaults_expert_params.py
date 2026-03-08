from __future__ import annotations

import json

import modules.core_components.tools as tools_mod


def test_load_config_includes_expert_defaults(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(tools_mod, "CONFIG_FILE", cfg_path)

    cfg = tools_mod.load_config()

    assert cfg["voice_clone_show_expert_params"] is False
    assert cfg["conversation_show_expert_params"] is False
    assert cfg["voice_changer_show_expert_params"] is False
    assert cfg["voice_changer_profile"] == "speech"
    assert cfg["chatterbox_min_p"] == 0.05
    assert cfg["chatterbox_max_new_tokens"] == 2048
    assert cfg["voice_changer_vc_n_cfm_timesteps"] == 0


def test_load_config_merges_existing_and_keeps_new_defaults(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "voice_clone_model": "Qwen3 - Small",
                "chatterbox_min_p": 0.12,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(tools_mod, "CONFIG_FILE", cfg_path)

    cfg = tools_mod.load_config()

    assert cfg["voice_clone_model"] == "Qwen3 - Small"
    assert cfg["chatterbox_min_p"] == 0.12
    assert cfg["voice_changer_profile"] == "speech"
    assert cfg["chatterbox_max_new_tokens"] == 2048
    assert cfg["voice_changer_vc_n_cfm_timesteps"] == 0
