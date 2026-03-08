"""Reusable audio routing helpers for cross-tab output -> input workflows."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable

import gradio as gr


AUDIO_ROUTE_KIND = "audio_route"

AUDIO_ROUTE_TARGETS = OrderedDict(
    [
        (
            "library_manager.processing",
            {
                "label": "Library Manager > Processing Studio",
                "tool_name": "Library Manager",
                "tab_id": "tab_library_manager",
            },
        ),
        (
            "singing_enhancements.source",
            {
                "label": "Singing Enhancements > Lead Vocal",
                "tool_name": "Singing Enhancements",
                "tab_id": "tab_singing_enhancements",
            },
        ),
        (
            "singing_enhancements.backing",
            {
                "label": "Singing Enhancements > Backing Track",
                "tool_name": "Singing Enhancements",
                "tab_id": "tab_singing_enhancements",
            },
        ),
    ]
)


def audio_route_build_payload(target_id: str, audio_path: str, source_label: str = "external") -> str:
    """Serialize a cross-tab audio route payload."""
    return json.dumps(
        {
            "kind": AUDIO_ROUTE_KIND,
            "target_id": str(target_id or "").strip(),
            "audio_path": str(audio_path or "").strip(),
            "source_label": str(source_label or "external").strip(),
        }
    )


def audio_route_parse_payload(payload: str | dict | None) -> dict | None:
    """Parse a cross-tab audio route payload."""
    if not payload:
        return None

    data = payload
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except Exception:
            return None

    if not isinstance(data, dict):
        return None

    if data.get("kind") != AUDIO_ROUTE_KIND:
        return None

    target_id = str(data.get("target_id") or "").strip()
    audio_path = str(data.get("audio_path") or "").strip()
    source_label = str(data.get("source_label") or "external").strip()
    if not target_id:
        return None

    return {
        "target_id": target_id,
        "audio_path": audio_path,
        "source_label": source_label,
    }


def audio_route_get_target_label(target_id: str) -> str:
    """Get the human-readable label for a target id."""
    meta = AUDIO_ROUTE_TARGETS.get(target_id, {})
    return meta.get("label", target_id)


def audio_route_get_target_tab_id(target_id: str) -> str:
    """Get the top-level tab id for a target id."""
    meta = AUDIO_ROUTE_TARGETS.get(target_id, {})
    return meta.get("tab_id", "")


def audio_route_get_targets_for_tool_names(tool_names: Iterable[str]) -> list[tuple[str, str]]:
    """Resolve available route targets for a set of instantiated/enabled tool names."""
    available_tool_names = set(tool_names or [])
    return [
        (meta["label"], target_id)
        for target_id, meta in AUDIO_ROUTE_TARGETS.items()
        if meta["tool_name"] in available_tool_names
    ]


def audio_route_get_available_targets(tool_components: Dict[str, dict] | None) -> list[tuple[str, str]]:
    """Resolve available route targets from instantiated tool components."""
    return audio_route_get_targets_for_tool_names((tool_components or {}).keys())


def create_audio_route_controls(initial_choices=None):
    """Create a compact destination dropdown + send button row."""
    choices = list(initial_choices or [])
    choice_labels = [label for label, _value in choices]
    default_value = choice_labels[0] if choice_labels else None
    with gr.Row():
        target_dropdown = gr.Dropdown(
            label="Destination",
            choices=choice_labels,
            value=default_value,
            interactive=bool(choice_labels),
            allow_custom_value=False,
            scale=4,
        )
        send_button = gr.Button("Send", size="sm", scale=1)
    return {
        "target_dropdown": target_dropdown,
        "send_button": send_button,
    }


def get_audio_route_dropdown_update(current_value, choices: list[tuple[str, str]]):
    """Build a dropdown update from the latest available choices."""
    labels = [label for label, _value in (choices or [])]
    next_value = current_value if current_value in labels else (labels[0] if labels else None)
    return gr.update(choices=labels, value=next_value, interactive=bool(labels))


def wire_audio_route_dropdown_refresh(selectable_component, target_dropdown, shared_state):
    """Refresh destination choices whenever a source tab is selected."""
    audio_route_get_available_targets = shared_state.get("audio_route_get_available_targets")

    if selectable_component is None or target_dropdown is None or audio_route_get_available_targets is None:
        return

    def refresh_choices(current_value):
        choices = audio_route_get_available_targets(shared_state.get("tool_components", {}))
        return get_audio_route_dropdown_update(current_value, choices)

    selectable_component.select(
        refresh_choices,
        inputs=[target_dropdown],
        outputs=[target_dropdown],
    )


def wire_audio_route_source(
    *,
    send_button,
    target_dropdown,
    audio_value_component,
    status_component,
    shared_state,
    source_label: str,
):
    """Wire a source output to the shared audio-routing trigger."""
    audio_route_trigger = shared_state.get("audio_route_trigger")
    main_tabs_component = shared_state.get("main_tabs_component")
    audio_route_build = shared_state.get("audio_route_build_payload")
    audio_route_get_available = shared_state.get("audio_route_get_available_targets")
    audio_route_get_tab_id = shared_state.get("audio_route_get_target_tab_id")
    audio_route_get_label = shared_state.get("audio_route_get_target_label")

    if send_button is None or target_dropdown is None or audio_value_component is None or status_component is None:
        return

    if main_tabs_component is not None:
        def send_and_switch(target_choice, audio_value):
            available_targets = dict(audio_route_get_available(shared_state.get("tool_components", {})))
            if not audio_route_trigger or not audio_route_build or not audio_route_get_tab_id:
                return "", "Audio routing is unavailable.", gr.update()
            if not target_choice or target_choice not in available_targets:
                return "", "Select a valid destination first.", gr.update()
            target_id = available_targets[target_choice]
            audio_path = str(audio_value or "").strip()
            if not audio_path:
                return "", "No audio available to send.", gr.update()
            if not Path(audio_path).exists():
                return "", f"Audio file not found: {audio_path}", gr.update()

            payload = audio_route_build(target_id, audio_path, source_label=source_label)
            label = audio_route_get_label(target_id) if audio_route_get_label else target_id
            tab_id = audio_route_get_tab_id(target_id)
            tab_update = gr.update(selected=tab_id) if tab_id else gr.update()
            return payload, f"Sent to {label}.", tab_update

        send_button.click(
            send_and_switch,
            inputs=[target_dropdown, audio_value_component],
            outputs=[audio_route_trigger, status_component, main_tabs_component],
        )
    else:
        def send_only(target_choice, audio_value):
            available_targets = dict(audio_route_get_available(shared_state.get("tool_components", {})))
            if not audio_route_trigger or not audio_route_build:
                return "", "Audio routing is unavailable."
            if not target_choice or target_choice not in available_targets:
                return "", "Select a valid destination first."
            target_id = available_targets[target_choice]
            audio_path = str(audio_value or "").strip()
            if not audio_path:
                return "", "No audio available to send."
            if not Path(audio_path).exists():
                return "", f"Audio file not found: {audio_path}"

            payload = audio_route_build(target_id, audio_path, source_label=source_label)
            label = audio_route_get_label(target_id) if audio_route_get_label else target_id
            return payload, f"Sent to {label}."

        send_button.click(
            send_only,
            inputs=[target_dropdown, audio_value_component],
            outputs=[audio_route_trigger, status_component],
        )


def wire_audio_route_listener(audio_route_trigger, target_components: Dict[str, object], status_component):
    """Wire a shared audio-route listener for simple filepath targets."""
    if audio_route_trigger is None or not target_components or status_component is None:
        return

    target_ids = list(target_components.keys())
    target_outputs = [target_components[target_id] for target_id in target_ids]

    def on_audio_route(trigger_value, *current_values):
        _ = current_values
        payload = audio_route_parse_payload(trigger_value)
        if not payload:
            return (*[gr.update() for _ in target_ids], gr.update())

        target_id = payload.get("target_id", "")
        if target_id not in target_components:
            return (*[gr.update() for _ in target_ids], gr.update())

        audio_path = payload.get("audio_path", "")
        if not audio_path:
            return (*[gr.update() for _ in target_ids], "Incoming audio route was empty.")

        source_label = payload.get("source_label", "external")
        label = audio_route_get_target_label(target_id)
        updates = []
        for current_target_id in target_ids:
            if current_target_id == target_id:
                updates.append(gr.update(value=audio_path))
            else:
                updates.append(gr.update())

        return (*updates, f"Loaded audio from {source_label} into {label}.")

    audio_route_trigger.change(
        on_audio_route,
        inputs=[audio_route_trigger, *target_outputs],
        outputs=[*target_outputs, status_component],
    )
