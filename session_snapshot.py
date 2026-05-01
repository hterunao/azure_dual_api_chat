import json

from conversation_state import ConversationManager
from session_storage_bridge import session_storage_bridge


SESSION_SNAPSHOT_KEY = "iasa_chat_snapshot_v1"
SESSION_SNAPSHOT_TRIM_STEP_BYTES = 100 * 1024
DEFAULT_UI_STATE_KEYS = [
    "selected_model_name",
    "reasoning_effort",
    "tool_for_files",
    "detail_level",
    "streaming",
    "show_code_and_logs",
    "tool_choice",
]


def snapshot_log(event, **kwargs):
    important_events = {
        "restore.apply_done",
        "restore.principal_mismatch",
        "bridge.mode",
        "save.queue_start",
        "save.process_bridge_response",
        "save.process_wait_init",
        "save.failed",
    }
    if event not in important_events:
        return
    try:
        print(f"[snapshot] {event} {json.dumps(kwargs, ensure_ascii=False)}")
    except Exception:
        print(f"[snapshot] {event} {kwargs}")


def _safe_json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(v) for v in value]
    return str(value)


def _collect_ui_state(session_state, ui_state_keys):
    ui_state = {key: _safe_json_value(session_state.get(key)) for key in ui_state_keys if key in session_state}
    ui_state["switches"] = _safe_json_value(session_state.get("switches", {}))
    return ui_state


def serialize_snapshot(principal, conversation, session_state, ui_state_keys=DEFAULT_UI_STATE_KEYS):
    payload = {
        "version": 1,
        "principal": principal,
        "ui_state": _collect_ui_state(session_state, ui_state_keys),
        "conversation": conversation.to_snapshot_dict(),
    }
    return json.dumps(payload, ensure_ascii=False)


def _apply_loaded_snapshot(principal, payload_json, clients, assistants, session_state):
    if not payload_json:
        return

    try:
        payload = json.loads(payload_json)
    except Exception:
        return

    if payload.get("principal") != principal:
        snapshot_log("restore.principal_mismatch", saved_principal=payload.get("principal"), current_principal=principal)
        session_state["_snapshot_clear_pending"] = True
        session_state["need_rerun"] = True
        return

    for key, value in payload.get("ui_state", {}).items():
        session_state[key] = value

    conversation = ConversationManager.from_snapshot_dict(payload.get("conversation", {}), clients, assistants)
    if not conversation:
        return

    session_state["conversation"] = conversation
    session_state["processing"] = False
    snapshot_log(
        "restore.apply_done",
        restored_messages=len(conversation.thread.messages),
        response_id=conversation.response_id,
        response_last_message_id=conversation.response_last_message_id,
    )


def queue_snapshot_save(session_state, principal, conversation, ui_state_keys=DEFAULT_UI_STATE_KEYS):
    snapshot_log(
        "save.queue_start",
        principal=principal,
        message_count=len(conversation.thread.messages),
        response_id=conversation.response_id,
        response_last_message_id=conversation.response_last_message_id,
    )
    payload_json = serialize_snapshot(principal, conversation, session_state, ui_state_keys)
    session_state["_snapshot_save_pending"] = {
        "principal": principal,
        "payload_json": payload_json,
    }
    session_state["need_rerun"] = True


def run_snapshot_bridge(session_state, principal, clients, assistants, on_save_failed=None):
    pending = session_state.get("_snapshot_save_pending")
    mode = "idle"
    payload_json = ""

    if session_state.get("_snapshot_clear_pending"):
        mode = "clear"
    elif not session_state.get("_snapshot_restore_completed"):
        mode = "load"
    elif isinstance(pending, dict):
        if pending.get("principal") != principal:
            session_state["_snapshot_save_pending"] = None
            mode = "idle"
        else:
            mode = "save"
            payload_json = pending.get("payload_json", "{}")

    if mode != "idle":
        snapshot_log("bridge.mode", mode=mode)

    status = session_storage_bridge(
        mode=mode,
        storage_key=SESSION_SNAPSHOT_KEY,
        payload_json=payload_json,
        trim_step_bytes=SESSION_SNAPSHOT_TRIM_STEP_BYTES,
        key="session_storage_bridge_single",
    )

    if mode == "idle":
        return

    if not isinstance(status, dict):
        return

    if status.get("status") == "init":
        if mode == "save":
            snapshot_log("save.process_wait_init")
        return

    if mode == "clear":
        session_state["_snapshot_clear_pending"] = False
        return

    if mode == "load":
        session_state["_snapshot_restore_completed"] = True
        if status.get("ok") and status.get("found"):
            _apply_loaded_snapshot(principal, status.get("payload_json"), clients, assistants, session_state)
        return

    if mode == "save":
        snapshot_log("save.process_bridge_response", status=status)
        session_state["_snapshot_save_pending"] = None
        if status.get("ok") is False:
            fail_status = status.get("status", "unknown")
            snapshot_log("save.failed", status=fail_status)
            if on_save_failed:
                on_save_failed(fail_status)
