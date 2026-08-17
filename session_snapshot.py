import json
from uuid import uuid4

from conversation_state import ConversationManager
from session_storage_bridge import session_storage_bridge


SESSION_SNAPSHOT_KEY = "iasa_chat_snapshot_v1"
SESSION_OUTBOX_KEY = "iasa_chat_outbox_v1"
SESSION_SNAPSHOT_TRIM_STEP_BYTES = 100 * 1024
SNAPSHOT_INIT_MAX_RETRY = 4
SNAPSHOT_EMPTY_MAX_RETRY = 2
SNAPSHOT_MISMATCH_MAX_RETRY = 3
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
        "bridge.status",
        "bridge.init_retry",
        "bridge.init_timeout",
        "save.queue_start",
        "save.process_bridge_response",
        "save.failed",
        "load.empty_retry",
        "load.empty_confirmed",
        "load.completed",
        "load.wait_principal_recovery",
        "load.schedule_clear",
        "outbox.queue_save",
        "outbox.queue_clear",
        "outbox.loaded",
        "outbox.load_invalid",
        "outbox.load_principal_mismatch",
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
    conversation_payload = conversation.to_snapshot_dict()
    write_seq = int(session_state.get("_snapshot_write_seq", 0))
    payload = {
        "version": 1,
        "principal": principal,
        "ui_state": _collect_ui_state(session_state, ui_state_keys),
        "conversation": conversation_payload,
        "_meta": {
            "write_seq": write_seq,
            "message_count": len(conversation_payload.get("messages", [])),
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _apply_loaded_snapshot(principal, payload_json, clients, assistants, session_state):
    if not payload_json:
        return False

    try:
        payload = json.loads(payload_json)
    except Exception:
        return False

    saved_principal = payload.get("principal")

    def _is_placeholder(p):
        return not p or str(p).startswith("no_header[") or str(p).startswith("no_client_ip[")

    def _normalize_principal(p):
        if _is_placeholder(p):
            return "placeholder_principal"
        return str(p)

    if _normalize_principal(saved_principal) != _normalize_principal(principal):
        snapshot_log("restore.principal_mismatch", saved_principal=saved_principal, current_principal=principal)

        if _is_placeholder(principal) or _is_placeholder(saved_principal):
            return "mismatch"

        return "clear"

    for key, value in payload.get("ui_state", {}).items():
        session_state[key] = value

    # 保存シーケンス番号を引き継ぐ（再起動直後の stale_write_blocked を防ぐ）
    loaded_write_seq = int(((payload.get("_meta") or {}).get("write_seq")) or 0)
    session_state["_snapshot_write_seq"] = max(int(session_state.get("_snapshot_write_seq", 0)), loaded_write_seq)

    conversation = ConversationManager.from_snapshot_dict(payload.get("conversation", {}), clients, assistants)
    if not conversation:
        return False

    session_state["conversation"] = conversation
    session_state["processing"] = False
    snapshot_log(
        "restore.apply_done",
        restored_messages=len(conversation.thread.messages),
        response_id=conversation.response_id,
        response_last_message_id=conversation.response_last_message_id,
    )
    return True


def queue_snapshot_save(session_state, principal, conversation, ui_state_keys=DEFAULT_UI_STATE_KEYS):
    snapshot_log(
        "save.queue_start",
        principal=principal,
        message_count=len(conversation.thread.messages),
        response_id=conversation.response_id,
        response_last_message_id=conversation.response_last_message_id,
    )
    session_state["_snapshot_write_seq"] = int(session_state.get("_snapshot_write_seq", 0)) + 1
    payload_json = serialize_snapshot(principal, conversation, session_state, ui_state_keys)
    session_state["_snapshot_save_pending"] = {
        "principal": principal,
        "payload_json": payload_json,
    }
    session_state["need_rerun"] = True


def _serialize_outbox(principal, pending_user_message):
    payload = {
        "version": 1,
        "principal": principal,
        "pending_user_message": _safe_json_value(pending_user_message),
    }
    return json.dumps(payload, ensure_ascii=False)


def queue_outbox_save(session_state, principal, pending_user_message):
    snapshot_log("outbox.queue_save", principal=principal)
    session_state["_outbox_save_pending"] = {
        "principal": principal,
        "payload_json": _serialize_outbox(principal, pending_user_message),
    }
    session_state["need_rerun"] = True


def queue_outbox_clear(session_state):
    snapshot_log("outbox.queue_clear")
    session_state["_outbox_clear_pending"] = True
    session_state["need_rerun"] = True


def pop_loaded_outbox_message(session_state):
    msg = session_state.get("_outbox_loaded_message")
    session_state["_outbox_loaded_message"] = None
    return msg


def _ensure_bridge_state(session_state):
    if "_snapshot_init_retry_count" not in session_state:
        session_state["_snapshot_init_retry_count"] = 0
    if "_snapshot_empty_retry_count" not in session_state:
        session_state["_snapshot_empty_retry_count"] = 0
    if "_snapshot_mismatch_retry_count" not in session_state:
        session_state["_snapshot_mismatch_retry_count"] = 0
    if "_snapshot_load_waiting" not in session_state:
        session_state["_snapshot_load_waiting"] = False
    if "_snapshot_inflight_mode" not in session_state:
        session_state["_snapshot_inflight_mode"] = None
    if "_snapshot_inflight_request_id" not in session_state:
        session_state["_snapshot_inflight_request_id"] = None
    if "_outbox_init_retry_count" not in session_state:
        session_state["_outbox_init_retry_count"] = 0
    if "_outbox_load_waiting" not in session_state:
        session_state["_outbox_load_waiting"] = False
    if "_outbox_inflight_mode" not in session_state:
        session_state["_outbox_inflight_mode"] = None
    if "_outbox_inflight_request_id" not in session_state:
        session_state["_outbox_inflight_request_id"] = None
    if "_outbox_restore_completed" not in session_state:
        session_state["_outbox_restore_completed"] = False
    if "_outbox_loaded_message" not in session_state:
        session_state["_outbox_loaded_message"] = None


def _get_or_create_request_id(session_state, prefix, mode):
    mode_key = f"_{prefix}_inflight_mode"
    req_key = f"_{prefix}_inflight_request_id"
    current_mode = session_state.get(mode_key)
    current_req = session_state.get(req_key)
    if current_mode == mode and current_req:
        return current_req
    new_req = f"{mode}:{uuid4().hex}"
    session_state[mode_key] = mode
    session_state[req_key] = new_req
    return new_req


def _clear_inflight_request(session_state, prefix):
    session_state[f"_{prefix}_inflight_mode"] = None
    session_state[f"_{prefix}_inflight_request_id"] = None


def _process_outbox_load_result(session_state, principal, status):
    if not (status.get("ok") and status.get("found")):
        session_state["_outbox_loaded_message"] = None
        return

    payload_json = status.get("payload_json")
    try:
        payload = json.loads(payload_json or "{}")
    except Exception:
        snapshot_log("outbox.load_invalid")
        session_state["_outbox_loaded_message"] = None
        return

    saved_principal = payload.get("principal")
    if saved_principal != principal:
        snapshot_log("outbox.load_principal_mismatch", saved_principal=saved_principal, current_principal=principal)
        session_state["_outbox_loaded_message"] = None
        return

    pending_msg = payload.get("pending_user_message")
    if isinstance(pending_msg, dict):
        session_state["_outbox_loaded_message"] = pending_msg
        snapshot_log("outbox.loaded", message_id=pending_msg.get("id"))
    else:
        session_state["_outbox_loaded_message"] = None


def run_outbox_bridge(session_state, principal):
    _ensure_bridge_state(session_state)

    pending = session_state.get("_outbox_save_pending")
    mode = "idle"
    payload_json = ""
    request_id = ""
    load_waiting = bool(session_state.get("_outbox_load_waiting"))

    if session_state.get("_outbox_clear_pending"):
        mode = "clear"
    elif not session_state.get("_outbox_restore_completed", False):
        if not load_waiting:
            mode = "load"
        else:
            mode = "idle"
            request_id = str(session_state.get("_outbox_inflight_request_id") or "")
    elif isinstance(pending, dict):
        if pending.get("principal") != principal:
            session_state["_outbox_save_pending"] = None
            mode = "idle"
        else:
            mode = "save"
            payload_json = pending.get("payload_json", "{}")

    if mode in ("load", "save", "clear"):
        request_id = _get_or_create_request_id(session_state, "outbox", mode)

    status = session_storage_bridge(
        mode=mode,
        storage_key=SESSION_OUTBOX_KEY,
        payload_json=payload_json,
        trim_step_bytes=SESSION_SNAPSHOT_TRIM_STEP_BYTES,
        request_id=request_id,
        key="session_storage_bridge_outbox",
    )

    if mode == "load":
        session_state["_outbox_load_waiting"] = True
        return

    if mode == "idle" and not load_waiting:
        return

    if not isinstance(status, dict):
        return

    expected_request_id = request_id if request_id else str(session_state.get("_outbox_inflight_request_id") or "")
    if status.get("request_id") != expected_request_id:
        return

    if mode == "idle":
        if status.get("mode") != "load":
            return
    elif status.get("mode") != mode:
        return

    if status.get("status") == "init":
        retry_count = session_state.get("_outbox_init_retry_count", 0) + 1
        session_state["_outbox_init_retry_count"] = retry_count
        if retry_count <= SNAPSHOT_INIT_MAX_RETRY:
            if mode != "idle":
                session_state["need_rerun"] = True
            return

        _clear_inflight_request(session_state, "outbox")
        session_state["_outbox_load_waiting"] = False
        if mode in ("load", "idle"):
            session_state["_outbox_restore_completed"] = True
            session_state["_outbox_loaded_message"] = None
        elif mode == "save":
            session_state["_outbox_save_pending"] = None
        elif mode == "clear":
            session_state["_outbox_clear_pending"] = False
        return

    session_state["_outbox_init_retry_count"] = 0
    _clear_inflight_request(session_state, "outbox")
    if mode in ("load", "idle"):
        session_state["_outbox_load_waiting"] = False

    if mode == "clear":
        session_state["_outbox_clear_pending"] = False
        session_state["_outbox_loaded_message"] = None
        return

    if mode in ("load", "idle"):
        _process_outbox_load_result(session_state, principal, status)
        session_state["_outbox_restore_completed"] = True
        return

    if mode == "save":
        session_state["_outbox_save_pending"] = None


def run_snapshot_bridge(session_state, principal, clients, assistants, on_save_failed=None):
    _ensure_bridge_state(session_state)

    pending = session_state.get("_snapshot_save_pending")
    mode = "idle"
    payload_json = ""
    request_id = ""
    load_waiting = bool(session_state.get("_snapshot_load_waiting"))

    if session_state.get("_snapshot_clear_pending"):
        mode = "clear"
    elif not session_state.get("_snapshot_restore_completed", False):
        if not load_waiting:
            mode = "load"
        else:
            mode = "idle"
            request_id = str(session_state.get("_snapshot_inflight_request_id") or "")
    elif isinstance(pending, dict):
        if pending.get("principal") != principal:
            session_state["_snapshot_save_pending"] = None
            mode = "idle"
        else:
            mode = "save"
            payload_json = pending.get("payload_json", "{}")

    if mode in ("load", "save", "clear"):
        snapshot_log("bridge.mode", mode=mode)
        request_id = _get_or_create_request_id(session_state, "snapshot", mode)

    status = session_storage_bridge(
        mode=mode,
        storage_key=SESSION_SNAPSHOT_KEY,
        payload_json=payload_json,
        trim_step_bytes=SESSION_SNAPSHOT_TRIM_STEP_BYTES,
        request_id=request_id,
        key="session_storage_bridge_single",
    )

    if mode == "load":
        session_state["_snapshot_load_waiting"] = True
        return

    if mode == "idle" and not load_waiting:
        return

    if not isinstance(status, dict):
        return

    expected_request_id = request_id if request_id else str(session_state.get("_snapshot_inflight_request_id") or "")
    if status.get("request_id") != expected_request_id:
        return

    if mode == "idle":
        if status.get("mode") != "load":
            return
    elif status.get("mode") != mode:
        return

    snapshot_log(
        "bridge.status",
        mode=mode,
        status=status.get("status"),
        ok=status.get("ok"),
        found=status.get("found"),
    )

    if status.get("status") == "init":
        retry_count = session_state.get("_snapshot_init_retry_count", 0) + 1
        session_state["_snapshot_init_retry_count"] = retry_count
        if retry_count <= SNAPSHOT_INIT_MAX_RETRY:
            snapshot_log("bridge.init_retry", retry_count=retry_count, mode=mode)
            if mode != "idle":
                session_state["need_rerun"] = True
            return

        snapshot_log("bridge.init_timeout", retry_count=retry_count, mode=mode)
        _clear_inflight_request(session_state, "snapshot")
        session_state["_snapshot_load_waiting"] = False
        if mode in ("load", "idle"):
            session_state["_snapshot_restore_completed"] = True
        elif mode == "save":
            session_state["_snapshot_save_pending"] = None
        return

    session_state["_snapshot_init_retry_count"] = 0
    _clear_inflight_request(session_state, "snapshot")
    if mode in ("load", "idle"):
        session_state["_snapshot_load_waiting"] = False

    if mode == "clear":
        session_state["_snapshot_clear_pending"] = False
        session_state["_snapshot_restore_completed"] = True
        return

    if mode in ("load", "idle"):
        if status.get("ok") and status.get("found"):
            applied = _apply_loaded_snapshot(principal, status.get("payload_json"), clients, assistants, session_state)
            if applied is True:
                session_state["_snapshot_restore_completed"] = True
                session_state["_snapshot_empty_retry_count"] = 0
                session_state["_snapshot_mismatch_retry_count"] = 0
                snapshot_log("load.completed", reason="loaded")
            elif applied == "mismatch":
                snapshot_log("load.wait_principal_recovery")
                mismatch_retry = session_state.get("_snapshot_mismatch_retry_count", 0) + 1
                session_state["_snapshot_mismatch_retry_count"] = mismatch_retry
                if mismatch_retry <= SNAPSHOT_MISMATCH_MAX_RETRY:
                    session_state["_snapshot_load_waiting"] = False
                    session_state["need_rerun"] = True
                    return

                # principal が回復しない場合は復元完了扱いにしてループを止める
                session_state["_snapshot_restore_completed"] = True
                snapshot_log("load.completed", reason="principal_mismatch_timeout")
                return
            elif applied == "clear":
                session_state["_snapshot_clear_pending"] = True
                session_state["need_rerun"] = True
                snapshot_log("load.schedule_clear")
                return
            else:
                session_state["_snapshot_restore_completed"] = True
                snapshot_log("load.completed", reason="invalid_payload")
        elif status.get("ok"):
            retry_count = session_state.get("_snapshot_empty_retry_count", 0) + 1
            session_state["_snapshot_empty_retry_count"] = retry_count
            if retry_count <= SNAPSHOT_EMPTY_MAX_RETRY:
                snapshot_log("load.empty_retry", retry_count=retry_count)
                session_state["_snapshot_load_waiting"] = False
                session_state["need_rerun"] = True
                return

            session_state["_snapshot_restore_completed"] = True
            session_state["_snapshot_empty_retry_count"] = 0
            snapshot_log("load.empty_confirmed", retry_count=retry_count)
            snapshot_log("load.completed", reason="empty_confirmed")
        return

    if mode == "save":
        snapshot_log("save.process_bridge_response", status=status)
        save_status = status.get("status")
        if save_status in ("saved", "saved_trimmed"):
            session_state["_snapshot_save_pending"] = None
            return

        if save_status == "stale_write_blocked":
            # ブラウザ側により新しい履歴がある。ここで強制ロードすると
            # 現在セッションの表示を巻き戻し得るため、保存は諦めて継続する。
            session_state["_snapshot_save_pending"] = None
            snapshot_log("save.failed", status="stale_write_blocked")
            if on_save_failed:
                on_save_failed("stale_write_blocked")
            return

        if status.get("ok") is False:
            fail_status = status.get("status", "unknown")
            snapshot_log("save.failed", status=fail_status)
            if on_save_failed:
                on_save_failed(fail_status)
            return

        # save応答として不正なステータスは再試行する
        session_state["need_rerun"] = True
