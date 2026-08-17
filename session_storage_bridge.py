import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(__file__), "session_storage_component")
_session_storage_component = components.declare_component(
    "session_storage_component",
    path=_COMPONENT_DIR,
)


def session_storage_bridge(
    mode,
    storage_key,
    payload_json="",
    trim_step_bytes=102400,
    request_id="",
    key=None,
):
    return _session_storage_component(
        mode=mode,
        storage_key=storage_key,
        payload_json=payload_json,
        trim_step_bytes=trim_step_bytes,
        request_id=request_id,
        key=key,
        default={"ok": False, "status": "init"},
    )
