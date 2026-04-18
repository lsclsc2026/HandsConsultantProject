import re
import time
from typing import Any

import requests
import streamlit as st

API_BASE_DEFAULT = "http://127.0.0.1:8099/api/v1"

st.set_page_config(
    page_title="手相双Agent对话系统",
    page_icon="🖐️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans SC', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(1000px 360px at 100% -10%, rgba(61,169,163,0.16), transparent 70%),
            radial-gradient(800px 360px at -10% 5%, rgba(242,194,107,0.18), transparent 65%),
            linear-gradient(180deg, #f8fbfd 0%, #f2f6f9 100%);
    }

    .main .block-container {
        max-width: 980px;
        padding-top: 1.2rem;
        padding-bottom: 6.6rem;
    }

    .hero-wrap {
        border: 1px solid #d9e3ea;
        border-radius: 22px;
        padding: 1rem 1.15rem;
        background: linear-gradient(150deg, rgba(255,255,255,0.95), rgba(236,243,247,0.92));
        box-shadow: 0 12px 30px rgba(25,50,74,0.08);
        margin-bottom: 1rem;
    }

    .hero-title {
        margin: 0;
        color: #19324a;
        font-size: 1.86rem;
        font-weight: 800;
        letter-spacing: 0.2px;
    }

    .hero-sub {
        margin: 0.35rem 0 0;
        color: #38546e;
        font-size: 1.02rem;
        font-weight: 600;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        border: 1px solid #d9e3ea;
        background: rgba(255, 255, 255, 0.97);
        box-shadow: 0 7px 18px rgba(15,43,66,0.07);
    }

    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        font-size: 1.05rem;
        line-height: 1.82;
        color: #13293d;
        font-weight: 500;
    }

    .think-wrap {
        margin-top: 0.55rem;
        padding: 0.75rem 0.9rem;
        border-radius: 12px;
        border: 1px dashed #c8d6e2;
        background: #f7fafc;
    }

    .answer-wrap {
        margin-top: 0.65rem;
        padding: 0.8rem 0.9rem;
        border-radius: 12px;
        border: 1px solid #d8e5de;
        background: #fbfefd;
    }

    [data-testid="stChatInput"] {
        position: fixed;
        left: 0;
        right: 0;
        margin: 0 auto;
        bottom: 1rem;
        width: calc(100% - 1.4rem);
        max-width: 940px;
        background: rgba(255,255,255,0.94);
        border: 1px solid #d9e3ea;
        border-radius: 18px;
        box-shadow: 0 16px 34px rgba(15,43,66,0.13);
        padding: 0.22rem 0.72rem;
        backdrop-filter: blur(8px);
        box-sizing: border-box;
        overflow: hidden;
    }

    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] form,
    [data-testid="stChatInput"] textarea {
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
    }

    [data-testid="stChatInput"] textarea {
        font-size: 1.04rem;
        font-weight: 500;
        color: #13293d;
    }

    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        border-radius: 14px;
        font-weight: 700;
        border: 1px solid #c9d8e3;
    }

    section[data-testid="stSidebar"] .stButton button[kind="secondary"] {
        background: #f6fbfd;
        color: #23445e;
    }

    section[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #214a6b 0%, #2b6689 100%);
        color: #ffffff;
        border-color: #214a6b;
        box-shadow: 0 8px 18px rgba(33,74,107,0.22);
    }

    @media (max-width: 768px) {
        .main .block-container {
            padding-bottom: 7rem;
        }

        [data-testid="stChatInput"] {
            width: calc(100% - 0.8rem);
            max-width: none;
            bottom: 0.35rem;
            border-radius: 14px;
            padding: 0.18rem 0.48rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _api_get(base_url: str, path: str) -> tuple[bool, Any]:
    try:
        resp = requests.get(f"{base_url}{path}", timeout=60)
        return resp.status_code < 400, resp.json()
    except Exception as exc:  # noqa: BLE001
        return False, {"detail": str(exc)}



def _api_post(base_url: str, path: str, **kwargs: Any) -> tuple[bool, Any]:
    try:
        resp = requests.post(f"{base_url}{path}", timeout=120, **kwargs)
        return resp.status_code < 400, resp.json()
    except Exception as exc:  # noqa: BLE001
        return False, {"detail": str(exc)}



def _api_delete(base_url: str, path: str) -> tuple[bool, Any]:
    try:
        resp = requests.delete(f"{base_url}{path}", timeout=60)
        return resp.status_code < 400, resp.json()
    except Exception as exc:  # noqa: BLE001
        return False, {"detail": str(exc)}



def _split_think_and_answer(content: str) -> tuple[str, str]:
    if not content:
        return "", ""
    think_parts = re.findall(r"<think>([\s\S]*?)</think>", content)
    thinking = "\n\n".join(part.strip() for part in think_parts if part.strip())
    answer = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
    if not answer:
        answer = "（模型未返回正式回答，仅返回了思考片段或空内容）"
    return thinking, answer


def _split_initial_sections(content: str) -> tuple[str, str]:
    if not content:
        return "", ""
    base_match = re.search(r"【手相的基础信息】\s*([\s\S]*?)(?:【综合信息】|$)", content)
    report_match = re.search(r"【综合信息】\s*([\s\S]*)$", content)
    base_info = base_match.group(1).strip() if base_match else ""
    report = report_match.group(1).strip() if report_match else content.strip()
    return base_info, report



def _stream_text(placeholder: st.delta_generator.DeltaGenerator, text: str, chunk_size: int = 12) -> None:
    if not text:
        placeholder.markdown("（无）")
        return
    buf = ""
    for idx in range(0, len(text), chunk_size):
        buf += text[idx: idx + chunk_size]
        placeholder.markdown(buf)
        time.sleep(0.015)



def _render_assistant(content: str, stream: bool = False) -> None:
    _, answer = _split_think_and_answer(content)
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()

        if stream:
            _stream_text(answer_placeholder, answer)
        else:
            answer_placeholder.markdown(answer)


def _render_initial_sections(content: str, stream: bool = False) -> None:
    base_info, report = _split_initial_sections(content)
    with st.chat_message("assistant"):
        st.markdown("### 手相的基础信息")
        base_placeholder = st.empty()
        st.markdown("### 综合信息")
        report_placeholder = st.empty()
        if stream:
            _stream_text(base_placeholder, base_info)
            _stream_text(report_placeholder, report)
        else:
            base_placeholder.markdown(base_info if base_info else "（无）")
            report_placeholder.markdown(report if report else "（无）")



def _init_state() -> None:
    defaults = {
        "api_base": API_BASE_DEFAULT,
        "sessions": [],
        "active_session_id": "",
        "active_record": {"history": [], "session_id": "", "profile": None},
        "pending_stream": None,
        "flash": None,
        "upload_assets": {},
        "gate_results": {},
        "pending_user_query": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value



def _refresh_sessions(base_url: str) -> None:
    ok, data = _api_get(base_url, "/sessions")
    st.session_state.sessions = data if ok and isinstance(data, list) else []



def _load_session(base_url: str, session_id: str) -> None:
    ok, data = _api_get(base_url, f"/sessions/{session_id}")
    if ok:
        st.session_state.active_record = data
    else:
        st.session_state.active_record = {"history": [], "session_id": session_id, "profile": None}



def _ensure_active_session(base_url: str) -> None:
    _refresh_sessions(base_url)
    if not st.session_state.sessions:
        ok, data = _api_post(base_url, "/sessions")
        if ok:
            st.session_state.active_session_id = data.get("session_id", "")
            _refresh_sessions(base_url)
        else:
            st.session_state.flash = ("error", f"创建会话失败: {data.get('detail', '未知错误')}")
            return

    if not st.session_state.active_session_id:
        st.session_state.active_session_id = st.session_state.sessions[0]["session_id"]

    all_ids = {item["session_id"] for item in st.session_state.sessions}
    if st.session_state.active_session_id not in all_ids:
        st.session_state.active_session_id = st.session_state.sessions[0]["session_id"]

    _load_session(base_url, st.session_state.active_session_id)



def _active_has_profile() -> bool:
    return bool(st.session_state.active_record.get("profile"))



def _get_active_asset() -> dict[str, Any] | None:
    return st.session_state.upload_assets.get(st.session_state.active_session_id)


def _upsert_active_asset(name: str, mime: str, uploaded_bytes: bytes) -> None:
    session_id = st.session_state.active_session_id
    upload_sig = f"{name}:{len(uploaded_bytes)}"
    existing = st.session_state.upload_assets.get(session_id)

    if existing and existing.get("sig") == upload_sig:
        existing["name"] = name
        existing["mime"] = mime
        existing["bytes"] = uploaded_bytes
        return

    st.session_state.upload_assets[session_id] = {
        "name": name,
        "mime": mime,
        "bytes": uploaded_bytes,
        "sig": upload_sig,
        "analyzed_sig": None,
    }
    st.session_state.gate_results.pop(session_id, None)


def _gate_status_for(session_id: str) -> dict[str, Any] | None:
    return st.session_state.gate_results.get(session_id)



def _render_flash() -> None:
    flash = st.session_state.get("flash")
    if not flash:
        return
    level, message = flash
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)
    st.session_state.flash = None



_init_state()
api_base = st.session_state.api_base.strip()
_ensure_active_session(api_base)
active_sid = st.session_state.active_session_id
active_record = st.session_state.active_record

with st.sidebar:
    st.markdown("## 会话管理")
    action_cols = st.columns(2)
    if action_cols[0].button("新建会话", use_container_width=True):
        ok, data = _api_post(api_base, "/sessions")
        if ok:
            st.session_state.active_session_id = data.get("session_id", "")
            st.session_state.flash = ("success", "已新建会话")
            st.rerun()
        st.session_state.flash = ("error", data.get("detail", "新建会话失败"))
        st.rerun()

    if action_cols[1].button("删除当前", use_container_width=True):
        sid = st.session_state.active_session_id
        ok, data = _api_delete(api_base, f"/sessions/{sid}")
        if ok and data.get("deleted", False):
            st.session_state.upload_assets.pop(sid, None)
            st.session_state.active_session_id = ""
            st.session_state.flash = ("success", "当前会话已删除")
        else:
            st.session_state.flash = ("warning", "删除失败或会话不存在")
        st.rerun()

    if st.button("重建知识库", use_container_width=True):
        ok, data = _api_post(api_base, "/knowledge/rebuild")
        if ok:
            st.session_state.flash = (
                "success",
                f"知识库已重连 | 规则 {data.get('rule_docs', 0)} | QA {data.get('qa_docs', 0)} | 图谱 {data.get('graph_nodes', 0)} 节点",
            )
        else:
            st.session_state.flash = ("error", data.get("detail", "重建知识库失败"))
        st.rerun()

    st.caption("历史会话")
    current_sid = st.session_state.active_session_id
    for session in st.session_state.sessions:
        sid = session.get("session_id", "")
        label = session.get("preview", "新对话") or "新对话"
        count = session.get("message_count", 0)
        if st.button(
            f"{label[:16]} ({count})",
            key=f"session_{sid}",
            use_container_width=True,
            type="primary" if sid == current_sid else "secondary",
        ):
            st.session_state.active_session_id = sid
            st.rerun()

    st.markdown("---")
    st.markdown("### 当前手相图")
    uploaded_file = st.file_uploader(
        "上传手相图片",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"palm_upload_{active_sid}",
    )
    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        _upsert_active_asset(
            uploaded_file.name,
            uploaded_file.type or "image/jpeg",
            uploaded_bytes,
        )

    asset = _get_active_asset()
    if asset is not None:
        st.image(asset["bytes"], width=240)
        st.caption(f"{asset['name']} | {len(asset['bytes']) / 1024:.1f} KB")
    else:
        st.caption("未上传图片")

    gate_info = _gate_status_for(active_sid)
    if gate_info:
        category = gate_info.get("category", "")
        confidence = gate_info.get("confidence", 0)
        reason = gate_info.get("reason", "")
        if category == "palm":
            st.success(f"检测通过 | 置信度 {confidence}")
        elif category:
            st.warning(f"检测结果 {category} | 置信度 {confidence}")
        if reason:
            st.caption(reason)

    with st.expander("连接设置", expanded=False):
        st.session_state.api_base = st.text_input("API Base URL", value=st.session_state.api_base)
        st.caption("默认: http://127.0.0.1:8099/api/v1")

st.markdown(
    """
    <div class='hero-wrap'>
        <h1 class='hero-title'>手相双Agent对话系统</h1>
        <p class='hero-sub'>上传手相图片后，直接在底部输入你的问题。首轮会自动把“图片 + 提问”作为一次完整对话发送。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

_render_flash()

if asset := _get_active_asset():
    st.image(asset["bytes"], caption="当前会话手相图", width=300)

if asset := _get_active_asset():
    analyzed_sig = asset.get("analyzed_sig")
    if asset.get("sig") and asset.get("sig") != analyzed_sig:
        with st.spinner("正在进行手相检测与基础解读..."):
            ok, payload = _api_post(
                api_base,
                "/palm/analyze",
                files={"image": (asset["name"], asset["bytes"], asset["mime"])},
                data={
                    "session_id": active_sid,
                    "initial_query": "请先给出基础的手相综合解读，简洁一些。",
                },
            )
        if not ok:
            st.session_state.flash = ("error", payload.get("detail", "图像检测失败"))
        else:
            gate = payload.get("gate", {})
            st.session_state.gate_results[active_sid] = gate
            asset["analyzed_sig"] = asset.get("sig")
            if gate.get("category") == "palm":
                base_info = payload.get("base_info", "")
                report = payload.get("report", "")
                st.session_state.pending_stream = {
                    "session_id": active_sid,
                    "mode": "initial",
                    "content": f"【手相的基础信息】\n{base_info}\n\n【综合信息】\n{report}",
                }
            else:
                st.session_state.pending_stream = None
            st.session_state.active_session_id = payload.get("session_id", active_sid)
            _load_session(api_base, st.session_state.active_session_id)
        st.rerun()

history = active_record.get("history", [])
pending = st.session_state.pending_stream
history_to_render = history
should_stream = False
if (
    pending
    and pending.get("session_id") == active_sid
    and history
    and history[-1].get("role") == "assistant"
    and history[-1].get("content") == pending.get("content")
):
    history_to_render = history[:-1]
    should_stream = True

if not history and not should_stream:
    st.info("当前会话暂无内容。先在左侧上传手相图，再在底部输入你的第一个问题。")

for msg in history_to_render:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    elif role == "assistant":
        if content.startswith("【手相的基础信息】"):
            _render_initial_sections(content, stream=False)
        else:
            _render_assistant(content, stream=False)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)

if should_stream:
    if pending.get("mode") == "initial":
        _render_initial_sections(pending.get("content", ""), stream=True)
    else:
        _render_assistant(pending.get("content", ""), stream=True)
    st.session_state.pending_stream = None

profile_ready = _active_has_profile()
user_input = st.chat_input(
    "输入你的问题，例如：我未来两年财运与感情如何？",
    disabled=not profile_ready,
)
if user_input:
    query = user_input.strip()
    if not query:
        st.stop()

    with st.chat_message("user"):
        st.markdown(query)

    ok, payload = _api_post(
        api_base,
        "/palm/chat",
        json={
            "session_id": active_sid,
            "query": query,
        },
    )
    if not ok:
        st.session_state.flash = ("error", payload.get("detail", "追问失败"))
        st.rerun()

    _load_session(api_base, active_sid)
    st.session_state.pending_stream = {
        "session_id": active_sid,
        "mode": "chat",
        "content": payload.get("answer", ""),
    }
    st.rerun()
