"""Reusable Streamlit-native server path pickers.

The application runs on a remote Linux server while Streamlit is rendered in the
user's browser.  Therefore the browser cannot use the desktop's native file
chooser to browse *server* paths.  This module provides a small Streamlit modal
that browses the server filesystem directly and requires no desktop GUI or X11 setup.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Literal

import streamlit as st


_ACTIVE_PICKER_KEY = "__medical_ai_server_picker_active"


def _hover_tooltip(label: str, value: str) -> None:
    label_json = json.dumps(str(label), ensure_ascii=False)
    value_json = json.dumps(str(value or ""), ensure_ascii=False)
    st.markdown(
        f"""
        <style>
        div[data-testid="stTextInput"]:has(input[aria-label={label_json}]) {{position: relative;}}
        div[data-testid="stTextInput"]:has(input[aria-label={label_json}]):hover::after {{
            content: {value_json}; position: absolute; left: 0; bottom: calc(100% + 4px);
            z-index: 99999; width: max-content; max-width: min(520px, 92vw);
            padding: 7px 9px; border-radius: 6px;
            background: rgba(28,28,28,.97); color: white; font-size: 12px; line-height: 1.35;
            white-space: normal; overflow-wrap: anywhere; pointer-events: none; box-sizing: border-box;
            box-shadow: 0 2px 10px rgba(0,0,0,.25);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _initial_directory(value: str | os.PathLike | None) -> Path:
    """Choose a useful starting directory for an existing or partially typed path."""
    text = str(value or "").strip()
    candidates: list[Path] = []
    if text:
        raw = Path(text).expanduser()
        candidates.extend([raw, raw.parent])
        if not raw.is_absolute():
            candidates.extend([(Path.cwd() / raw), (Path.cwd() / raw).parent])
    candidates.extend([Path.cwd(), Path.home(), Path("/")])

    for candidate in candidates:
        try:
            if candidate.is_file():
                candidate = candidate.parent
            if candidate.is_dir():
                return candidate.resolve()
        except OSError:
            continue
    return Path("/")


def _allowed_extensions(file_filter: str) -> tuple[str, ...]:
    """Extract extensions from the legacy Qt-style filter string used by callers."""
    matches = re.findall(r"\*\.([A-Za-z0-9_+-]+)", str(file_filter or ""))
    return tuple(sorted({f".{m.lower()}" for m in matches}))


def _list_directory(path: Path, *, extensions: tuple[str, ...] = ()) -> tuple[list[Path], list[Path], str | None]:
    directories: list[Path] = []
    files: list[Path] = []
    try:
        with os.scandir(path) as iterator:
            for entry in iterator:
                try:
                    entry_path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        directories.append(entry_path)
                    elif entry.is_file(follow_symlinks=False):
                        if not extensions or entry_path.suffix.lower() in extensions:
                            files.append(entry_path)
                except OSError:
                    continue
    except (PermissionError, FileNotFoundError, NotADirectoryError, OSError) as exc:
        return [], [], str(exc)

    directories.sort(key=lambda p: p.name.lower())
    files.sort(key=lambda p: p.name.lower())
    return directories, files, None


def _set_cwd(widget_key: str, path: Path) -> None:
    try:
        resolved = path.expanduser().resolve()
    except OSError:
        resolved = path.expanduser()
    st.session_state[f"{widget_key}__browser_cwd"] = str(resolved)


def _close_picker(widget_key: str) -> None:
    st.session_state[f"{widget_key}__browser_open"] = False
    if st.session_state.get(_ACTIVE_PICKER_KEY) == widget_key:
        st.session_state.pop(_ACTIVE_PICKER_KEY, None)


def _navigate(widget_key: str, target: Path) -> None:
    """Navigate without mutating any already-instantiated widget state.

    Streamlit forbids assigning to ``st.session_state[key]`` after a widget with
    the same key has been instantiated during the current run.  The directory
    picker therefore keeps navigation state only in ``__browser_cwd``; the
    editable path remains the main input outside the dialog.
    """
    _set_cwd(widget_key, target)
    st.rerun()


def _choose(widget_key: str, selected: Path) -> None:
    st.session_state[f"{widget_key}__browser_pending"] = str(selected)
    _close_picker(widget_key)
    st.rerun()


def _dialog_header(widget_key: str, current: Path) -> None:
    """Render a deliberately small, stable server-directory toolbar."""
    st.caption("服务器路径")
    st.code(str(current), language="text")

    c1, c2 = st.columns(2)
    parent = current.parent if current.parent != current else current
    if c1.button("↑ 上一级", key=f"{widget_key}__picker_parent", use_container_width=True):
        _navigate(widget_key, parent)
    if c2.button("⌂ Home", key=f"{widget_key}__picker_home", use_container_width=True):
        _navigate(widget_key, Path.home())


def _render_entries(widget_key: str, current: Path, *, mode: Literal["directory", "file"], extensions: tuple[str, ...]) -> None:
    directories, files, error = _list_directory(current, extensions=extensions)
    if error:
        st.error(f"无法读取目录：{error}")
        return

    if not directories and (mode == "directory" or not files):
        st.caption("当前目录没有可显示的内容。")

    for folder in directories:
        if st.button(
            f"📁 {folder.name}",
            key=f"{widget_key}__dir_{_digest(str(folder))}",
            use_container_width=True,
        ):
            _navigate(widget_key, folder)

    if mode == "file":
        if directories and files:
            st.divider()
        for file_path in files:
            if st.button(
                f"📄 {file_path.name}",
                key=f"{widget_key}__file_{_digest(str(file_path))}",
                use_container_width=True,
            ):
                _choose(widget_key, file_path.resolve())


@st.dialog("选择服务器目录", width="small", dismissible=False)
def _directory_browser_dialog(widget_key: str, title: str) -> None:
    cwd_key = f"{widget_key}__browser_cwd"
    current = Path(st.session_state.get(cwd_key, str(Path.cwd()))).expanduser()
    if not current.is_dir():
        current = _initial_directory(current)
        _set_cwd(widget_key, current)

    st.markdown(f"**{title}**")
    _dialog_header(widget_key, current)
    st.divider()
    _render_entries(widget_key, current, mode="directory", extensions=())
    st.divider()

    c1, c2 = st.columns(2)
    if c1.button("选择当前目录", type="primary", key=f"{widget_key}__picker_select_dir", use_container_width=True):
        _choose(widget_key, current.resolve())
    if c2.button("取消", key=f"{widget_key}__picker_cancel_dir", use_container_width=True):
        _close_picker(widget_key)
        st.rerun()


@st.dialog("选择服务器文件", width="small", dismissible=False)
def _file_browser_dialog(widget_key: str, title: str, extensions: tuple[str, ...]) -> None:
    cwd_key = f"{widget_key}__browser_cwd"
    current = Path(st.session_state.get(cwd_key, str(Path.cwd()))).expanduser()
    if not current.is_dir():
        current = _initial_directory(current)
        _set_cwd(widget_key, current)

    st.markdown(f"**{title}**")
    if extensions:
        st.caption("文件类型：" + ", ".join(extensions))
    _dialog_header(widget_key, current)
    st.divider()
    _render_entries(widget_key, current, mode="file", extensions=extensions)
    st.divider()
    if st.button("取消", key=f"{widget_key}__picker_cancel_file", use_container_width=True):
        _close_picker(widget_key)
        st.rerun()


def path_input(
    label: str,
    *,
    key: str,
    value: str = "",
    mode: Literal["directory", "file"] = "directory",
    file_filter: str = "All files (*)",
    help_text: str | None = None,
    placeholder: str | None = None,
    button_label: str | None = None,
    editable: bool = False,
) -> str:
    """uiPSF-style single server-path selector for Streamlit.

    The original uiPSF desktop UI presents the current selection beside an
    ``Open Files``/``Open Dir`` button.  Streamlit runs in a browser, so it
    cannot reuse a desktop-native chooser for the *remote server* filesystem.  This
    control mirrors that interaction while opening the existing server-side
    Streamlit browser dialog.

    ``editable`` is intentionally off for normal filesystem paths so users do
    not have to type remote paths.  It is enabled only for fields such as a
    Hugging Face model id, where a non-filesystem identifier is also valid.
    """
    pending_key = f"{key}__browser_pending"
    open_key = f"{key}__browser_open"

    if pending_key in st.session_state:
        st.session_state[key] = st.session_state.pop(pending_key)

    if key not in st.session_state:
        st.session_state[key] = str(value or "")

    current = str(st.session_state.get(key, "") or "")
    open_label = button_label or ("Open Files" if mode == "file" else "Open Dir")

    # Match uiPSF's layout: current selection on the left, actions on the right.
    c1, c2 = st.columns([4.4, 1.6], vertical_alignment="bottom")
    with c1:
        if editable:
            current = st.text_input(
                label,
                key=key,
                help=help_text,
                placeholder=placeholder,
            )
            _hover_tooltip(label, current)
        else:
            display_key = f"{key}__display"
            # This key belongs only to the disabled display widget, so it is
            # safe to synchronize before the widget is instantiated.
            st.session_state[display_key] = current
            st.text_input(
                label,
                key=display_key,
                help=help_text,
                placeholder=placeholder or "No path selected",
                disabled=True,
            )
            _hover_tooltip(label, current)

    with c2:
        if st.button(open_label, key=f"{key}__browser_button", use_container_width=True):
            _set_cwd(key, _initial_directory(current))
            st.session_state[open_key] = True
            st.session_state[_ACTIVE_PICKER_KEY] = key
            st.rerun()
        if st.button("Clear All", key=f"{key}__browser_clear", use_container_width=True):
            st.session_state[key] = ""
            _close_picker(key)
            st.rerun()

    is_active = st.session_state.get(_ACTIVE_PICKER_KEY) == key
    if bool(st.session_state.get(open_key)) and is_active:
        if mode == "directory":
            _directory_browser_dialog(key, label)
        else:
            _file_browser_dialog(key, label, _allowed_extensions(file_filter))

    return str(current or "")


def inline_directory_input(
    label: str,
    *,
    key: str,
    value: str = "",
    help_text: str | None = None,
    button_label: str = "选择目录",
    allow_create: bool = False,
) -> str:
    """Directory selector safe to use *inside* an existing ``st.dialog``.

    ``path_input`` opens its own Streamlit dialog, which is ideal for normal
    sidebar fields but cannot be nested reliably inside another dialog.  The
    medical knowledge-base builder is itself a dialog, so this variant expands
    a compact server-side browser inline instead.  Users never need to type a
    remote absolute path.
    """
    open_key = f"{key}__inline_browser_open"
    cwd_key = f"{key}__inline_browser_cwd"

    if key not in st.session_state:
        st.session_state[key] = str(value or "")
    current_value = str(st.session_state.get(key, "") or "")

    c1, c2 = st.columns([4.8, 1.5], vertical_alignment="bottom")
    with c1:
        display_key = f"{key}__inline_display"
        st.session_state[display_key] = current_value
        st.text_input(
            label,
            key=display_key,
            help=help_text,
            placeholder="尚未选择服务器目录",
            disabled=True,
        )
        _hover_tooltip(label, current_value)
    with c2:
        if st.button(button_label, key=f"{key}__inline_open_button", use_container_width=True):
            st.session_state[cwd_key] = str(_initial_directory(current_value))
            st.session_state[open_key] = True
            st.rerun()

    if not bool(st.session_state.get(open_key, False)):
        return current_value

    current = Path(st.session_state.get(cwd_key, str(_initial_directory(current_value)))).expanduser()
    if not current.is_dir():
        current = _initial_directory(current)
        st.session_state[cwd_key] = str(current)

    with st.container(border=True):
        st.caption(f"正在浏览服务器：`{current}`")
        nav1, nav2, nav3 = st.columns(3)
        parent = current.parent if current.parent != current else current
        if nav1.button("↑ 上一级", key=f"{key}__inline_parent", use_container_width=True):
            st.session_state[cwd_key] = str(parent)
            st.rerun()
        if nav2.button("⌂ Home", key=f"{key}__inline_home", use_container_width=True):
            st.session_state[cwd_key] = str(Path.home())
            st.rerun()
        if nav3.button("✕ 关闭", key=f"{key}__inline_close", use_container_width=True):
            st.session_state[open_key] = False
            st.rerun()

        directories, _, error = _list_directory(current)
        if error:
            st.error(f"无法读取目录：{error}")
        elif not directories:
            st.caption("当前目录没有子目录。")
        else:
            for folder in directories:
                if st.button(
                    f"📁 {folder.name}",
                    key=f"{key}__inline_dir_{_digest(str(folder))}",
                    use_container_width=True,
                ):
                    st.session_state[cwd_key] = str(folder.resolve())
                    st.rerun()

        if allow_create:
            st.divider()
            new_name = st.text_input(
                "新建子目录名称",
                key=f"{key}__inline_new_folder_name",
                placeholder="例如 medical_kb_demo",
            ).strip()
            if st.button(
                "新建子目录",
                key=f"{key}__inline_create_folder",
                disabled=not bool(new_name),
                use_container_width=True,
            ):
                if Path(new_name).name != new_name or new_name in {".", ".."}:
                    st.error("请输入单个目录名称，不要包含 /。")
                else:
                    try:
                        created = (current / new_name).resolve()
                        created.mkdir(parents=False, exist_ok=True)
                        st.session_state[cwd_key] = str(created)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"新建目录失败：{exc}")

        st.divider()
        if st.button(
            "选择当前目录",
            type="primary",
            key=f"{key}__inline_choose_current",
            use_container_width=True,
        ):
            st.session_state[key] = str(current.resolve())
            st.session_state[open_key] = False
            st.rerun()

    return str(st.session_state.get(key, "") or "")
