"""OpenAI-compatible client for the vLLM medical VLM service."""
import base64
import mimetypes
import time
from pathlib import Path


def _data_url(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _history_messages(history):
    messages = []
    for item in history or []:
        role = str(item.get("role", "")).strip()
        text = str(item.get("content", "")).strip()
        if role in {"user", "assistant", "system"} and text:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})
    return messages


def chat_completion(
    endpoint,
    model,
    prompt,
    image_path=None,
    max_tokens=256,
    temperature=0.0,
    timeout=180,
    history=None,
):
    import requests

    messages = _history_messages(history)
    content = []
    if image_path:
        content.append({"type": "image_url", "image_url": {"url": _data_url(image_path)}})
    content.append({"type": "text", "text": str(prompt)})
    messages.append({"role": "user", "content": content})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    started = time.perf_counter()
    response = requests.post(endpoint.rstrip("/") + "/chat/completions", json=payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    completion = int(usage.get("completion_tokens") or 0)
    return {
        "text": text,
        "profile": {
            "total_latency_s": elapsed,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": completion,
            "tokens_per_second": completion / elapsed if elapsed > 0 and completion else 0.0,
            "peak_vram_mb": 0.0,
        },
        "raw": data,
    }
