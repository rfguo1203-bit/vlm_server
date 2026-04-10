#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[apc-multimodal] {timestamp} {message}", flush=True)


def _build_image_part(request_mode: str, image_path: Path) -> dict:
    if request_mode == "path":
        return {
            "type": "image_path",
            "image_path": {
                "path": str(image_path),
            },
        }
    if request_mode == "base64":
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return {
            "type": "image_base64",
            "image_base64": {
                "data": encoded,
            },
        }
    raise SystemExit(f"unsupported REQUEST_MODE={request_mode}, expected path or base64")


def _post_json(
    server_url: str,
    request_id: str,
    body: dict,
    timeout_seconds: float,
) -> tuple[dict, dict[str, str], float]:
    started_at = time.perf_counter()
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        server_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_body = json.loads(response.read().decode("utf-8"))
        response_headers = {key.lower(): value for key, value in response.headers.items()}
    latency_ms = (time.perf_counter() - started_at) * 1000.0
    return response_body, response_headers, latency_ms


def _reset_caches(
    base_server_url: str,
    request_id: str,
    timeout_seconds: float,
) -> tuple[dict, dict[str, str], float]:
    reset_url = f"{base_server_url}/internal/reset-caches"
    return _post_json(
        server_url=reset_url,
        request_id=request_id,
        body={
            "reset_prefix_cache": True,
            "reset_mm_cache": True,
            "reset_running_requests": False,
        },
        timeout_seconds=timeout_seconds,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _assistant_text(response: dict) -> str:
    return response["choices"][0]["message"]["content"]


def _build_long_text_block(seed_text: str, repeat_count: int) -> str:
    block = (
        "Multimodal prefix-caching evaluation context. "
        "Keep this text unchanged across requests so the shared prompt prefix stays large. "
        f"Seed text: {seed_text}\n"
    )
    return block * repeat_count


def main() -> int:
    base_server_url = _env("SERVER_URL", "http://127.0.0.1:8972").rstrip("/")
    server_url = f"{base_server_url}/v1/chat/completions"
    model_name = _env("MODEL_NAME", "Qwen3.5-27B")
    request_mode = _env("REQUEST_MODE", "path")
    image_path = Path(_env("IMAGE_PATH", "/tmp/example.png"))
    max_tokens = int(_env("MAX_TOKENS", "8"))
    temperature = float(_env("TEMPERATURE", "0.0"))
    timeout_seconds = float(_env("CLIENT_TIMEOUT_SECONDS", "240"))
    output_dir = Path(_env("OUTPUT_DIR", "tmp/apc_multimodal"))
    session_id = _env("SESSION_ID", f"apc-mm-{uuid.uuid4().hex}")
    new_session_id = _env("NEW_SESSION_ID", f"apc-mm-new-{uuid.uuid4().hex}")
    prompt_repeat_count = int(_env("PROMPT_REPEAT_COUNT", "192"))
    long_prefix = _build_long_text_block(
        _env("PREFIX_SEED_TEXT", "Primary object tracking. cache-check."),
        prompt_repeat_count,
    )
    round1_prompt = _env(
        "ROUND1_PROMPT",
        (
            f"{long_prefix}\n"
            "Question: Identify the main object in the image. "
            "Answer with only a short noun phrase."
        ),
    )
    round2_prompt = _env(
        "ROUND2_PROMPT",
        (
            f"{long_prefix}\n"
            "Follow-up question: Repeat the same main object from the image. "
            "Answer with only a short noun phrase."
        ),
    )

    if not image_path.is_file():
        raise SystemExit(f"image file does not exist: {image_path}")

    image_part = _build_image_part(request_mode, image_path)
    image_bytes = image_path.stat().st_size
    messages: list[dict] = []

    _log(f"server_url={server_url}")
    _log(f"model_name={model_name}")
    _log("mode=prompt_heavy_decode_light")
    _log(f"prompt_repeat_count={prompt_repeat_count}")
    _log(f"max_tokens={max_tokens}")
    _log(f"temperature={temperature}")
    _log(f"request_mode={request_mode}")
    _log(f"image_path={image_path}")
    _log(f"image_bytes={image_bytes}")
    _log(f"session_id={session_id}")
    _log(f"new_session_id={new_session_id}")
    _log(
        "goal=reset caches before a fresh multimodal conversation, send image+text then follow-up, "
        "then reset caches again before starting a brand-new multimodal conversation"
    )
    _log("tip=compare server logs by request_id to judge whether image prefixes benefit from APC before the next explicit reset")

    try:
        reset1_request_id = f"{session_id}-reset-before-round1"
        _log(f"reset start request_id={reset1_request_id} reset_prefix_cache=true reset_mm_cache=true")
        reset1_response, reset1_headers, reset1_latency_ms = _reset_caches(
            base_server_url=base_server_url,
            request_id=reset1_request_id,
            timeout_seconds=timeout_seconds,
        )
        _log(
            f"reset done request_id={reset1_request_id} returned_request_id={reset1_headers.get('x-request-id', 'missing')} "
            f"latency_ms={reset1_latency_ms:.1f} response={json.dumps(reset1_response, ensure_ascii=False)}"
        )

        messages.append(
            {
                "role": "user",
                "content": [
                    image_part,
                    {
                        "type": "text",
                        "text": round1_prompt,
                    },
                ],
            }
        )
        round1_request_id = f"{session_id}-round1"
        round1_body = {
            "model": model_name,
            "session_id": session_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        _log(
            f"send round=round1 request_id={round1_request_id} session_id={session_id} "
            f"message_count={len(messages)} image_count=1 prompt_text_chars={len(round1_prompt)}"
        )
        round1_response, round1_headers, round1_latency_ms = _post_json(
            server_url=server_url,
            request_id=round1_request_id,
            body=round1_body,
            timeout_seconds=timeout_seconds,
        )
        round1_file = output_dir / "round1.json"
        _write_json(round1_file, round1_response)
        round1_reply = _assistant_text(round1_response)
        round1_usage = round1_response.get("usage", {})
        _log(
            f"done round=round1 request_id={round1_request_id} "
            f"returned_request_id={round1_headers.get('x-request-id', 'missing')} "
            f"latency_ms={round1_latency_ms:.1f} prompt_tokens={round1_usage.get('prompt_tokens')} "
            f"completion_tokens={round1_usage.get('completion_tokens')} total_tokens={round1_usage.get('total_tokens')}"
        )
        _log(f"assistant round=round1 text={round1_reply!r}")
        _log(f"saved response_file={round1_file}")
        messages.append({"role": "assistant", "content": round1_reply})

        messages.append({"role": "user", "content": round2_prompt})
        round2_request_id = f"{session_id}-round2"
        round2_body = {
            "model": model_name,
            "session_id": session_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        _log(
            f"send round=round2 request_id={round2_request_id} session_id={session_id} "
            f"message_count={len(messages)} image_count=1 "
            f"prompt_text_chars={len(round1_prompt) + len(round2_prompt) + len(round1_reply)}"
        )
        round2_response, round2_headers, round2_latency_ms = _post_json(
            server_url=server_url,
            request_id=round2_request_id,
            body=round2_body,
            timeout_seconds=timeout_seconds,
        )
        round2_file = output_dir / "round2.json"
        _write_json(round2_file, round2_response)
        round2_reply = _assistant_text(round2_response)
        round2_usage = round2_response.get("usage", {})
        _log(
            f"done round=round2 request_id={round2_request_id} "
            f"returned_request_id={round2_headers.get('x-request-id', 'missing')} "
            f"latency_ms={round2_latency_ms:.1f} prompt_tokens={round2_usage.get('prompt_tokens')} "
            f"completion_tokens={round2_usage.get('completion_tokens')} total_tokens={round2_usage.get('total_tokens')}"
        )
        _log(f"assistant round=round2 text={round2_reply!r}")
        _log(f"saved response_file={round2_file}")
        messages.append({"role": "assistant", "content": round2_reply})

        reset2_request_id = f"{new_session_id}-reset-before-new-session"
        _log(f"reset start request_id={reset2_request_id} reset_prefix_cache=true reset_mm_cache=true")
        reset2_response, reset2_headers, reset2_latency_ms = _reset_caches(
            base_server_url=base_server_url,
            request_id=reset2_request_id,
            timeout_seconds=timeout_seconds,
        )
        _log(
            f"reset done request_id={reset2_request_id} returned_request_id={reset2_headers.get('x-request-id', 'missing')} "
            f"latency_ms={reset2_latency_ms:.1f} response={json.dumps(reset2_response, ensure_ascii=False)}"
        )

        new_messages = [
            {
                "role": "user",
                "content": [
                    image_part,
                    {
                        "type": "text",
                        "text": round1_prompt,
                    },
                ],
            },
            {"role": "assistant", "content": round1_reply},
            {"role": "user", "content": round2_prompt},
        ]
        new_session_request_id = f"{new_session_id}-round1"
        new_session_body = {
            "model": model_name,
            "session_id": new_session_id,
            "messages": new_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        _log(
            f"send round=new_session_round1 request_id={new_session_request_id} session_id={new_session_id} "
            f"message_count={len(new_messages)} image_count=1 "
            f"prompt_text_chars={len(round1_prompt) + len(round2_prompt) + len(round1_reply)}"
        )
        new_session_response, new_session_headers, new_session_latency_ms = _post_json(
            server_url=server_url,
            request_id=new_session_request_id,
            body=new_session_body,
            timeout_seconds=timeout_seconds,
        )
        new_session_file = output_dir / "new_session_round1.json"
        _write_json(new_session_file, new_session_response)
        new_session_usage = new_session_response.get("usage", {})
        _log(
            f"done round=new_session_round1 request_id={new_session_request_id} "
            f"returned_request_id={new_session_headers.get('x-request-id', 'missing')} "
            f"latency_ms={new_session_latency_ms:.1f} prompt_tokens={new_session_usage.get('prompt_tokens')} "
            f"completion_tokens={new_session_usage.get('completion_tokens')} total_tokens={new_session_usage.get('total_tokens')}"
        )
        _log(f"assistant round=new_session_round1 text={_assistant_text(new_session_response)!r}")
        _log(f"saved response_file={new_session_file}")
        _log(
            "inspect server logs for the reset and inference request_ids above; "
            "round2 should reflect APC reuse inside one conversation, while new_session_round1 now uses the same effective history length after explicit prefix/mm cache reset. "
            "Because this script uses a very long text prefix and very short decode, latency differences are more indicative of prefix-cache reuse."
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        _log(f"http_error status={exc.code} reason={exc.reason} body={detail}")
        return 1
    except urllib.error.URLError as exc:
        _log(f"url_error reason={exc.reason}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
