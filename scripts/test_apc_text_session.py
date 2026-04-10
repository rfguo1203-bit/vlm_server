#!/usr/bin/env python3

from __future__ import annotations

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
    print(f"[apc-text] {timestamp} {message}", flush=True)


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
    reset_mm_cache: bool,
) -> tuple[dict, dict[str, str], float]:
    reset_url = f"{base_server_url}/internal/reset-caches"
    return _post_json(
        server_url=reset_url,
        request_id=request_id,
        body={
            "reset_prefix_cache": True,
            "reset_mm_cache": reset_mm_cache,
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
        "Context block for prefix-caching evaluation. "
        "Keep this content unchanged across requests so the shared prompt prefix stays large. "
        f"Seed text: {seed_text}\n"
    )
    return block * repeat_count


def main() -> int:
    base_server_url = _env("SERVER_URL", "http://127.0.0.1:8972").rstrip("/")
    server_url = f"{base_server_url}/v1/chat/completions"
    model_name = _env("MODEL_NAME", "Qwen3.5-27B")
    max_tokens = int(_env("MAX_TOKENS", "8"))
    temperature = float(_env("TEMPERATURE", "0.0"))
    timeout_seconds = float(_env("CLIENT_TIMEOUT_SECONDS", "180"))
    output_dir = Path(_env("OUTPUT_DIR", "tmp/apc_text"))
    session_id = _env("SESSION_ID", f"apc-text-{uuid.uuid4().hex}")
    new_session_id = _env("NEW_SESSION_ID", f"apc-text-new-{uuid.uuid4().hex}")
    prompt_repeat_count = int(_env("PROMPT_REPEAT_COUNT", "256"))
    long_prefix = _build_long_text_block(
        _env("PREFIX_SEED_TEXT", "Atlas mission log. cache-check."),
        prompt_repeat_count,
    )

    messages: list[dict[str, str]] = []
    first_prompt = _env(
        "ROUND1_PROMPT",
        (
            f"{long_prefix}\n"
            "Question: What is the exact codename stated in the repeated context? "
            "Answer with only the codename."
        ),
    )
    second_prompt = _env(
        "ROUND2_PROMPT",
        (
            f"{long_prefix}\n"
            "Follow-up question: Repeat the codename from the repeated context. "
            "Answer with only the codename."
        ),
    )

    _log(f"server_url={server_url}")
    _log(f"model_name={model_name}")
    _log("mode=prompt_heavy_decode_light")
    _log(f"prompt_repeat_count={prompt_repeat_count}")
    _log(f"max_tokens={max_tokens}")
    _log(f"temperature={temperature}")
    _log(f"session_id={session_id}")
    _log(f"new_session_id={new_session_id}")
    _log(
        "goal=reset caches before a fresh text conversation, send two turns with full history, "
        "then reset caches again before starting a brand-new conversation"
    )
    _log("tip=grep server logs by the printed x-request-id values and compare reset and inference timing")

    try:
        reset1_request_id = f"{session_id}-reset-before-round1"
        _log(f"reset start request_id={reset1_request_id} reset_prefix_cache=true reset_mm_cache=false")
        reset1_response, reset1_headers, reset1_latency_ms = _reset_caches(
            base_server_url=base_server_url,
            request_id=reset1_request_id,
            timeout_seconds=timeout_seconds,
            reset_mm_cache=False,
        )
        _log(
            f"reset done request_id={reset1_request_id} returned_request_id={reset1_headers.get('x-request-id', 'missing')} "
            f"latency_ms={reset1_latency_ms:.1f} response={json.dumps(reset1_response, ensure_ascii=False)}"
        )

        round_specs = [
            ("round1", session_id, first_prompt),
            ("round2", session_id, second_prompt),
        ]

        for round_name, current_session_id, user_prompt in round_specs:
            messages.append({"role": "user", "content": user_prompt})
            request_id = f"{current_session_id}-{round_name}"
            request_body = {
                "model": model_name,
                "session_id": current_session_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            image_count = 0
            _log(
                f"send round={round_name} request_id={request_id} session_id={current_session_id} "
                f"message_count={len(messages)} image_count={image_count} "
                f"prompt_chars={sum(len(item['content']) for item in messages if isinstance(item.get('content'), str))}"
            )
            response_body, response_headers, latency_ms = _post_json(
                server_url=server_url,
                request_id=request_id,
                body=request_body,
                timeout_seconds=timeout_seconds,
            )
            response_file = output_dir / f"{round_name}.json"
            _write_json(response_file, response_body)
            assistant_reply = _assistant_text(response_body)
            usage = response_body.get("usage", {})
            returned_request_id = response_headers.get("x-request-id", "missing")
            _log(
                f"done round={round_name} request_id={request_id} returned_request_id={returned_request_id} "
                f"latency_ms={latency_ms:.1f} prompt_tokens={usage.get('prompt_tokens')} "
                f"completion_tokens={usage.get('completion_tokens')} total_tokens={usage.get('total_tokens')}"
            )
            _log(f"assistant round={round_name} text={assistant_reply!r}")
            _log(f"saved response_file={response_file}")
            messages.append({"role": "assistant", "content": assistant_reply})

        reset2_request_id = f"{new_session_id}-reset-before-new-session"
        _log(f"reset start request_id={reset2_request_id} reset_prefix_cache=true reset_mm_cache=false")
        reset2_response, reset2_headers, reset2_latency_ms = _reset_caches(
            base_server_url=base_server_url,
            request_id=reset2_request_id,
            timeout_seconds=timeout_seconds,
            reset_mm_cache=False,
        )
        _log(
            f"reset done request_id={reset2_request_id} returned_request_id={reset2_headers.get('x-request-id', 'missing')} "
            f"latency_ms={reset2_latency_ms:.1f} response={json.dumps(reset2_response, ensure_ascii=False)}"
        )

        new_messages = [{"role": "user", "content": first_prompt}]
        new_session_request_id = f"{new_session_id}-round1"
        new_session_request_body = {
            "model": model_name,
            "session_id": new_session_id,
            "messages": new_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        _log(
            f"send round=new_session_round1 request_id={new_session_request_id} session_id={new_session_id} "
            f"message_count={len(new_messages)} image_count=0 "
            f"prompt_chars={sum(len(item['content']) for item in new_messages if isinstance(item.get('content'), str))}"
        )
        new_session_response, new_session_headers, new_session_latency_ms = _post_json(
            server_url=server_url,
            request_id=new_session_request_id,
            body=new_session_request_body,
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
            "round2 should benefit from APC reuse within the same conversation, while new_session_round1 runs after an explicit cache reset. "
            "Because this script uses a very long prompt and very short decode, latency differences are more indicative of prefix-cache reuse."
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
