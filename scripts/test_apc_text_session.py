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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _assistant_text(response: dict) -> str:
    return response["choices"][0]["message"]["content"]


def main() -> int:
    server_url = _env("SERVER_URL", "http://127.0.0.1:8972/v1/chat/completions")
    model_name = _env("MODEL_NAME", "Qwen3.5-27B")
    max_tokens = int(_env("MAX_TOKENS", "256"))
    temperature = float(_env("TEMPERATURE", "0.1"))
    timeout_seconds = float(_env("CLIENT_TIMEOUT_SECONDS", "180"))
    output_dir = Path(_env("OUTPUT_DIR", "tmp/apc_text"))
    session_id = _env("SESSION_ID", f"apc-text-{uuid.uuid4().hex}")
    isolate_session_id = _env("ISOLATE_SESSION_ID", f"apc-text-isolate-{uuid.uuid4().hex}")

    messages: list[dict[str, str]] = []
    first_prompt = _env(
        "ROUND1_PROMPT",
        "Please remember that my project codename is Atlas and answer briefly.",
    )
    second_prompt = _env(
        "ROUND2_PROMPT",
        "What is the project codename I just told you?",
    )

    _log(f"server_url={server_url}")
    _log(f"model_name={model_name}")
    _log(f"session_id={session_id}")
    _log(f"isolate_session_id={isolate_session_id}")
    _log(
        "goal=send two turns with the same session_id and full message history, "
        "then replay the same history with a different session_id for isolation checking"
    )
    _log("tip=grep server logs by the printed x-request-id values and compare queue/inference timing")

    try:
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
                f"message_count={len(messages)} image_count={image_count}"
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

        isolate_request_id = f"{isolate_session_id}-round2-replay"
        isolate_request_body = {
            "model": model_name,
            "session_id": isolate_session_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        _log(
            f"send round=round2_replay request_id={isolate_request_id} session_id={isolate_session_id} "
            f"message_count={len(messages)} image_count=0"
        )
        isolate_response, isolate_headers, isolate_latency_ms = _post_json(
            server_url=server_url,
            request_id=isolate_request_id,
            body=isolate_request_body,
            timeout_seconds=timeout_seconds,
        )
        isolate_file = output_dir / "round2_replay_new_session.json"
        _write_json(isolate_file, isolate_response)
        isolate_usage = isolate_response.get("usage", {})
        _log(
            f"done round=round2_replay request_id={isolate_request_id} "
            f"returned_request_id={isolate_headers.get('x-request-id', 'missing')} "
            f"latency_ms={isolate_latency_ms:.1f} prompt_tokens={isolate_usage.get('prompt_tokens')} "
            f"completion_tokens={isolate_usage.get('completion_tokens')} total_tokens={isolate_usage.get('total_tokens')}"
        )
        _log(f"assistant round=round2_replay text={_assistant_text(isolate_response)!r}")
        _log(f"saved response_file={isolate_file}")
        _log(
            "inspect server logs for the three request_ids above; "
            "round2 should be the strongest APC-hit candidate, while round2_replay uses a different session_id"
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
