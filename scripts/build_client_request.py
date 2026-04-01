#!/usr/bin/env python3

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path


def _get_required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def _build_path_payload() -> dict:
    return {
        "type": "image_path",
        "image_path": {
            "path": _get_required_env("IMAGE_PATH"),
        },
    }


def _build_base64_payload() -> dict:
    image_path = Path(_get_required_env("IMAGE_PATH"))
    if not image_path.is_file():
        raise SystemExit(f"image file does not exist: {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "type": "image_base64",
        "image_base64": {
            "data": encoded,
        },
    }


def build_request_json() -> str:
    request_mode = _get_required_env("REQUEST_MODE")
    if request_mode == "path":
        image_payload = _build_path_payload()
    elif request_mode == "base64":
        image_payload = _build_base64_payload()
    else:
        raise SystemExit(f"unsupported REQUEST_MODE={request_mode}, expected path or base64")

    payload = {
        "model": _get_required_env("MODEL_NAME"),
        "messages": [
            {
                "role": "user",
                "content": [
                    image_payload,
                    {
                        "type": "text",
                        "text": _get_required_env("PROMPT"),
                    },
                ],
            }
        ],
        "max_tokens": int(_get_required_env("MAX_TOKENS")),
        "temperature": float(_get_required_env("TEMPERATURE")),
    }
    return json.dumps(payload, ensure_ascii=False)


def main() -> int:
    sys.stdout.write(build_request_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
