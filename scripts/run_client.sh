#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_SERVER_URL="http://127.0.0.1:8000"
DEFAULT_MODEL_NAME="Qwen3.5-27B"
DEFAULT_REQUEST_MODE="path"
DEFAULT_IMAGE_PATH="/tmp/example.jpg"
DEFAULT_PROMPT="请描述这张图片中的主要内容。"
DEFAULT_MAX_TOKENS="256"
DEFAULT_TEMPERATURE="0.1"
DEFAULT_OUTPUT_FILE="${ROOT_DIR}/tmp/client_response.json"

SERVER_URL="${SERVER_URL:-${DEFAULT_SERVER_URL}}"
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
REQUEST_MODE="${REQUEST_MODE:-${DEFAULT_REQUEST_MODE}}"
IMAGE_PATH="${IMAGE_PATH:-${DEFAULT_IMAGE_PATH}}"
PROMPT="${PROMPT:-${DEFAULT_PROMPT}}"
MAX_TOKENS="${MAX_TOKENS:-${DEFAULT_MAX_TOKENS}}"
TEMPERATURE="${TEMPERATURE:-${DEFAULT_TEMPERATURE}}"
OUTPUT_FILE="${OUTPUT_FILE:-${DEFAULT_OUTPUT_FILE}}"

mkdir -p "$(dirname "${OUTPUT_FILE}")"

echo "[run_client] server_url=${SERVER_URL}"
echo "[run_client] model_name=${MODEL_NAME}"
echo "[run_client] request_mode=${REQUEST_MODE}"
echo "[run_client] image_path=${IMAGE_PATH}"

curl -fsS "${SERVER_URL}/healthz" >/dev/null
curl -fsS "${SERVER_URL}/readyz" | tee "${ROOT_DIR}/tmp/readyz.json" >/dev/null

if [[ "${REQUEST_MODE}" == "path" ]]; then
  REQUEST_JSON="$(
    python3 - <<'PY'
import json
import os

payload = {
    "model": os.environ["MODEL_NAME"],
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_path",
                    "image_path": {
                        "path": os.environ["IMAGE_PATH"],
                    },
                },
                {
                    "type": "text",
                    "text": os.environ["PROMPT"],
                },
            ],
        }
    ],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": float(os.environ["TEMPERATURE"]),
}
print(json.dumps(payload, ensure_ascii=False))
PY
  )"
elif [[ "${REQUEST_MODE}" == "base64" ]]; then
  REQUEST_JSON="$(
    python3 - <<'PY'
import base64
import json
import os
from pathlib import Path

image_path = Path(os.environ["IMAGE_PATH"])
if not image_path.is_file():
    raise SystemExit(f"image file does not exist: {image_path}")

encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
payload = {
    "model": os.environ["MODEL_NAME"],
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_base64",
                    "image_base64": {
                        "data": encoded,
                    },
                },
                {
                    "type": "text",
                    "text": os.environ["PROMPT"],
                },
            ],
        }
    ],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": float(os.environ["TEMPERATURE"]),
}
print(json.dumps(payload, ensure_ascii=False))
PY
  )"
else
  echo "unsupported REQUEST_MODE=${REQUEST_MODE}, expected path or base64" >&2
  exit 1
fi

curl -fsS \
  -X POST "${SERVER_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${REQUEST_JSON}" | tee "${OUTPUT_FILE}"

echo
echo "[run_client] response saved to ${OUTPUT_FILE}"
