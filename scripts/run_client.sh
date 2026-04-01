#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_SERVER_URL="http://127.0.0.1:8972"
DEFAULT_MODEL_NAME="Qwen3.5-27B"
DEFAULT_REQUEST_MODE="path"
DEFAULT_IMAGE_PATH="/home/user/g00806422/embodiment/vlm_server/tmp/1.png"
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

export SERVER_URL
export MODEL_NAME
export REQUEST_MODE
export IMAGE_PATH
export PROMPT
export MAX_TOKENS
export TEMPERATURE
export OUTPUT_FILE

mkdir -p "$(dirname "${OUTPUT_FILE}")"

echo "[run_client] server_url=${SERVER_URL}"
echo "[run_client] model_name=${MODEL_NAME}"
echo "[run_client] request_mode=${REQUEST_MODE}"
echo "[run_client] image_path=${IMAGE_PATH}"

curl -fsS "${SERVER_URL}/healthz" >/dev/null
curl -fsS "${SERVER_URL}/readyz" | tee "${ROOT_DIR}/tmp/readyz.json" >/dev/null

REQUEST_JSON="$(python3 "${ROOT_DIR}/scripts/build_client_request.py")"

curl -fsS \
  -X POST "${SERVER_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "${REQUEST_JSON}" | tee "${OUTPUT_FILE}"

echo
echo "[run_client] response saved to ${OUTPUT_FILE}"
