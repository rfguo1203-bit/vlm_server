# vlm_server

单机多卡 VLM 推理服务，默认基于 `FastAPI + vLLM`，当前默认按两卡启动，提供本机 HTTP 接口，也可以在 Python 代码里直接调用内部函数完成对话。

## 准备

```bash
cp .env.example .env
```

常用配置：

- `HOST=127.0.0.1`
- `PORT=8972`
- `MODEL_NAME=gemma-4-E4B-it`
- `ADDITIONAL_MODEL_PATHS_JSON={}`
- `TENSOR_PARALLEL_SIZE=2`
- `ENABLE_PREFIX_CACHING=true`
- `SESSION_CACHE_SECRET=please-change-this-in-prod`
- `INFERENCE_CONCURRENCY=1`
- `REQUEST_TIMEOUT_SECONDS=120`
- `MAX_OUTPUT_TOKENS_LIMIT=10240`

如果只想做结构调试、不加载真实模型：

```bash
SKIP_MODEL_LOAD=true
```

## 启动服务

```bash
conda activate vllm
bash scripts/run_server.sh
```

默认等价于两卡启动，例如：

```bash
CUDA_VISIBLE_DEVICES=4,5 TENSOR_PARALLEL_SIZE=2 bash scripts/run_server.sh
```

指定别的两张卡：

```bash
CUDA_VISIBLE_DEVICES=6,7 TENSOR_PARALLEL_SIZE=2 bash scripts/run_server.sh
```

如果你要临时切回单卡：

```bash
CUDA_VISIBLE_DEVICES=0 TENSOR_PARALLEL_SIZE=1 bash scripts/run_server.sh
```

## HTTP 调用

健康检查：

```bash
curl http://127.0.0.1:8972/healthz
curl http://127.0.0.1:8972/readyz
curl http://127.0.0.1:8972/internal/engine-status
```

路径图片请求：

```bash
IMAGE_PATH=/path/to/example.png bash scripts/run_client.sh
```

Base64 图片请求：

```bash
REQUEST_MODE=base64 IMAGE_PATH=/path/to/example.png bash scripts/run_client.sh
```

也可以直接发 `POST /v1/chat/completions`：

```bash
curl -X POST http://127.0.0.1:8972/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it",
    "session_id": "demo-session-001",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_path",
            "image_path": {
              "path": "/path/to/example.png"
            }
          },
          {
            "type": "text",
            "text": "请描述这张图片中的主要内容。"
          }
        ]
      }
    ],
    "max_tokens": 256,
    "temperature": 0.1
  }'
```

### 模型切换

当前服务支持以下模型名：

- `gemma-4-E4B-it`（默认）
- `Qwen3.5-27B`
- `gemma-4-26B-A4B-it`

请求中通过 `model` 字段切换，服务会按需加载目标模型。内置映射如下：

- `gemma-4-E4B-it`: `/home/user/g00806422/data/weight/gemma-4-26B-A4B-it`
- `Qwen3.5-27B`: `/home/user/g00806422/data/weight/Qwen3.5-27B`
- `gemma-4-26B-A4B-it`: `/home/user/g00806422/data/weight/gemma-4-26B-A4B-it`

如果需要扩展更多模型，可在 `.env` 里设置 `ADDITIONAL_MODEL_PATHS_JSON`，例如：

```bash
ADDITIONAL_MODEL_PATHS_JSON='{"my-model":"/path/to/my-model"}'
```

支持的图片输入：

- `image_path`
- `image_base64`
- `image_url` 中的 `file://...`
- `image_url` 中的 `data:...`

不支持：

- 远程 `http://` / `https://` 图片

## 手动 Reset 的多轮对话

当前工程的多轮对话推荐按“手动 reset cache”模式使用。

原因：

- 当前服务端不会保存会话历史
- 客户端每轮都需要发送完整 `messages`
- 当前运行时如果不支持 `cache_salt`，就不能按 `session_id` 做缓存隔离
- 所以新会话开始前，需要显式清理 prefix cache；图文会话还要一起清理 multimodal cache

### 调用约定

- 同一会话内：`session_id` 保持不变
- 同一会话内：每轮请求继续发送完整历史 `messages`
- 新会话开始前：先调用 `/internal/reset-caches`
- 新会话开始后：再使用一个新的 `session_id`

`session_id` 在当前实现里主要用于请求链路标识和未来兼容，不应该把它理解成“服务端一定按 session 隔离了 KV cache”。

### Reset 接口

文本会话建议这样 reset：

```bash
curl -X POST http://127.0.0.1:8972/internal/reset-caches \
  -H "Content-Type: application/json" \
  -d '{
    "reset_prefix_cache": true,
    "reset_mm_cache": false,
    "reset_running_requests": false
  }'
```

图文会话建议这样 reset：

```bash
curl -X POST http://127.0.0.1:8972/internal/reset-caches \
  -H "Content-Type: application/json" \
  -d '{
    "reset_prefix_cache": true,
    "reset_mm_cache": true,
    "reset_running_requests": false
  }'
```

字段含义：

- `reset_prefix_cache`: 清理 prefix cache
- `reset_mm_cache`: 清理多模态缓存
- `reset_running_requests`: 是否连正在运行中的请求一起处理，默认保持 `false`

### `urllib.request` 多轮示例

下面是一个文本多轮对话示例，按当前推荐方式：

```python
import json
import urllib.request
import uuid

base_url = "http://127.0.0.1:8972"
chat_url = f"{base_url}/v1/chat/completions"
reset_url = f"{base_url}/internal/reset-caches"

session_id = str(uuid.uuid4())
messages = []

reset_req = urllib.request.Request(
    reset_url,
    data=json.dumps(
        {
            "reset_prefix_cache": True,
            "reset_mm_cache": False,
            "reset_running_requests": False,
        },
        ensure_ascii=False,
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(reset_req) as resp:
    print("reset:", resp.read().decode("utf-8"))

for user_text in ["你好，请先记住我叫小王。", "我刚才叫什么名字？"]:
    messages.append({"role": "user", "content": user_text})
    body = {
        "model": "Qwen3.5-27B",
        "session_id": session_id,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        chat_url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    assistant_text = result["choices"][0]["message"]["content"]
    print(assistant_text)
    messages.append({"role": "assistant", "content": assistant_text})
```

### 推荐用法

- 文本多轮：
  - 新会话前 reset `prefix cache`
  - 同一会话内不断追加 `messages`
- 图文多轮：
  - 新会话前 reset `prefix cache + mm cache`
  - 首轮带图，后续继续发送完整历史
- 如果服务端日志里出现 `cache_salt_not_supported`
  - 说明当前运行时已经自动退回到“无 session 隔离盐”的 APC 模式
  - 这时更应该严格遵守“新会话前先 reset”的调用方式

### APC 验证脚本

- 文本 APC 验证脚本：[scripts/test_apc_text_session.py](/Users/rkos/Workspace/vlm_server/scripts/test_apc_text_session.py)
- 图文 APC 验证脚本：[scripts/test_apc_multimodal_session.py](/Users/rkos/Workspace/vlm_server/scripts/test_apc_multimodal_session.py)

这两个脚本默认已经切到“长 prompt、短 decode”模式，方便观察 prefix cache 效果：

- 默认 `MAX_TOKENS=8`
- 默认 `TEMPERATURE=0.0`
- 默认会自动构造大量重复前缀文本

如果你想进一步放大 APC 命中前后的差异，可以继续增大：

- `PROMPT_REPEAT_COUNT`

例如：

```bash
PROMPT_REPEAT_COUNT=512 MAX_TOKENS=8 python scripts/test_apc_text_session.py
```

```bash
IMAGE_PATH=/path/to/example.png PROMPT_REPEAT_COUNT=384 MAX_TOKENS=8 python scripts/test_apc_multimodal_session.py
```

## Python 代码直接调用

如果你不想走 HTTP，而是想在同一个 Python 进程里直接复用仓库里的函数，可以直接调用 [`create_chat_completion`](/Users/rkos/Workspace/vlm_server/app/services/chat_service.py)。

最小示例：

```python
import asyncio

from app.core.config import get_settings
from app.engine.manager import initialize_engine_manager
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatImagePathContentPart,
    ChatMessage,
    ChatTextContentPart,
    ImagePathPayload,
)
from app.services.chat_service import create_chat_completion


async def main() -> None:
    settings = get_settings()
    engine_manager = initialize_engine_manager(settings)

    if engine_manager.engine is None and not engine_manager.status.loaded:
        engine_manager.load()

    request = ChatCompletionRequest(
        model=settings.model_name,
        messages=[
            ChatMessage(
                role="user",
                content=[
                    ChatImagePathContentPart(
                        type="image_path",
                        image_path=ImagePathPayload(path="/path/to/example.png"),
                    ),
                    ChatTextContentPart(
                        type="text",
                        text="请描述这张图片中的主要内容。",
                    ),
                ],
            )
        ],
        max_tokens=256,
        temperature=0.1,
    )

    response = await create_chat_completion(
        request=request,
        engine_manager=engine_manager,
        settings=settings,
        request_id="local-call-1",
    )
    print(response.choices[0].message.content)


asyncio.run(main())
```

说明：

- `get_settings()` 负责读取 `.env`
- `initialize_engine_manager(settings)` 负责拿到全局单例 engine manager
- 第一次调用前需要确保 `engine_manager.load()` 已执行
- `create_chat_completion(...)` 是异步函数，需要在 `asyncio` 环境里调用
- 直接函数调用和 HTTP 接口走的是同一套图片处理、超时、并发保护和错误逻辑

### Python 多轮示例

如果你在同一个 Python 进程里做多轮对话，也建议沿用和 HTTP 一样的调用约定：

- 新会话前先 reset cache
- 同一会话内维护完整 `messages`
- 每轮把完整历史继续传给 `create_chat_completion(...)`

文本多轮最小示例：

```python
import asyncio
import uuid

from app.core.config import get_settings
from app.engine.manager import initialize_engine_manager
from app.schemas.chat import ChatCompletionRequest, ChatMessage
from app.services.chat_service import create_chat_completion, reset_runtime_caches


async def main() -> None:
    settings = get_settings()
    engine_manager = initialize_engine_manager(settings)

    if engine_manager.engine is None and not engine_manager.status.loaded:
        engine_manager.load()

    await reset_runtime_caches(
        engine_manager=engine_manager,
        request_id="local-reset-1",
        reset_prefix_cache=True,
        reset_mm_cache=False,
        reset_running_requests=False,
    )

    session_id = str(uuid.uuid4())
    messages: list[ChatMessage] = []

    for idx, user_text in enumerate(
        [
            "你好，请先记住我叫小王。",
            "我刚才叫什么名字？",
        ],
        start=1,
    ):
        messages.append(ChatMessage(role="user", content=user_text))
        request = ChatCompletionRequest(
            model=settings.model_name,
            session_id=session_id,
            messages=messages,
            max_tokens=256,
            temperature=0.1,
        )
        response = await create_chat_completion(
            request=request,
            engine_manager=engine_manager,
            settings=settings,
            request_id=f"local-chat-{idx}",
        )
        assistant_text = response.choices[0].message.content
        print(assistant_text)
        messages.append(ChatMessage(role="assistant", content=assistant_text))


asyncio.run(main())
```

图文多轮时，把 reset 改成：

```python
await reset_runtime_caches(
    engine_manager=engine_manager,
    request_id="local-reset-mm-1",
    reset_prefix_cache=True,
    reset_mm_cache=True,
    reset_running_requests=False,
)
```

## 相关脚本

- 服务启动脚本：[scripts/run_server.sh](/Users/rkos/Workspace/vlm_server/scripts/run_server.sh)
- 客户端请求脚本：[scripts/run_client.sh](/Users/rkos/Workspace/vlm_server/scripts/run_client.sh)
