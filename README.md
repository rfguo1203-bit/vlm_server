# vlm_server

单机多卡 VLM 推理服务，默认基于 `FastAPI + vLLM`，当前默认按两卡启动，提供本机 HTTP 接口，也可以在 Python 代码里直接调用内部函数完成对话。

## 准备

```bash
cp .env.example .env
```

常用配置：

- `HOST=127.0.0.1`
- `PORT=8972`
- `MODEL_NAME=Qwen3.5-27B`
- `MODEL_PATH=/home/user/g00806422/data/weight/Qwen3.5-27B`
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
    "model": "Qwen3.5-27B",
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

多轮对话时，客户端每轮都带完整历史，并保持同一个 `session_id`。服务端不会保存历史，只会把 `session_id` 映射成请求级缓存隔离键，配合 vLLM 的 prefix caching 复用相同前缀的 KV cache。

最小约定：

- 同一会话：`session_id` 不变
- 新开会话：换一个新的 `session_id`
- 每轮请求：`messages = 历史消息 + 本轮新增消息`
- 收到回复后：把 assistant 回复追加回本地 `messages`

一个简化的 `urllib.request` 多轮示例：

```python
import json
import urllib.request
import uuid

server_url = "http://127.0.0.1:8972/v1/chat/completions"
session_id = str(uuid.uuid4())
messages = []

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
        server_url,
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

说明：

- 不传 `session_id` 时，接口仍可正常使用，但不会提供“同会话隔离的 KV cache 复用保证”
- 传了 `session_id` 时，要求当前 vLLM 版本支持 APC 和 `cache_salt`
- 图片多轮场景也可以带 `session_id`，但图片前缀是否稳定命中缓存依赖底层模型与 vLLM 版本，需要单独压测验证

支持的图片输入：

- `image_path`
- `image_base64`
- `image_url` 中的 `file://...`
- `image_url` 中的 `data:...`

不支持：

- 远程 `http://` / `https://` 图片

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

## 相关脚本

- 服务启动脚本：[scripts/run_server.sh](/Users/rkos/Workspace/vlm_server/scripts/run_server.sh)
- 客户端请求脚本：[scripts/run_client.sh](/Users/rkos/Workspace/vlm_server/scripts/run_client.sh)
