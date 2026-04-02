# vlm_server

单机单卡 VLM 推理服务，默认基于 `FastAPI + vLLM`，提供本机 HTTP 接口，也可以在 Python 代码里直接调用内部函数完成对话。

## 准备

```bash
cp .env.example .env
```

常用配置：

- `HOST=127.0.0.1`
- `PORT=8972`
- `MODEL_NAME=Qwen3.5-27B`
- `MODEL_PATH=/home/user/g00806422/data/weight/Qwen3.5-27B`
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

指定单卡 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_server.sh
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
