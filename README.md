# vlm_server

最小骨架版本已经包含：

- `FastAPI` 服务入口
- 基础配置模块
- `GET /healthz`
- `GET /readyz`
- `POST /v1/chat/completions`
- 启动时模型加载
- 全局单例 engine manager
- 本地 `.env` 配置模板
- 预留的 `engine / schemas / services` 目录

## 启动方式

```bash
cp .env.example .env
bash scripts/run_server.sh
```

如果只想在本地做不加载模型的结构调试，可在 `.env` 中设置：

```bash
SKIP_MODEL_LOAD=true
```

## 当前状态

- `P1` 已完成
- `P2` 已完成基础模型加载链路
- `P3` 已完成最小接口形态

## 当前接口说明

- `GET /healthz`: 进程健康检查
- `GET /readyz`: 模型加载状态检查
- `POST /v1/chat/completions`: 最小 OpenAI 风格接口

当前 `chat/completions` 已支持：

- 文本输入
- 本机图片路径输入
- base64 图片输入
- `file://` 和 `data:` 形式的本地图片引用

当前仍不支持：

- 远程 `http://` / `https://` 拉图

## 图片输入示例

### 本机路径

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_path",
          "image_path": {
            "path": "/tmp/example.jpg"
          }
        },
        {
          "type": "text",
          "text": "描述这张图片"
        }
      ]
    }
  ]
}
```

### Base64

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_base64",
          "image_base64": {
            "data": "<base64-string>"
          }
        },
        {
          "type": "text",
          "text": "描述这张图片"
        }
      ]
    }
  ]
}
```

## P5 联调脚本

- 服务启动脚本：[run_server.sh](/Users/rkos/Workspace/vlm_server/scripts/run_server.sh)
- 客户端请求脚本：[run_client.sh](/Users/rkos/Workspace/vlm_server/scripts/run_client.sh)
- 联调说明：[P5.md](/Users/rkos/Workspace/vlm_server/docs/P5.md)
