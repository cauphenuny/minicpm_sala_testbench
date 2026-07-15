#!/bin/bash
# Bug 2: CUDA graph batch size mismatch
# 复现: 启动server + 并发请求导致 batch size 变化时 crash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/share-evpfs/tj/micromamba/envs/minicpm-sala/bin/python"
MODEL_PATH="/share-evpfs/tj/models/MiniCPM-SALA"
PORT=32997

echo "=== Bug 2: CUDA graph batch size mismatch ==="
echo "Log dir: $SCRIPT_DIR"
echo ""

# 启动服务 (后台)
echo "Starting server on port $PORT..."
CUDA_VISIBLE_DEVICES=1 http_proxy="" https_proxy="" no_proxy="127.0.0.1,localhost" \
$PYTHON -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --attention-backend minicpm_flashattn \
  --trust-remote-code \
  --minicpm-dense-as-sparse \
  --port $PORT \
  > "$SCRIPT_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# 等待服务就绪
echo "Waiting for server to be ready..."
for i in $(seq 1 180); do
    if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died during startup. Check server.log"
        exit 1
    fi
    sleep 1
done

# 发送多个并发请求 (不同长度, 完成时间不同, 触发 batch size 变化)
echo "Sending 4 concurrent requests..."
for i in 1 2 3 4; do
    curl -s http://127.0.0.1:$PORT/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$MODEL_PATH\",
        \"messages\": [{\"role\": \"user\", \"content\": \"请写一篇关于人工智能的文章，第${i}段\"}],
        \"temperature\": 0,
        \"max_tokens\": 500
      }" > /dev/null 2>&1 &
done

echo "Waiting for requests to complete or server to crash..."
wait
sleep 5

if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server still alive (unexpected for this bug, may need more requests)"
    kill $SERVER_PID 2>/dev/null
else
    echo "Server crashed (expected)"
fi

wait $SERVER_PID 2>/dev/null

# 保存 client 结果
echo "Concurrent requests sent, server crashed during processing" > "$SCRIPT_DIR/client.log"

echo ""
echo "=== Done. Logs: server.log, client.log ==="
