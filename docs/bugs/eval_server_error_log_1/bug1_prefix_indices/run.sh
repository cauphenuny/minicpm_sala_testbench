#!/bin/bash
# Bug 1: Missing sparse_k1 prefix indices
# 复现: 启动server + 发送长序列请求 (>8192 tokens) 触发 chunked prefill crash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/share-evpfs/tj/micromamba/envs/minicpm-sala/bin/python"
MODEL_PATH="/share-evpfs/tj/models/MiniCPM-SALA"
PORT=32997

echo "=== Bug 1: Missing sparse_k1 prefix indices ==="
echo "Log dir: $SCRIPT_DIR"
echo ""

# 启动服务 (后台)
echo "Starting server on port $PORT..."
CUDA_VISIBLE_DEVICES=1 http_proxy="" https_proxy="" no_proxy="127.0.0.1,localhost" \
$PYTHON -m sglang.launch_server \
  --model-path $MODEL_PATH \
  --attention-backend minicpm_flashattn \
  --minicpm-dense-as-sparse \
  --trust-remote-code \
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

# 发送长序列请求触发 bug
echo "Sending long sequence request (~15000 tokens)..."
LONG_TEXT=$(python3 -c "print('人工智能正在改变世界。深度学习使机器能够从数据中学习。自然语言处理让计算机理解人类语言。计算机视觉让机器看懂图像。' * 280 + '请总结上文主题：')")
curl -s http://127.0.0.1:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL_PATH\",
    \"messages\": [{\"role\": \"user\", \"content\": \"$LONG_TEXT\"}],
    \"temperature\": 0,
    \"max_tokens\": 30
  }" 2>&1 | tee "$SCRIPT_DIR/client.log"

echo ""

# 等待几秒看 server 是否 crash
sleep 5
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server still alive (unexpected)"
    kill $SERVER_PID 2>/dev/null
else
    echo "Server crashed (expected)"
fi

wait $SERVER_PID 2>/dev/null
echo ""
echo "=== Done. Logs: server.log, client.log ==="
