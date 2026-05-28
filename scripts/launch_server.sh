SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/meta.sh

# uv run -m sglang.launch_server \
#     --model $model_path \
#     --trust-remote-code   \
#     --disable-radix-cache \
#     --attention-backend minicpm_flashinfer \
#     --chunked-prefill-size 8192 \
#     --max-running-requests 1 \
#     --skip-server-warmup \
#     --split-stage1 \
#     --mem-fraction-static 0.6 \
#     --port 30000


uv run -m sglang.launch_server \
    --model $model_path \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashattn \
    --chunked-prefill-size 8192 \
    --max-running-requests 1 \
    --max-prefill-tokens 20480 \
    --skip-server-warmup \
    --disable-piecewise-cuda-graph \
    --port 30000
