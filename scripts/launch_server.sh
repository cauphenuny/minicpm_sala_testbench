SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/meta.sh

uv run -m sglang.launch_server \
    --model /home/weight/MiniCPM/job_156565_iter_5000_1m/ \
    --trust-remote-code   \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer  \
    --chunked-prefill-size 8192 \
    --max-running-requests 1 \
    --skip-server-warmup \
    --split-stage1 \
    --port 30000 \