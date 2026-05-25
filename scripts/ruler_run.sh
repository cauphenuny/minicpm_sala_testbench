SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/meta.sh

uv run $SCRIPT_DIR/../RULER/scripts/pred/call_api.py \
    --data_dir ${data_path} \
    --save_dir ${data_path} \
    --benchmark synthetic \
    --task cwe \
    --server_type sglang \
    --model_name_or_path ${model_path} \
    --temperature 0.0 \
    --top_k 32 \
    --top_p 1.0 \
    --batch_size 1 \
    --server_port 30000