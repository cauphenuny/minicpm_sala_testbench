SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/meta.sh

uv run $SCRIPT_DIR/../RULER/scripts/data/prepare.py \
    --save_dir ${data_path} \
    --benchmark synthetic \
    --task cwe \
    --tokenizer_path ${tokenizer_path} \
    --tokenizer_type hf \
    --max_seq_length 16384 \
    --model_template_type minicpm3-chat \
    --num_samples 50