SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $SCRIPT_DIR/meta.sh

uv run $SCRIPT_DIR/../RULER/scripts/eval/evaluate.py --data_dir ${data_path} --benchmark synthetic