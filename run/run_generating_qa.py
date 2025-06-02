import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from source.evaluator.utils import (
    load_llm,
    get_all_nodes_from_qdrant,
    run_generate_question_context_pairs
)
from source.settings import setting  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sinh câu hỏi từ Qdrant context")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Đường dẫn file JSON output để lưu QA dataset"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=2,
        help="Số câu hỏi sinh ra cho mỗi đoạn văn"
    )

    args = parser.parse_args()

    nodes = get_all_nodes_from_qdrant()
    llm = load_llm(model_name=setting.model_name)

    run_generate_question_context_pairs(
        nodes,
        llm,
        args.num_questions,
        args.output_path,
    )

    ## Ensure to write with encoding utf-8
    import json
    with open(args.output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

