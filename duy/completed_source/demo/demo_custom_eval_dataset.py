import sys
import os
from pathlib import Path
sys.path.append(str(Path.cwd()))
from duy.completed_source.demo.demo_custom_eval_dataset import LegalQuery, LegalQueryEvalDataset

# Tạo một số truy vấn mẫu
sample_queries = {
    "q1": LegalQuery(query="Thủ tục cấp lại giấy phép lái xe bị mất?", intent="procedure", complexity=2),
    "q2": LegalQuery(query="So sánh mức phạt giữa vi phạm tốc độ và vượt đèn đỏ", intent="comparison", complexity=3),
    "q3": LegalQuery(query="Tìm hiểu về quy định nồng độ cồn khi lái xe", intent="learning", complexity=1),
}

# Tạo tập dữ liệu và bổ sung thêm các thành phần cần thiết
dataset = LegalQueryEvalDataset()
dataset.queries = sample_queries
dataset.corpus = {
    "d1": "Theo Điều 58 Luật Giao thông đường bộ...",
    "d2": "Nghị định 100/2019/NĐ-CP quy định về xử phạt vi phạm hành chính trong lĩnh vực giao thông...",
}
dataset.relevant_docs = {
    "q1": ["d1"],
    "q2": ["d1", "d2"],
    "q3": ["d2"]
}
dataset.mode = "text"

# Lưu ra file JSON
dataset.save_json("legal_eval_dataset.json")

# Load lại từ file JSON
loaded_dataset = LegalQueryEvalDataset.from_json("legal_eval_dataset.json")

# In ra truy vấn đã load
for qid, query in loaded_dataset.queries.items():
    print(f"ID: {qid}, Query: {query.query}, Intent: {query.intent}, Complexity: {query.complexity}")

