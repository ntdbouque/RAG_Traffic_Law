import os
import sys
from pathlib import Path
import uuid
from tqdm import tqdm
from typing import List
import pandas as pd
import re   
import json
from dotenv import load_dotenv
sys.path.append(str(Path.cwd()))

from llama_index.core import Document 
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import MetadataMode, TextNode

from qdrant_client import QdrantClient

from duy.completed_source.custom_dataset import LegalQueryEvalDataset, LegalQuery

load_dotenv(override=True)


QA_PAIRS_GENERATING_PROMPT = """\
Thông tin ngữ cảnh nằm bên dưới.

---------------------
{context_str}
---------------------

Chỉ dựa vào thông tin ngữ cảnh ở trên (không sử dụng kiến thức bên ngoài),
hãy tạo một tập gồm {num_questions_per_chunk} câu truy vấn đa dạng của người dùng liên quan đến luật giao thông.
Các truy vấn này sẽ được sử dụng để đánh giá một hệ thống truy vấn thông tin pháp luật.

Bạn đang đóng vai trò là một chuyên gia pháp lý và người thiết kế bài giảng. Nhiệm vụ của bạn là tạo ra các câu hỏi đáp ứng các tiêu chí sau:

1. Bao gồm nhiều **mục đích truy vấn** khác nhau của người dùng, như:
   - **Tra cứu quy định**: Người dùng muốn biết quy định hoặc mức phạt cụ thể. (Ví dụ: “Mức phạt khi chạy quá tốc độ là bao nhiêu?”)
   - **Tình huống thực tế**: Người dùng mô tả một tình huống cá nhân và hỏi liệu điều đó có hợp pháp hoặc bị xử phạt không.
   - **Thủ tục hành chính**: Người dùng hỏi về giấy tờ, bước thực hiện, điều kiện để được cấp phép, nộp phạt, v.v.
   - **So sánh / ra quyết định**: Người dùng so sánh các lựa chọn hoặc quy định để đưa ra quyết định. (Ví dụ: “Hành vi nào bị phạt nặng hơn…”)
   - **Học tập pháp luật**: Người dùng đang nghiên cứu hoặc học luật, muốn hiểu cấu trúc, phạm vi hoặc định nghĩa.

2. Chỉ tạo câu hỏi ở **mức độ phức tạp 1 và 2**:
   - **Cấp độ 1 (Đơn giản)**: Câu hỏi tra cứu trực tiếp, chỉ liên quan đến một quy định cụ thể.
   - **Cấp độ 2 (Trung bình)**: Câu hỏi tình huống hoặc so sánh, yêu cầu kết hợp hoặc suy luận từ nhiều thông tin trong một đoạn luật.

Hướng dẫn:
- Chỉ sử dụng nội dung pháp lý được cung cấp. Không dùng kiến thức từ bên ngoài.
- Mỗi câu hỏi cần có thể trả lời từ ngữ cảnh, nhưng không được trích dẫn nguyên văn.
- Kết quả trả về dưới dạng **danh sách JSON** gồm các đối tượng. Mỗi đối tượng phải bao gồm:
  - "query": câu hỏi của người dùng viết bằng ngôn ngữ tự nhiên
  - "intent": một trong các giá trị ["lookup", "situation", "procedure", "comparison", "learning"]
  - "complexity": số nguyên từ 1 đến 2

Ví dụ định dạng đầu ra:
[
  {{
    "query": "Mức phạt khi điều khiển xe máy không đội mũ bảo hiểm là bao nhiêu?",
    "intent": "lookup",
    "complexity": 1
  }},
  {{
    "query": "Tôi điều khiển xe tay ga mà quên mang theo bằng lái, như vậy có bị phạt không?",
    "intent": "situation",
    "complexity": 2
  }}
]
"""

# get all node from qdrant database:
def get_nodes_from_collection(collection_name: str):
    '''
    Get all nodes from a qdrant collection
    Args:
        collection_name: a qdrant collection name
    '''
    client = QdrantClient(url="http://localhost:6333")

    qdrant_nodes, _ = client.scroll(
        collection_name="contextual_rag_nckh",
        limit=1489
    )
    nodes = []
    for node in qdrant_nodes:
        nodes.append(TextNode(text=node.payload['text'], id_=node.id))
    return nodes

nodes = get_nodes_from_collection('contextual_rag_nckh')

def generate_qa_embedding_pairs(
    nodes: List[TextNode],
    llm = None,
    qa_generate_prompt_tmpl: str = QA_PAIRS_GENERATING_PROMPT,
    num_questions_per_chunk: int = 5,
) -> LegalQueryEvalDataset:
    """Generate examples given a set of nodes and return as LegalQueryEvalDataset."""
    llm = llm or Settings.llm  # Sử dụng LLM mặc định nếu không có LLM cụ thể
    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    queries = {}
    relevant_docs = {}

    for node_id, text in tqdm(node_dict.items()):
        # Tạo câu hỏi từ node text
        query = qa_generate_prompt_tmpl.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )
        response = llm.complete(query)

        result = str(response).strip().split("\n")

        # Xử lý chuỗi JSON trả về từ LLM
        json_string = ''.join(result[1:-1])  # Loại bỏ phần đầu và cuối không cần thiết
        questions = json.loads(json_string)

        # Tạo các câu hỏi và lưu vào queries, relevant_docs
        for question in questions:
            question_id = str(uuid.uuid4())

            # Tạo đối tượng LegalQuery từ câu hỏi
            legal_query = LegalQuery(
                query=question["query"],
                intent=question["intent"],
                complexity=question["complexity"],
            )
            
            # Lưu vào queries và relevant_docs
            queries[question_id] = legal_query
            relevant_docs[question_id] = [node_id]  # Liên kết câu hỏi với node_id
        # break # only for test

    # Trả về đối tượng LegalQueryEvalDataset
    return LegalQueryEvalDataset(
        queries=queries,
        corpus=node_dict,  # Lưu trữ corpus là nội dung của các node
        relevant_docs=relevant_docs,
        mode="text"
    )

if __name__ == '__main__':
    print('hello world')