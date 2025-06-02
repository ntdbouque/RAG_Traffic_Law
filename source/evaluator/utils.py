import os
import sys
from icecream import ic
import json
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).parent.parent.parent))

from source.database.qdrant import QdrantVectorDatabase
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from source.settings import setting

load_dotenv(override=True)

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset,
)
from llama_index.core.schema import TextNode

import pandas as pd

QA_GENERATE_PROMPT = """
Thông tin ngữ cảnh được cung cấp bên dưới.

---------------------
{context_str}
---------------------

Dựa trên thông tin ngữ cảnh này, không sử dụng kiến thức ngoài lề,  
hãy tạo ra các câu hỏi phù hợp với đoạn văn bản đã cho.

Bạn là một chuyên gia pháp luật. Nhiệm vụ của bạn là soạn {num_questions_per_chunk} câu hỏi mà người dân có thể đặt ra liên quan đến nội dung trong đoạn văn (thường là một khoản của luật).  

Các câu hỏi cần được diễn đạt rõ ràng, gần gũi, dễ hiểu, và phản ánh những thắc mắc thực tế.

Chỉ tạo câu hỏi dựa trên thông tin trong đoạn văn và bằng tiếng việt.

"""

def load_llm(model_name):
    return OpenAI(
        model=setting.model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        logprobs=None,
        default_headers={},
    )

def get_all_chunks_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    all_chunks = [row['Chunk'] for _, row in df.iterrows()]
    
    return all_chunks

def get_all_nodes_from_qdrant():
    qdrant_client = QdrantVectorDatabase(
            url = setting.qdrant_url
        )
        
    vector_store = QdrantVectorStore(client=qdrant_client.client, collection_name=setting.contextual_rag_collection_name)
    nodes_with_original_content = [TextNode(text=node.metadata['original_content'], id_=node.id_) for node in vector_store.get_nodes()]
    return nodes_with_original_content

def run_generate_question_context_pairs(nodes, llm, num_question_per_chunk, save_path):
    qa_dataset = generate_question_context_pairs(
        nodes, llm=llm, num_questions_per_chunk=2, qa_generate_prompt_tmpl=QA_GENERATE_PROMPT
    )
    qa_dataset.save_json(save_path)
    print('COMPLETE')

def run_sample_qa_eval(qa_dataset_path, retriever, metrics=['hit_rate', 'mrr']):
    qa_dataset = EmbeddingQAFinetuneDataset.from_json(qa_dataset_path)
    retriever_evaluator = RetrieverEvaluator.from_metric_names(
        metrics, retriever=retriever
    )
    
    # try it out on a sample query
    sample_id, sample_query = list(qa_dataset.queries.items())[5]
    sample_expected = qa_dataset.relevant_docs[sample_id]

    eval_result = retriever_evaluator.evaluate(sample_query, sample_expected)
    return eval_results


if __name__ == '__main__':
    # nodes = get_all_nodes_from_qdrant()
    llm = OpenAI(
        model=setting.model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        logprobs=None,
        default_headers={},
    )

    # run_generate_question_context_pairs(nodes=nodes, llm = llm, num_question_per_chunk=2, save_path='./data/retrieval_evaluation/qa_dataset2.json')

    import json
    with open("/workspace/competitions/Sly/Duy_NCKH_2025/data/retrieval_evaluation/qa_dataset2.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Bước 2: Ghi lại file với ensure_ascii=False để giữ tiếng Việt đúng
    with open("/workspace/competitions/Sly/Duy_NCKH_2025/data/retrieval_evaluation/qa_dataset2.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # from source.rag.retrieval import RetrievalPipeline
    # from llama_index.core.evaluation import RetrieverEvaluator

    # retriever = RetrievalPipeline()
    # eval_results = run_sample_qa_eval(qa_dataset_path = '/workspace/competitions/Sly/Duy_NCKH_2025/data/retrieval_evaluation/qa_dataset.json',
    #         retriever=retriever)
    
    # print(eval_results)
    


  
        


