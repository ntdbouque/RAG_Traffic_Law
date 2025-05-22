import os
import sys
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path.cwd()))

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from duy.completed_source.generate_dataset import generate_qa_embedding_pairs, get_nodes_from_collection

load_dotenv(override=True)


if __name__ == '__main__':
    # Init Embedding Model:
    embed_model = OpenAIEmbedding(model='text-embedding-3-large', api_key=os.getenv('OPENAI_API_KEY'))
    Settings.embed_model = embed_model
    llm = OpenAI(model='gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
    Settings.llm = llm
    
    nodes = get_nodes_from_collection('contextual_rag_nckh')

    qa_dataset = generate_qa_embedding_pairs(
        llm = Settings.llm,
        nodes = nodes, 
        num_questions_per_chunk = 5
    )

    os.makedirs('./duy/data', exist_ok=True)
    des_json_path = './duy/data/eval_dataset_4_retrieval.json'
    
    qa_dataset.save_json(des_json_path)
    print('>>> Question Answer Pairs Saved at:', des_json_path)
