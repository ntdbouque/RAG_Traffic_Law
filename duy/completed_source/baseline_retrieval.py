import os

from qdrant_client import AsyncQdrantClient, models
from qdrant_client import QdrantClient

from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import (
    Settings,
    Document,
    QueryBundle,
    StorageContext,
    VectorStoreIndex,
)

import nest_asyncio
nest_asyncio.apply()

# Init Embedding Model:
embed_model = OpenAIEmbedding(model='text-embedding-3-large', api_key=os.getenv('OPENAI_API_KEY'))
Settings.embed_model = embed_model
llm = OpenAI(model='gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
Settings.llm = llm

def get_base_retriever(collection_name='contextual_rag_nckh'):
    '''
    Get an Asynchronous Qdrant Retriever
    '''
    # Dùng async client
    async_client = AsyncQdrantClient(url="http://localhost:6333")
    qdrant_client = QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(aclient=async_client, client=qdrant_client, collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Phải bật use_async
    qdrant_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, 
        storage_context=storage_context,
        use_async=True
    )

    retriever = qdrant_index.as_retriever(similarity_top_k=50)
    return retriever

def get_gpt_reranker(llm=None, top_n = 10):
    '''
    Get a simple GPT Reranker
    '''
    llm = llm or Settings.llm 
    return RankGPTRerank(
            llm = Settings.llm,
            top_n=10,
    )
    
def retrieve_top_k_nodes(query):
    '''
    An example function to retrieve top k nodes
    '''
    # test:
    retriever = get_base_retriever()
    retrieved_nodes = retriever.retrieve(query)
    return retrieved_nodes