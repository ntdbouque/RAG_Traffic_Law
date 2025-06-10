import os
import sys
from icecream import ic
import json
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).parent.parent.parent))

from source.settings import Settings as ConfigSettings#, setting as config_setting
from source.database.elastic import ElasticSearch
from source.database.qdrant import QdrantVectorDatabase
from source.logging.log_retrieval import log_retrieval

from qdrant_client import QdrantClient
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbeddingMode
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever
)
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, Node, TextNode
from llama_index.core import (
    Settings,
    Document,
    QueryBundle,
    StorageContext,
    VectorStoreIndex,
)
from typing import Sequence

load_dotenv(override=True)

class RetrievalPipeline(BaseRetriever):
    '''
    Contextual Retrieval-Augmented Generation (RAG) class to search for relevant article from Vietnamese Traffic Law
    '''
    class Config:
        extra = 'allow'

    def __init__(self, k=150):
        super().__init__()
        setting = ConfigSettings()
        #ic(setting)
        self.setting = setting

        self.llm = OpenAI(
                model=self.setting.model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                logprobs=None,
                default_headers={},
            )
        self.embed_model = OpenAIEmbedding(model=setting.embed_model, 
                                            api_key=os.getenv('OPENAI_API_KEY'),
                                            mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE)

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.k = k
        
        self.es_client = ElasticSearch(
            url = self.setting.elastic_search_url,
            index_name = self.setting.elastic_search_index_name
        )
        
        self.qdrant_client = QdrantVectorDatabase(
            url = self.setting.qdrant_url
        )
        
        vector_store = QdrantVectorStore(client=self.qdrant_client.client, collection_name=self.setting.contextual_rag_collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        self.retriever = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, 
            storage_context=storage_context,
            use_async=True).as_retriever(similarity_top_k=100)
      
        
        self.reranker_gpt = RankGPTRerank(
            llm=Settings.llm,
            top_n=10,
            verbose=False,
        )

        Settings.node_postprocessors = [self.reranker_gpt]        
    
    def contextual_search(self, query):
        '''
        Search the query with the contextual RAG (Qdrant)
        
        Args:
            query (str): The query string
            k (int): The number of documents to return  
        Return:
            list: The list of document id from the search result
        ''' 
        
        #ic(query, k)
        semantic_results: Response = self.retriever.retrieve(query)
        
        return semantic_results
    
    def bm25_search(self, query, k: int = 50) -> str:
        '''
        Search the query with the keyword search (ElasticSearch)
        
        Args:
            query (str): The query string
            k (int): The number of documents to return  
        '''
        
        #ic(query, k)
        bm25_results = self.es_client.search(query, k)
        
        return bm25_results
    
    def gpt_reranking(
        self, 
        nodes:list[NodeWithScore],
        query: QueryBundle) -> list[NodeWithScore]:
        '''
        Reranking the search results with the GPT Reranker
        '''
        return self.reranker_gpt.postprocess_nodes(nodes, query)

    def map_reduce_postprocess(
        self, 
        nodes: list[NodeWithScore],
        query: QueryBundle) -> list[NodeWithScore]:
        '''
        Context compression with map reduce postprocessing
        '''
        return self.map_reduce_postprocessor.postprocess_nodes(nodes, query)

    def combine_results(
        self,
        semantic_results: list[NodeWithScore],
        bm25_results: list[NodeWithScore]) -> list[NodeWithScore]:
        '''
        Combine the search results from semantic search and bm25 search
        '''

        semantic_doc_id = [
            node.metadata["article_uuid"] for node in semantic_results
        ]
        bm25_doc_id = [result.doc_id for result in bm25_results]
        
        def get_content_by_doc_id(doc_id: str):
            for node in semantic_results:
                if node.metadata["article_uuid"] == doc_id:
                    return node.text
            return ""
        
        semantic_weight = self.setting.semantic_weight
        bm25_weight = self.setting.bm25_weight
        
        combined_nodes: list[NodeWithScore] = []
        combined_ids = list(set(semantic_doc_id + bm25_doc_id))
        
        #ic(len(combined_ids))
        
        semantic_count = 0
        bm25_count = 0
        both_count = 0
        
        for id in combined_ids:
            score = 0
            content = ''
            if id in semantic_doc_id:
                index = semantic_doc_id.index(id)
                score += semantic_weight * (1 / (index + 1))
                content = get_content_by_doc_id(id)
                semantic_count += 1
                
            if id in bm25_doc_id:
                index = bm25_doc_id.index(id)
                score += bm25_weight * (1 / (index + 1))
                bm25_count += 1
                
                if content == "":
                        content = (
                            bm25_results[index].contextual_content
                            + "\n\n"
                            + bm25_results[index].original_content
                        ) 
                
            if id in semantic_doc_id and id in bm25_doc_id:
                both_count += 1
                
            combined_nodes.append(
                NodeWithScore(
                    node=TextNode(text=content, id_=id),
                    score=score
                )
            )
        combined_nodes.sort(key=lambda x: x.score, reverse=True)  # Sắp xếp giảm dần theo score
        return combined_nodes 
    
    def _retrieve(self, query) -> list[NodeWithScore]:
        '''
        Inherit from BaseRetriever
        '''
        semantic_results = self.contextual_search(query)
        bm25_results = self.bm25_search(query.query_str)

        combined_nodes = self.combine_results(
            semantic_results = semantic_results,
            bm25_results = bm25_results
        )
        
        reranked_nodes = self.gpt_reranking(
            query=query,
            nodes = combined_nodes[:30]
        )

        # map_reduced_nodes = self.map_reduce_postprocess(
        #     query=query,
        #     nodes = reranked_nodes
        # )

        log_retrieval(
            contextual_results=semantic_results,
            bm25_results=bm25_results, 
            combined_results=combined_nodes, 
            reranked_results=reranked_nodes, 
            query=query,
            response=None)

        return reranked_nodes