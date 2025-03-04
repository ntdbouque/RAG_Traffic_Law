'''
Author:Nguyen Truong Duy
Purpose: Implement the retrieval pipeline
Date: 18-02-2025
'''

import os
import sys
from icecream import ic
import json
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).parent.parent.parent))

from source.settings import setting as ConfigSetting
from source.database.elastic import ElasticSearch
from source.database.qdrant import QdrantVectorDatabase
from source.constants import QA_PROMPT

from qdrant_client import QdrantClient
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, Node
from llama_index.core import (
    Settings,
    Document,
    QueryBundle,
    StorageContext,
    VectorStoreIndex,
)
from typing import Literal, Sequence

load_dotenv()

class RetrievalPipeline():
    '''
    Contextual Retrieval-Augmented Generation (RAG) class to search for relevant article from Vietnamese Traffic Law
    '''
    def __init__(self, setting: ConfigSetting, k: int = 150):
        self.setting = setting
        ic(self.setting)
        
        self.k = k
        
        self.es_client = ElasticSearch(
            url = self.setting.elastic_search_url,
            index_name = self.setting.elastic_search_index_name
        )
        
        self.qdrant_client = QdrantVectorDatabase(
            url = self.setting.qdrant_url
        )
        
        self.qdrant_index = self.get_qdrant_vector_store_index(
            self.qdrant_client.client, self.setting.contextual_rag_collection_name
        )
        
        self.retriever = VectorIndexRetriever(
            index=self.qdrant_index,
            similarity_top_k = k,
        )
        
        self.query_engine = RetrieverQueryEngine(retriever=self.retriever)
        
        self.reranker = CohereRerank(
            top_n = self.setting.top_n,
            api_key = os.getenv('COHERE_API_KEY')
        )
        
        self.llm = OpenAI(model=self.setting.model_name)
    
    def get_qdrant_vector_store_index(self, client: QdrantClient, collection_name: str):
        '''
        Get the QdrantVectorStoreIndex from the QdrantVectorStore
        
        Args:
            client (QdrantClient): The Qdrant client
            collection_name (str): The collection name
        Returns:
            VectorStoreIndex: The VectorStoreIndex from QdrantVectorStore
        '''
        
        ic(collection_name)
        vector_store = QdrantVectorStore(client=client, collection_name=collection_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        return VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    
    def contextual_search(self, query, k: int = 150):
        '''
        Search the query with the contextual RAG (Qdrant)
        
        Args:
            query (str): The query string
            k (int): The number of documents to return  
        Return:
            list: The list of document id from the search result
        ''' 
        
        ic(query, k)
        semantic_results: Response = self.query_engine.query(query)
        # semantic_doc_id = [
        #     node.metadata["article_uuid"] for node in semantic_results.source_nodes
        # ]
        ic(len(semantic_results.source_nodes))
        return semantic_results
    
    def bm25_search(self, query, k: int = 150) -> str:
        '''
        Search the query with the keyword search (ElasticSearch)
        
        Args:
            query (str): The query string
            k (int): The number of documents to return  
        '''
        
        ic(query, k)
        bm25_results = self.es_client.search(query, k)
        ic(len(bm25_results))
        #bm25_doc_id = [doc.doc_id for doc in bm25_results]
        
        return bm25_results
    
    def rewrite_query_and_rerank(self, combined_nodes, query) -> list[NodeWithScore]:
        '''
        Rewrite query and rerank the search results with the Cohere model
        Args:
            combined_nodes (list[NodeWithScore]): The combined nodes
            rewrited_query (str): The rewrited query
        Return:
            list[NodeWithScore]: The reranked nodes
        '''
        query_bundle = QueryBundle(query_str=query)
        return self.reranker.postprocess_nodes(combined_nodes, query_bundle)

    def combine_results(self, semantic_results, bm25_results) -> list[NodeWithScore]:
        '''
        Combine the search results from semantic search and bm25 search
        # Compute score reference at https://github.com/anthropics/anthropic-cookbook/blob/main/skills/contextual-embeddings/guide.ipynb
        
        Args:
            semantic_doc_id (list): The list of document id from semantic search
            bm25_doc_id (list): The list of document id from bm25 search
        Return:
            list: The combined list of document id
        '''
        
        semantic_doc_id = [
            node.metadata["article_uuid"] for node in semantic_results.source_nodes
        ]
        bm25_doc_id = [result.doc_id for result in bm25_results]
        
        
        def get_content_by_doc_id(doc_id: str):
            for node in semantic_results.source_nodes:
                if node.metadata["article_uuid"] == doc_id:
                    return node.text
            return ""
        
        semantic_weight = self.setting.semantic_weight
        bm25_weight = self.setting.bm25_weight
        
        combined_nodes: list[NodeWithScore] = []
        combined_ids = list(set(semantic_doc_id + bm25_doc_id))
        
        ic(len(combined_ids))
        
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
                    node=Node(text=content),
                    score=score
                )
            )
        ic(both_count)
        return combined_nodes 

    def preprocess_message(self, messages: Sequence[ChatMessage]):
        '''
        Preprocess the message for the LLM response

        Args:
            message (Sequence[ChatMessage]): The message to preprocess

        Returns: 
            Sequence[ChatMessage]: the preprocessed message
        '''
        return messages

    def generate_response(self, query, contexts):
        messages = [
            ChatMessage(
                role="system",
                content="You are a helpful assistant.",
            ),
            ChatMessage(
                role="user",
                content=QA_PROMPT.format(
                    context_str=json.dumps(contexts),
                    query_str=query,
                ),
            ),
        ]
        return self.llm.chat(self.preprocess_message(messages)).message.content
    
    def hybrid_rag_search(self, query) -> str:
        '''
        Search the query with the contextual RAG
        
        Args:
            query (str): The query string
            k (int): The number of documents to return  
        ''' 
        
        bm25_results = self.bm25_search(query, self.k)
        semantic_results = self.contextual_search(query, self.k)
        
        combined_nodes = self.combine_results(semantic_results, bm25_results)
                
        retrieved_nodes = self.rewrite_query_and_rerank(combined_nodes, query)
        
        contexts = [n.node.text for n in retrieved_nodes]
        
        response  = self.generate_response(query, contexts)
        ic(response)
        return response
    
if __name__ == '__main__':
    query = "tai sao can doi mu bao hiem khi tham gia giao thong"
    RetrievalPipeline = RetrievalPipeline(ConfigSetting)
    response = RetrievalPipeline.hybrid_rag_search(query)
    ic(response)