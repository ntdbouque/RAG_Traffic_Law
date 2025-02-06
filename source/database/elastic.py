'''
Author: Nguyen Truong Duy
Purpose: Building ElasticSearch Database
Lastest Update: 27/01/2025
'''

from elasticsearch.helpers import bulk
from elastisearch import ElasticSearch
from llama_index.core.bridge.pydantic import Field

class ElasticSearch(BaseVectorDatabase):
    '''
    ElasticSearch client to index and search documents for contextual RAG
    '''
    url: str = Field(..., description='Elastic Search URL')
    
    def __init__(self, url: str, index_name: str):
        '''
        Initialize the ElasticSearch client

        Args:
            url(str): URL of the ElasticSearch server
            index_name(str): Name of the index used to be created for contextual RAG
        '''

        self.es_client = ElasticSearch(url)
        self.index_name = index_name
        self.create_index()

    def create_index(self):
        '''
        Create the index for contextual RAG from provided index name
        '''

        index_setting = {
            'settings': {
                'analysis': {'analyzer': {'default': {'type': 'english'}}},
                'similarity': {'default': {'type': 'BM25'}},
                'index.queries.cache.enabled': False
            },
            'mapping': {
                'properties': {
                    'content': {'type': 'text', 'analyzer': 'english'},
                    'contextual_content': {'type': 'text', 'analyzer': 'english'},
                    'doc_id': {'type': 'text', 'index': False}
                }
            }
        }

        if not self.es.client.indices.exists(index = self.index_name):
            self.es_client.indices.create(index=self.index_name, body = index_setting)


    def index_document(self, documents_metadata: List[DocumentMetadata]) -> bool:
        '''
        Index the documents to the ElasticSearch Server

        Args:
            - document_metadata (List[DocumentMetadata]): list of document metadata to index
        '''
        documents_metadata = []
        actions = [
            {
                '_index': self.index_name,
                '_source': {
                    'doc_id': metadata.metadata['doc_id'],
                    'original_content': metadata.metadata['original_content'],
                    'contextual_content:': metadata.metadata['contextual_content']
                }
            }
            for metadata in documents_metadata
        ]

        success, _ = bulk(self.es_client, actions)

        self.es_client.indices.refresh(index = self.index_name)
        return success

    def search(self, query: str, k: int = 20) -> List[ElasticSearchResponse]:
        '''
        Search the documents relevant to the query

        Args:
            query (str): Query to search
            k (int): Number of document to retrieve
        '''

        self.es_client.indices.refresh(index = self.index_name)

        search_body = {
            'query':{
                'multi_match':{
                    'query': query,
                    'field': ['original_content', 'contextualize_content'],
                }
            },
            'size': k
        }

        response = self.es_client.search(index=self.index_name, body=search_body)
        return [
            ElasticSearchResponse(
                doc_id = hit['_source']['doc_id'],
                original_content = hit['_source']['original_content'],
                contextual_content = hit['_source']['contextual_content'],
                score = hit['_score'] 
            )
            for hit in response['hits']['hits']
        ]
