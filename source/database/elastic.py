'''
Author: Nguyen Truong Duy
Purpose: Building ElasticSearch Database
- Adding test case
Lastest Update: 10/02/2025
'''

import sys
import os
from pathlib import Path
from icecream import ic
sys.path.append(str(Path(__file__).parent.parent.parent))

from elasticsearch.helpers import bulk
from elasticsearch import Elasticsearch
from llama_index.core.bridge.pydantic import Field

from source.database.base import BaseVectorDatabase
from source.schemas import ElasticSearchResponse

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

        self.es_client = Elasticsearch(url)
        self.index_name = index_name
        if self.test_connection():
            if self.check_collection_exists():
                ic(f'Collection {self.index_name} have existed')
            else:
                self.create_collection()
                ic(f'Collection {self.index_name} is created')

        else:
            ic('Can not connect to ElasticSearch server')

    def check_collection_exists(self):
        '''
        Check whether a specified collection existed

        Return: 
            boolean: True if existed and False if did not existed
        '''
        return self.es_client.indices.exists(index=self.index_name)
    
    def test_connection(self):
        '''
        Test the connection to ElasticSearch server

        Return:
            boolean: True if successfully connect and False if unsuccessfully connect
        '''
        return self.es_client.ping()

    def create_collection(self):
        '''
        Create the index for contextual RAG from provided index name
        '''

        index_setting = {
            'settings': {
                'analysis': {'analyzer': {'default': {'type': 'english'}}},
                'similarity': {'default': {'type': 'BM25'}},
                'index.queries.cache.enabled': False
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "english"},
                    "contextualized_content": {"type": "text", "analyzer": "english"},
                    "doc_id": {"type": "text", "index": False},
                }
            },
        }

        if not self.es_client.indices.exists(index = self.index_name):
            self.es_client.indices.create(index=self.index_name, body = index_setting)


    def index_document(self, actions) -> bool:
        '''
        Index the documents to the ElasticSearch Server

        Args:
            - actions: list of actions
        '''
        success, _ = bulk(self.es_client, actions)

        self.es_client.indices.refresh(index = self.index_name)
        return success

    def search(self, query: str, k: int = 20) -> list[ElasticSearchResponse]:
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

if __name__ == '__main__':
    from source.settings import setting
    ic(setting)
    
    es = ElasticSearch(
        url=setting.elastic_search_url, index_name=setting.elastic_search_index_name
    )
