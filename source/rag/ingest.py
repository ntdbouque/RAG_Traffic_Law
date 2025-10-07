'''
Author: Nguyen Truong Duy
Purpose: Ingest Data to Qdrant and ElasticSearch server
Latest Update: 10/02/2025
'''

import os
import sys
import uuid
import json
from tqdm import tqdm
from icecream import ic
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import glob
import pandas as pd
sys.path.append(str(Path(__file__).parent.parent.parent))
load_dotenv(override=True)

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from typing import Literal, Sequence
from source.database.elastic import ElasticSearch
from source.database.qdrant import QdrantVectorDatabase
from llama_index.core import (
    Document, 
    Settings
)

from openai import OpenAI

from source.schemas import (
    QdrantPayload,
    RAGType,
    DocumentMetadata
)

from source.constants import (
    QA_PROMPT,
    CONTEXTUAL_MODEL,
)

from source.settings import setting

import tiktoken

def cut_text_to_token_limit(text: str, max_tokens: int = 8100) -> str:
    # Sử dụng mã hóa GPT-3/4 (cl100k_base)
    encoder = tiktoken.get_encoding("cl100k_base")
    
    # Mã hóa văn bản thành các token
    tokens = encoder.encode(text)
    
    # Kiểm tra số token và cắt bớt nếu cần
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]  # Giới hạn số token
        text = encoder.decode(tokens)  # Giải mã lại văn bản sau khi cắt
        print(f"Văn bản đã được cắt xuống {max_tokens} token.")
    
    return text


class DocumentIngestionPipeline:
    '''
    Ingest document to ElasticSearch server and Qdrant server
    '''
    
    def __init__(self, setting):
        ic(setting)
        self.setting = setting

        # Initialize model and database
        self.embed_model = OpenAIEmbedding(model=setting.embed_model, api_key=os.getenv('OPENAI_API_KEY'))
        
        Settings.embed_model = self.embed_model
        
        self.es = ElasticSearch(
            url=setting.elastic_search_url, index_name=setting.elastic_search_index_name
        )
        self.qdrant_client = QdrantVectorDatabase(url=setting.qdrant_url)

        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def get_embedding(
        self,
        chunk: str, 
    ):
        '''
        Get embedding for each chunk

        Args:
            chunk (str): the chunk to embedding
        
        Return:
            Tensor: embedding format of chunk
        '''
        
        chunk = cut_text_to_token_limit(chunk)
        
        return self.embed_model.get_text_embedding(chunk)


    def ingest_data_elastic(
        self, 
        documents_metadata: list[DocumentMetadata],
        type: Literal['origin', 'contextual'] = 'contextual'
    ):
        '''
        Ingest document to ElasticSearch Server

        Args:
            documents_metadata (list[DocumentMetadata]): list of document to ingest
            type (str): mode to ingest, origin or contextual
        '''
        actions = [
            {
                '_index': self.setting.elastic_search_index_name,
                '_source': {
                    'doc_id': metadata.article_uuid,
                    'article_id': metadata.article_id,
                    'original_content': metadata.article_content,
                    'contextual_content': metadata.contextualized_article_content
                }
            }
            for metadata in documents_metadata
        ]

        success = self.es.index_document(actions)
        return success

    def ingest_data_qdrant(
        self,
        documents: list[Document],
        type: Literal['origin', 'contextual'] = 'contextual'
    ):
        '''
        Ingest the data to the QdrantVectorStore

        Args:
            documents: list of chapter content to ingest to Qdrant Server
            type: mode to ingest (origin or contextual)
            
        '''
        
        vectors = [self.get_embedding(doc.text) for doc in documents]
        payloads = [
            QdrantPayload(
                chapter_uuid = doc.metadata['chapter_uuid'],
                text = doc.text,
                article_uuid = doc.metadata['article_uuid'], 
                article_id = doc.metadata['article_id'],
                original_content = doc.metadata['article_content']
            )
            for doc in documents
        ]

        self.qdrant_client.add_vectors(
            collection_name=self.setting.contextual_rag_collection_name,
            vectors=vectors,
            payloads = payloads,
            vector_ids = [doc.metadata['article_uuid'] for doc in documents]
        )

    def get_context(
        self,
        chunk: str,
        chapter_title: str,
        article_title: str,
        sarticle_position,
        full_article_content,
        decree_name: str = "Quy định xử phạt vi phạm hành chính về trật tự, an toàn giao thông trong lĩnh vực giao thông đường bộ; trừ điểm, phục hồi giấy phép lái xe",
    ):
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            response_format={ 
                "type": "json_object"
            },
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Bạn là một trợ lý pháp lý thông minh, có nhiệm vụ tạo context cho từng đoạn văn bản pháp luật (chunk) "
                        "để hỗ trợ truy vấn ngữ nghĩa (contextual RAG)."
                    ),
                },
                {
                    "role": "user",
                    "content": CONTEXTUAL_PROMPT.format(
                        DECREE_NAME=decree_name,
                        CHAPTER_NAME=chapter_title,
                        ARTICLE_TITLE=article_title, 
                        SARTICLE_POSITION=sarticle_position,
                        FULL_ARTICLE_CONTENT=full_article_content,
                        CHUNK_CONTENT=chunk,
                    ),
                },
            ]
        )
        return json.loads(response.choices[0].message.content)['context']


    def run_ingest_from_csv(
        self,
        folder_path: str,
    ) -> None:
        
        lst_csv_paths = glob.glob(os.path.join(folder_path, '*.csv'))
        for csv_path in lst_csv_paths: 
            contextual_documents : list[Document] = []
            contextual_documents_metadata: list[DocumentMetadata] = []
            df = pd.read_csv(csv_path)
            
            for i, row in tqdm(df.iterrows(), total=len(df)):
                chunk = row['Chunk'].replace('\r', '').replace('\n', '')
                chapter_title = row['chapter_title'].replace('\r', '').replace('\n', '')
                article_title = row['article_title'].replace('\r', '').replace('\n', '')
                sarticle_position = row['sarticle_position']
            
                
                if pd.isna(sarticle_position):
                    full_article_content = None
                else:
                    full_article_content = row['full_article_content'].replace('\r', '').replace('\n', '')

                #contextual_content = self.get_context(chunk, chapter_title, article_title, sarticle_position, full_article_content)
                #new_chunk = contextual_content + '\n\n' + chunk
                new_chunk = article_title + '\n\n' + chunk
                article_uuid = str(uuid.uuid4())
            
                contextual_documents.append(
                    Document(
                            text=new_chunk,
                            metadata = dict(
                                chapter_id = '',
                                chapter_uuid = '',
                                article_id = '',
                                article_uuid = article_uuid,
                                article_content = chunk,    
                                contextualized_article_content = ''
                            )
                        )
                )
                
                contextual_documents_metadata.append(
                    DocumentMetadata(
                                new_chunk = new_chunk,
                                chapter_id = '',
                                chapter_uuid = '',
                                article_id = '',
                                article_uuid = article_uuid,
                                article_content = chunk,
                                contextualized_article_content = '',
                        ),
                )
            ic(len(contextual_documents))
            ic(len(contextual_documents_metadata))

            self.ingest_data_qdrant(contextual_documents)
            self.ingest_data_elastic(contextual_documents_metadata)

if __name__ == '__main__':
    ingestor = DocumentIngestionPipeline(setting)
    folder_dir = './sample'
    ingestor.run_ingest_from_csv(folder_dir)
