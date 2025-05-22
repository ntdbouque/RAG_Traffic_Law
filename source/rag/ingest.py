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

sys.path.append(str(Path(__file__).parent.parent.parent))

from llama_index.llms.openai import OpenAI
from llama_index.core.extractors.metadata_extractors import KeywordExtractor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.llms import ChatMessage
from llama_index.core.async_utils import asyncio_run
from typing import Literal, Sequence
from source.reader.llama_parse_reader import parse_multiple_files
from source.database.elastic import ElasticSearch
from source.database.qdrant import QdrantVectorDatabase
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core import (
    Document, 
    QueryBundle, 
    StorageContext, 
    VectorStoreIndex,
)
from source.schemas import (
    QdrantPayload,
    RAGType,
    DocumentMetadata
)

from source.constants import (
    CONTEXTUAL_PROMPT,
    QA_PROMPT,
    CONTEXTUAL_MODEL,
    CONTEXTUAL_CHUNK_SIZE,
    METADATA_PROMPT,
)

from source.settings import Settings as setting
from source.reader.structured_csv_parser import parse_and_format_csv


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

    setting: setting
    llm: OpenAI
    extractor: KeywordExtractor
    
    def __init__(self, setting):
        self.setting = setting

        # Initialize model and database
        self.embed_model = OpenAIEmbedding(model=setting.embed_model, api_key=os.getenv('OPENAI_API_KEY'))
        self.llm = self.load_llm(setting.model_name)
        self.keyword_extractor = KeywordExtractor(
                                nodes=1,
                                llm = self.llm,
                                prompt_template = METADATA_PROMPT
                        )

        self.es = ElasticSearch(
            url=setting.elastic_search_url, index_name=setting.elastic_search_index_name
        )
        self.qdrant_client = QdrantVectorDatabase(url=setting.qdrant_url)

    def load_llm(self, model_name):
        return OpenAI(model=model_name)

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

    def preprocess_message(self, messages: Sequence[ChatMessage]):
        '''
        Preprocess the message for the LLM response

        Args:
            message (Sequence[ChatMessage]): The message to preprocess

        Returns: 
            Sequence[ChatMessage]: the preprocessed message
        '''
        return messages


    def add_contextual_content(
        self,
        chapter: Document, 
        articles: list[Document],
        chapter_id: int,
        chapter_uuid: str,
    ):
        '''
        Adding context to each chunk base on chapter content, article title, chapter title, document title, chunk content

        Args:
            chapter (Document): a chapter in the document
            articles (list[Document]): articles belong to this chapter in the document
            chapter_id (int): chapter index
            chapter_uuid: unique id for each chapter
        Returns:
            list[Document]:  list of document instance (used for qdrant)
            list[DocumentMetadata]: list of document metadata instance (used for elasticsearch)
        '''

        documents: list[Document] = []
        documents_metadata: list[DocumentMetadata] = []

        for idx, chunk in enumerate(articles):
            messages = [
                ChatMessage(
                    role='system',
                    content="Bạn là một trợ lý AI chuyên nghiệp, được thiết kế để cung cấp ngữ cảnh súc tích và rõ ràng cho các 'Điều' trong một 'Chương' cụ thể của Luật pháp Việt Nam"
                ),
                ChatMessage(
                    role='user',
                    content=CONTEXTUAL_PROMPT.format(
                        WHOLE_DOCUMENT=chapter.text,
                        ARTICLE_TITLE=chunk.metadata['article_title'],
                        CHAPTER_TITLE=chunk.metadata['chapter_title'],
                        TITLE=chunk.metadata['ten_luat'],
                        CHUNK_CONTENT=chunk.text
                    )
                )
            ]

            response = self.llm.chat(self.preprocess_message(messages))
            contextualized_article_content = response.message.content

            new_chunk = contextualized_article_content + '\n\n' + chunk.metadata['article_title'] + '\n' + chunk.text

            chunk_id = f'chapter{chapter_id}_article{idx}'
            article_uuid = str(uuid.uuid4())
            documents.append(
                Document(
                    text=new_chunk,
                    metadata = dict(
                        chapter_id = f'chapter{chapter_id}',
                        chapter_uuid = chapter_uuid,
                        article_id = chunk_id,
                        article_uuid = article_uuid,
                        article_content = chunk.metadata['article_title'] + '\n' + chunk.text,    
                        contextualized_article_content = contextualized_article_content
                    )
                )
            )

            documents_metadata.append(
                DocumentMetadata(
                        new_chunk = new_chunk,
                        chapter_id = f'chapter{chapter_id}',
                        chapter_uuid = chapter_uuid,
                        article_id = chunk_id,
                        article_uuid = article_uuid,
                        article_content = chunk.metadata['article_title'] + '\n' + chunk.text,
                        contextualized_article_content = contextualized_article_content,
                ),
            )

        return documents, documents_metadata

    def get_contextual_documents(self, splitted_chapters: list[Document], splitted_articles: list[list[Document]]):
        '''
        Get the contextual documents from the splitted chapters and articles

        Args:
            splitted_chapters (list[Document]): List of chapters in the document
            splitted_articles (list[list[Document]]): List of list of articles in the document which a list of article is a chapter

        Return:
            list[Document]: used for ingest Qdrant Server
            list[DocumentMetadata]: used for ingest ElasticSearch Server 
        '''
        chapter_id = 0
        documents: list[Document] = []
        documents_metadata: list[DocumentMetadata] = []

        for chapter, articles in tqdm(
            zip(splitted_chapters, splitted_articles), 
            desc='Adding contextual content for each document...',
            total=len(splitted_articles)
        ):
            chapter_uuid = str(uuid.uuid4())
            document, metadata = self.add_contextual_content(chapter, articles, chapter_id, chapter_uuid)

            documents.extend(document)
            documents_metadata.extend(metadata)

            chapter_id += 1

        return documents, documents_metadata

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
        ic(documents[0].metadata['article_content'])
        
        vectors = [self.get_embedding(doc.text) for doc in tqdm(documents, desc='Getting embeddings ...')]
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

    def run_ingest(
        self,
        folder_dir: str,
        type: Literal['origin', 'contextual', 'both'] = 'contextual'
    ) -> None:
        '''
        Run the ingest process for Retrieval Augmented Generation System

        Args:
            folder_dir (str | Path): The folder directory containing documents
            type: Literal['origin, contextual', 'both']: The type to ingest. Default `contextual`
        '''

        # raw_documents = parse_multiple_files(folder_dir)
        # ic(len(raw_documents))
        # splitted_chapters, splitted_articles = split_documents(self.keyword_extractor, raw_documents) 
        
        #splitted_chapters, splitted_articles = parse_and_format_csv(folder_dir)
        
        #ic(len(splitted_chapters), len(splitted_articles))
        #contextual_documents, contextual_documents_metadata = (
        #    self.get_contextual_documents(splitted_chapters, splitted_articles)
        #)

        # Ingest data to Vector DB
        self.ingest_data_qdrant(contextual_documents)
        self.ingest_data_elastic(contextual_documents_metadata)
    