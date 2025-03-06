import os
import sys
import uuid
import json
from tqdm import tqdm
from icecream import ic
from pathlib import Path
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
