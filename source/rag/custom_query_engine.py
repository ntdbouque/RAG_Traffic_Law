import os
import sys
from icecream import ic
import json
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).parent.parent.parent))

from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI


load_dotenv(override=True)

class MyQueryEngine(CustomQueryEngine):
    """My Query Engine."""

    retriever: BaseRetriever
    response_synthesizer: BaseSynthesizer

    def custom_query(self, query_str: str):
        nodes = self.retriever.retrieve(query_str)
        response_obj = self.response_synthesizer.synthesize(query_str, nodes)
        return response_obj