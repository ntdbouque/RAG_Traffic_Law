import os
import sys
from pathlib import Path
from dotenv import load_dotenv


from llama_index.core import PromptTemplate
from llama_index.core import get_response_synthesizer
from llama_index.llms.openai import OpenAI

from icecream import ic
from source.rag.query_engine import MyQueryEngine
from source.rag.retrieval import RetrievalPipeline
from source.settings import setting
from source.constants import QA_PROMPT

load_dotenv(override=True)

retriever = RetrievalPipeline()
llm = OpenAI(
                model=setting.model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                logprobs=None,
                default_headers={},
            )
synthesizer = get_response_synthesizer(response_mode="compact")


query = 'Tôi lái xe hơi mà trong hơi thở có nồng độ cồn thì sao?'
my_query_engine = MyQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    llm=llm,
    qa_prompt=PromptTemplate(QA_PROMPT),
)
response = my_query_engine.query(query)
ic(response)
