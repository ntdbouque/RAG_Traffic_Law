import os
import sys
from pathlib import Path
from dotenv import load_dotenv


from llama_index.core import PromptTemplate
from llama_index.core import get_response_synthesizer
from llama_index.llms.openai import OpenAI
from llama_index.question_gen.openai import OpenAIQuestionGenerator
from llama_index.embeddings.openai import OpenAIEmbeddingMode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

from icecream import ic
from source.rag.custom_query_engine import MyQueryEngine
from source.rag.retrieval import RetrievalPipeline
from source.settings import setting
from source.constants import(
    QA_PROMPT,
    CUSTOM_OPENAI_SUB_QUESTION_PROMPT,
    CUSTOM_REFINE_PROMPT
)

load_dotenv(override=True)

retriever = RetrievalPipeline()

# 0. llm
llm = OpenAI(
    model=setting.model_name,
    api_key=os.getenv("OPENAI_API_KEY"),
    logprobs=None,
    default_headers={},
)
# 1. Response_synthesizer
synthesizer = get_response_synthesizer(
    response_mode="compact", 
    text_qa_template=PromptTemplate(QA_PROMPT),
    refine_template=PromptTemplate(CUSTOM_REFINE_PROMPT)
)

# 2. Query Engine
my_query_engine = MyQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    llm=llm,
    qa_prompt=PromptTemplate(QA_PROMPT),
)

# 3. Question Gen:
question_gen = OpenAIQuestionGenerator.from_defaults(
    llm=llm, 
    prompt_template_str=CUSTOM_OPENAI_SUB_QUESTION_PROMPT)

# Setting:
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(
    model=setting.embed_model, 
    api_key=os.getenv('OPENAI_API_KEY'),
    mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE)

# 4.Query Engine Tools:
query_engine_tools = [
    QueryEngineTool(
        query_engine=my_query_engine,
        metadata=ToolMetadata(
            name="RAG Traffic Law tool",
            description="content of decree 168/2024",
        ),
    ),
]

# 5. Sub Queston Query Engine:
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    question_gen = question_gen,
    response_synthesizer = synthesizer,
    use_async=True,
)

# RUNNING:
query = 'Người chạy xe máy lấn làn thì bị phạt thế nào?'
response = query_engine.query(query)
ic(response)