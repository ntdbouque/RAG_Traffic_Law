import os
from dotenv import load_dotenv
from llama_index.core import PromptTemplate, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingMode
from llama_index.core import get_response_synthesizer
from icecream import ic

from source.rag.custom_query_engine import MyQueryEngine
from source.rag.retrieval import RetrievalPipeline
from source.settings import setting
from source.constants import (
    QA_PROMPT,
    CUSTOM_REFINE_PROMPT,
)

# Load .env variables
load_dotenv(override=True)

# 0. Retriever
retriever = RetrievalPipeline()

# 1. LLM setup
llm = OpenAI(
    model=setting.model_name,
    api_key=os.getenv("OPENAI_API_KEY"),
)

# 2. Response Synthesizer
synthesizer = get_response_synthesizer(
    response_mode="compact",
    text_qa_template=PromptTemplate(QA_PROMPT),
    refine_template=PromptTemplate(CUSTOM_REFINE_PROMPT),
)

# 3. Embed model
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(
    model=setting.embed_model,
    api_key=os.getenv('OPENAI_API_KEY'),
    mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE,
)

# 4. My Query Engine (không dùng sub-question)
query_engine = MyQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    llm=llm,
    qa_prompt=PromptTemplate(QA_PROMPT),
)

# 5. RUN
query = "Người chạy xe máy lấn làn thì bị phạt thế nào?"
response = query_engine.query(query)
ic(response.response)
