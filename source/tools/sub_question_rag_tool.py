import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from llama_index.core.tools import FunctionTool

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
    CUSTOM_REFINE_PROMPT,
    CUSTOM_OPENAI_SUB_QUESTION_PROMPT
)

import nest_asyncio

load_dotenv(override=True)


def load_sub_question_rag_tool():
    retriever = RetrievalPipeline()
    llm = OpenAI(
        model=setting.model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        logprobs=None,
        default_headers={},
    )
    synthesizer = get_response_synthesizer(
        response_mode="refine", 
        text_qa_template=PromptTemplate(QA_PROMPT)
)

    my_query_engine = MyQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        llm=llm,
        qa_prompt=PromptTemplate(QA_PROMPT),
    )

    question_gen = OpenAIQuestionGenerator.from_defaults(
        llm=llm, 
        prompt_template_str=CUSTOM_OPENAI_SUB_QUESTION_PROMPT)

    Settings.llm = llm
    Settings.embed_model = OpenAIEmbedding(
        model=setting.embed_model, 
        api_key=os.getenv('OPENAI_API_KEY'),
        mode=OpenAIEmbeddingMode.TEXT_SEARCH_MODE)

    query_engine_tools = [
        QueryEngineTool(
            query_engine=my_query_engine,
            metadata=ToolMetadata(
                name="RAG Traffic Law tool",
                description="content of decree 168/2024",
            ),
        ),
    ]
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        question_gen = question_gen,
        response_synthesizer = synthesizer,
        use_async=True,
    )

    def answer_complex_query(query_str: str) -> str:
        """
        A helpful function to answer a complex, multi_part query.

        Args:
            query_str (str): The query string to search for.

        Returns:
            str: The answer to the query.
        """
        return query_engine.query(query_str)

    return FunctionTool.from_defaults(
        fn=answer_complex_query,
        return_direct=True,
        description = (
            "Use this tool for complex, multi-part, or high-level questions that require "
            "breaking down into sub-questions. Useful when answering requires reasoning "
            "over multiple laws or conditions using RAG."
        )
    )