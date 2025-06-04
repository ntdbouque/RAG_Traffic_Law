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

from icecream import ic
from source.rag.custom_query_engine import MyQueryEngine
from source.rag.retrieval import RetrievalPipeline
from source.settings import setting
from source.constants import(
    QA_PROMPT,
    CUSTOM_REFINE_PROMPT
)

import nest_asyncio

load_dotenv(override=True)

def load_contextual_rag_tool():
    retriever = RetrievalPipeline()
    llm = OpenAI(
                    model=setting.model_name,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    logprobs=None,
                    default_headers={},
                )
    synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=PromptTemplate(QA_PROMPT),
        refine_template=PromptTemplate(CUSTOM_REFINE_PROMPT),
    )
    my_query_engine = MyQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
        llm=llm,
        qa_prompt=PromptTemplate(QA_PROMPT),
    )

    def answer_simple_query(query_str: str) -> str:
        """
        A helpful function to answer a simple query.

        Args:
            query_str (str): The query string to search for.

        Returns:
            str: The answer to the query.
        """
        return my_query_engine.query(query_str)

    return FunctionTool.from_defaults(
        fn=answer_simple_query,
        return_direct=True,
        description = (
            "Use this tool for simple legal or factual questions that can be answered "
            "in one step using Retrieval-Augmented Generation (RAG). "
            "For example, questions that ask about a single regulation, fine, or condition."
        )
    )