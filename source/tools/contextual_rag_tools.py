import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from source.rag.retrieval import RetrievalPipeline
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings
from time import time
# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
# llama_debug = LlamaDebugHandler(print_trace_on_end=True)
# callback_manager = CallbackManager([llama_debug])

# Settings.callback_manager = callback_manager


def load_contextual_rag_tool():
    my_query_engine:CustomQueryEngine = RetrievalPipeline()
    my_query_engine_tools = [
        QueryEngineTool(
            query_engine = my_query_engine,
            metadata = ToolMetadata(
                name = 'Traffic_Law_RAG',
                description = "A useful tool to answer queries of using Traffic Law RAG"
            )
        )
    ]
    sub_questions_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools = my_query_engine_tools,
        use_async = True,
    )
    
    return QueryEngineTool.from_defaults(
        query_engine = sub_questions_query_engine,
        name='Integrating_Sub_Questions_Query_Engine_into_Traffic_Law_RAG',
        description='A tool to decompose user queries into sub-questions and process them with Traffic Law RAG'
        )
    
# if __name__ == '__main__':
    ## test sub question query engine:
    
from llama_index.core import QueryBundle
import nest_asyncio
import asyncio
nest_asyncio.apply()

# my_query_engine:CustomQueryEngine = RetrievalPipeline()
my_query_engine = RetrievalPipeline()
my_query_engine_tools = [
    QueryEngineTool(
        query_engine = my_query_engine,
        metadata = ToolMetadata(
            name = 'Traffic_Law_RAG',
            description = "A useful tool to answer queries of using Traffic Law RAG"
        )
    )
]

sub_questions_query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools = my_query_engine_tools,
    use_async = True,
)

response = sub_questions_query_engine.query(
    "Tôi uống rượu bia và tham gia giao thông, tôi còn gây tai nạn tôi bị phạt thế nào?"
)

# query_bundle = QueryBundle("Tôi uống rượu bia và tham gia giao thông, tôi còn gây tai nạn tôi bị phạt thế nào?")
# start = time()

# async def async_query(query_bundle_):
#     response = await sub_questions_query_engine.aquery(
#         query_bundle_
#     )
# print("time: ", time() -start)

# # # Run the async query within an event loop
# asyncio.run(async_query(query_bundle))