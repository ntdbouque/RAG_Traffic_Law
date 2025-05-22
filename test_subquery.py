import os

os.environ["OPENAI_API_KEY"] = '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'

import nest_asyncio

nest_asyncio.apply()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

Settings.callback_manager = callback_manager

pg_essay = SimpleDirectoryReader(input_dir="data/graham").load_data()

# build index and query engine
vector_query_engine = VectorStoreIndex.from_documents(
    pg_essay,
    use_async=True,
).as_query_engine()

# setup base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="pg_essay",
            description="Paul Graham essay on What I Worked On",
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
)

response = query_engine.query(
    "How was Paul Grahams life different before, during, and after YC?"
)
