import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from icecream import ic
from source.rag.retrieval import RetrievalPipeline
from source.settings import Settings

# query = 'Tôi bị tai nạn giao thông, tôi phải làm gì?'
# bot = ChatbotTrafficLawRAG()
# ic(bot.predict(query))


query = 'Tôi bị tai nạn giao thông, tôi phải làm gì?'
# setting = Settings()
# ic(setting)
#retriever = RetrievalPipeline(setting.elastic_search_index_name, setting.contextual_rag_collection_name)
retriever = RetrievalPipeline()
response = retriever.query(query)
ic(response)
