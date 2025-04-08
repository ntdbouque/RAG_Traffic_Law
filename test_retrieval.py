import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from icecream import ic
from api.services import ChatbotTrafficLawRAG
from source.rag.retrieval import RetrievalPipeline
from source.settings import setting as ConfigSetting


# query = 'Tôi bị tai nạn giao thông, tôi phải làm gì?'
# bot = ChatbotTrafficLawRAG()
# ic(bot.predict(query))


query = 'Tôi bị tai nạn giao thông, tôi phải làm gì?'
retriever = RetrievalPipeline(ConfigSetting)
ic(retriever.hybrid_rag_search(query))