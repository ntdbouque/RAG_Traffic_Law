import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from icecream import ic

from source.settings import setting as ConfigSetting
from source.rag.retrieval import RetrievalPipeline

RetrievalPipeline = RetrievalPipeline(ConfigSetting)
query = 'Tôi bị tai nạn giao thông, tôi phải làm gì?'
response = RetrievalPipeline.hybrid_rag_search(query)
ic(response)