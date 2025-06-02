import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from icecream import ic
from source.rag.retrieval import RetrievalPipeline
from source.settings import Settings


query = 'người được chở trên xe máy mà sử dụng ô dù thì bị phạt thế nào?'
retriever = RetrievalPipeline()
response = retriever.retrieve(query)
ic(response[0:3])
