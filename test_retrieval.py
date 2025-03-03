import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from icecream import ic
from api.services import ChatbotTrafficLawRAG

query = 'Tôi bị tai nạn giao thông, tôi phải làm gì?'
bot = ChatbotTrafficLawRAG()
ic(bot.predict(query))