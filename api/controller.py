from fastapi import APIRouter, Request
from .services import ChatbotTrafficLawRAG

router = APIRouter()
assistant = ChatbotTrafficLawRAG()

@router.post('/complete')
async def complete_text(request: Request):
    data = await request.json()
    message = data.get('message')

    response =  assistant.predict(message)
    return response