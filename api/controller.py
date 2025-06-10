from fastapi import APIRouter, Request
from .services import ChatbotTrafficLawRAG
from llama_index.storage.chat_store.redis import RedisChatStore
import uuid
from icecream import ic


router = APIRouter()
chat_store = RedisChatStore(redis_url="redis://localhost:6379", ttl=86400)

user_agent_dct = {} 

@router.post('/new_session')
async def new_session(request: Request):
    data = await request.json()
    user_id = data.get('user_id')

    # if trial version, default user_id = 'trial_user'
    if not user_id or user_id == '':
        user_id = 'trial_user'
    
    # create new session id for each user (trial or member)
    session_id = uuid.uuid4()
    
    # get all chat from redis
    existing_chats = chat_store.get_keys()
    ic(existing_chats)
    user_session_pair_key = f'{user_id}:{session_id}'

    # check if chat in credis, not create any OpenAIAgent
    if user_session_pair_key in existing_chats:
        return {'message': f"Session already exists for user id: {user_id}"}
    else:
        # user
        user_agent_dct[user_id] = ChatbotTrafficLawRAG(chat_store, user_session_pair_key)
        ic(user_agent_dct)

        return {
            'message': f"Session not found, create a new sesion",
            'key': f"{user_id}:{session_id}"        
        }
        
@router.post('/end_session')
async def end_session(request: Request):
    # khi người dùng đóng ứng dụng
    pass

@router.post('/complete')
async def complete(request: Request):
    data = await request.json()
    message = data.get('message')
    user_id = data.get('user_id')
    session_id = data.get('session_id')

    assistant = user_agent_dct[f"{user_id}"]
    ic(assistant)
    response =  assistant.predict(message)
    return response