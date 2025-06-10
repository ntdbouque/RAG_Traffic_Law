import asyncio
from llama_index.storage.chat_store.redis import RedisChatStore

async def main():
    # Khởi tạo RedisChatStore với TTL 300 giây
    chat_store = RedisChatStore(redis_url="redis://localhost:6379", ttl=300)

    # Key Redis cần kiểm tra TTL
    user_id = "user_2"
    session_id = "8b4f5664-55ac-490a-a5a6-6154a828344a"
    key = f"{user_id}:{session_id}"

    # Lấy TTL còn lại
    remaining =  chat_store.redis_client.ttl(key)

    # In ra kết quả
    if remaining == -2:
        print(f"Key '{key}' không tồn tại.")
    elif remaining == -1:
        print(f"Key '{key}' tồn tại nhưng không có TTL (không hết hạn).")
    else:
        print(f"TTL còn lại của key '{key}': {remaining} giây.")

if __name__ == "__main__":
    asyncio.run(main())
