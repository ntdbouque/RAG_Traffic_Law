import requests

def chat_with_my_bot(query: str) -> str:
    url = "http://localhost:9186/v1/complete"
    payload = {"message": query}

    try:
        response = requests.post(url, json=payload)
        return response.text
    except requests.RequestException as e:
        return f"Lỗi: {str(e)}"


print(chat_with_my_bot("Xin chào!"))
