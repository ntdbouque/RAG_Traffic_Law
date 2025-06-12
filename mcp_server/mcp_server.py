from mcp.server.fastmcp import FastMCP
import requests

# Auto mở ở cổng 8000
mcp = FastMCP(
    name="mcp-server",
)

@mcp.tool()
def chat_with_my_bot(query: str) -> str:
    """Chat with Traffic Law RAG Chatbot"""
    url = "http://localhost:9186/v1/complete"
    payload = {"message": query}

    try:
        response = requests.post(url, json=payload)
        return response.text
    except requests.RequestException as e:
        return f"Lỗi: {str(e)}"

if __name__ == "__main__":
    # Initialize and run the server
    # print("Listening...")
    mcp.run(transport='stdio') #'stdio'