from mcp import ClientSession
from mcp.client.sse import sse_client

async def check():
    async with sse_client("http://localhost:8000/sse") as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()

            # List avail tool
            tools = await session.list_tools()
            print(tools)

            # Call add tool
            result = await session.call_tool("chat_with_my_bot", arguments={'query': "Xin chào"})
            print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(check())