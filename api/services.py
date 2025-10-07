import os
import logging 
from icecream import ic
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool

from starlette.responses import StreamingResponse, Response

from source.tools.contextual_rag_tool import load_contextual_rag_tool
from source.tools.sub_question_rag_tool import load_sub_question_rag_tool
from source.settings import setting

# for async
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
import nest_asyncio
nest_asyncio.apply()

load_dotenv(override=True)

class ChatbotTrafficLawRAG:
    def __init__(self):
        self.tools = self.load_tools()
        self.query_engine = self.create_query_engine()

    def load_tools(self):
        """
        Load default RAG tool.
        """
        contextual_rag_tool = load_contextual_rag_tool()
        #sub_question_rag_tool = load_sub_question_rag_tool()
        return [contextual_rag_tool]

    def load_model(self):
        '''
        Select a model for text generation using multiple services
        Args:
            service (str): Service name indicating the type of model to load
            model_id (str): Model ID to load
        Return:
            LLM: llama-index LLM for text generation
        Raise:
            ValueError: If the service is not supported
        '''
        return OpenAI(
            model=setting.model_name,
            temperature=0.2,
            api_key = os.getenv('OPENAI_API_KEY')
        )

    def create_query_engine(self):
        '''
        Create a query engine 
        '''

        llm = self.load_model()
        Settings.llm = llm

        query_engine = OpenAIAgent.from_tools(
            tools = self.tools, verbose = True, llm = llm
        )
        
        return query_engine
    
    def complete(self, query: str) -> str:
        """
        Generate response for given query (useless).

        Args:
            query (str): The input query.

        Returns:
            str: The response.
        """
        return self.query_engine.chat(query)

    def predict(self, prompt):
        '''
        Predict the next sequence of text given a prompt using loaded language model

        Args:
            prompt (str): The prompt to generate text from
        Return:
            str: The generated text
        '''

        response = self.query_engine.chat(prompt)
        return Response(
            content=response.response,  
            media_type="application/text; charset=utf8"
        )