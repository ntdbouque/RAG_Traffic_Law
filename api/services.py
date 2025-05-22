'''
Author: Nguyen Truong Duy
Purpose: Building a Chatbot for API service
Latest Update: 03-03-2025
'''

import os
import logging 
from icecream import ic
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.agent import AgentRunner
from llama_index.core.tools import FunctionTool

from starlette.responses import StreamingResponse, Response

from source.tools.contextual_rag_tools import (
    load_contextual_rag_tool
)
from source.tools.location_search_tools import load_location_search_tool

from source.constants import (
    SERVICE,
    TEMPERATURE,
    MODEL_ID,
    STREAM,
    AGENT_TYPE,
)

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
        location_search_tool = load_location_search_tool()
        return [contextual_rag_tool]

    # def add_tools(self, tools: FunctionTool | list[FunctionTool]) -> None:
    #     """
    #     Add more tools to the agent.

    #     Args:
    #         tools (FunctionTool | list[FunctionTool]): A single tool or a list of tools to add to the agent.
    #     """
    #     if isinstance(tools, FunctionTool):
    #         tools = [tools]

    #     self.tools.extend(tools)
    #     ic(f"Add: {len(tools)} tools.")

    #     self.query_engine = (
    #         self.create_query_engine()
    #     )  # Re-create the query engine with the new tools

    def load_model(self, service, model_id):
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
        if service == 'openai':
            return OpenAI(
                model=model_id,
                temperature=TEMPERATURE,
                api_key = os.getenv('OPENAI_API_KEY')
                )
        else:
            raise NotImplementedError(f'Service {service} is not supported')


    def create_query_engine(self):
        '''
        Create a query engine and confige it for routing queries to approciate tools
        
        This method initializes and configures a query engine for routing queries to specialize tool based on the query type
        It loads a language model along with specific tools for specific tasks such as code search or paper search

        Return:
            AgentRunner: An instance of AgentRunner configured with the neccessary tools and settings
        '''

        llm = self.load_model(SERVICE, MODEL_ID)
        Settings.llm = llm

        ic(llm)
        if AGENT_TYPE == 'openai':
            query_engine = OpenAIAgent.from_tools(
                tools = self.tools, verbose = True, llm = llm
            )
        else:
            raise ValueError(f'Agent type {AGENT_TYPE} is not supported')
        
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

        if STREAM:
            streaming_response = self.query_engine.stream_chat(prompt)

            async def async_generator():
                for chunk in streaming_response.response_gen:
                    yield chunk

            return StreamingResponse(
                async_generator(), 
                media_type='application/text; charset=utf8'
            )
        else:
            response = self.query_engine.chat(prompt)
            return Response(
                content=response.response,  
                media_type="application/text; charset=utf8"
            )