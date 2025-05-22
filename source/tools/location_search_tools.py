import sys
from typing import List
from icecream import ic
from pathlib import Path
from dotenv import load_dotenv
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from llama_index.core.tools import FunctionTool
from llama_index.core import get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from llama_index.core.data_structs import Node
from llama_index.llms.openai import OpenAI

# from source.settings import setting
from source.database.elastic import ElasticSearch
from source.schemas import ElasticSearchResponse
from source.settings import setting
load_dotenv()

def load_location_search_tool():
    llm = OpenAI(model='gpt-4o-mini')
    response_synthesizer = get_response_synthesizer(llm = llm, response_mode="context_only")
    es_client = ElasticSearch(
        url = setting.elastic_search_url,
        index_name = setting.elastic_search_index_name
    )
    def retrieve_content(query: str) -> List[str]:
        '''
        A useful function to retrieve chunk revelant to query describing position of chunk and response
        Args:
            query (str): The query describing position of a chunk to answer.
        Returns:
            List[str]: a list of chunk revalent to position query
        '''
        bm25_results:List[ElasticSearchResponse] = []
        bm25_results = es_client.search_by_location(query, k=3)
        ic(bm25_results)
        #lst_original_contents = [result.original_content for result in bm25_results]
        nodes = [NodeWithScore(node=Node(text=result.original_content), score=result.score) for result in bm25_results]
        response = response_synthesizer.synthesize(
            query, nodes=nodes
        )
        return response
    
    return FunctionTool.from_defaults(
        fn = retrieve_content,
        return_direct = True,
        description = 'A useful to retrieve content and generate response based on the query describing position of chunk in the document'
    )
