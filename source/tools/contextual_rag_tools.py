import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from llama_index.core.tools import FunctionTool

from source.rag.retrieval import RetrievalPipeline
from source.settings import setting

def load_contextual_rag_tool():
    retrieval_pipeline = RetrievalPipeline(setting)
    
    def answer_query(query: str) -> str:
        '''
        Answer a query using the contextual RAG.
        Args:
            query (str): The query to answer.
        Returns:
            str: The answer to the query.
        '''
        return retrieval_pipeline.hybrid_rag_search(query)
    
    return FunctionTool.from_defaults(
        fn = answer_query,
        return_direct = True,
        description = 'A useful to answer query using Traffic Law Augmented Retrieval'
    )
