
def retrieval(self, query) -> Response:
    '''
    Search the query with the contextual RAG
    
    Args:
        query (str): The query string
        k (int): The number of documents to return  
    ''' 
    
    semantic_results = self.contextual_search(query, self.k)

    
    query_bundle = QueryBundle(query)

    new_nodes = self.reranker_gpt.postprocess_nodes(
        combined_nodes[0:10], query_bundle
    )

    
