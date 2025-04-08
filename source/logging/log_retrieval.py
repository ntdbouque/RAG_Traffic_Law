import sys
import os
from pathlib import Path
import time
from icecream import ic
import json

def log_retrieval(contextual_results, bm25_results, combined_results, reranked_results, query, response):    
    logs_dir = 'logs/retrieval'
    os.makedirs(logs_dir, exist_ok=True)
    
    log_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    log_folder = os.path.join(logs_dir, log_time)
    os.makedirs(log_folder, exist_ok=True)
    
    def preprocess_reranked_results(reranked_results): #list[NodeWithScore]
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return [f'{i+1}:\nScore: {result.score}\nText:' + result.node.text + '\n' + ('-'*100) for i, result in enumerate(reranked_results)]
    
    def preprocess_contextual_results(contextual_results): # list[NodeWithScore]
        contextual_results.source_nodes.sort(key=lambda x: x.score, reverse=True)
        return [f'{i+1}:\nScore: {result.score}\nText:' + result.text + '\n'+ ('-'*100) for i, result in enumerate(contextual_results.source_nodes)]
        
    def preprocess_bm25_results(bm25_results): # list[ElasticSearchResponse]
        bm25_results.sort(key=lambda x: x.score, reverse=True)
        return [f'{i+1}:\nScore: {result.score}\nText:' + result.original_content + '\n'+ ('-'*100) for i, result in enumerate(bm25_results)]
    
    def preprocess_combined_results(combined_results): # list[NodeWithScore]
        combined_results.sort(key=lambda x: x.score, reverse=True)
        return [f'{i+1}:\nScore: {result.score}\nText:' + result.text + '\n'+ ('-'*100) for i, result in enumerate(combined_results)]
    
    log_content = {
        'contextual_results.txt': preprocess_contextual_results(contextual_results),
        'bm25_results.txt': preprocess_bm25_results(bm25_results),
        'combined_results.txt': preprocess_combined_results(combined_results),
        'reranked_results.txt': preprocess_reranked_results(reranked_results),
        'query.txt': [query],
        'response.txt': [response]
    }
    
    for filename, contents in log_content.items():
        file_path = os.path.join(log_folder, filename)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(contents))
    
    ic(f'Logging retrieval done in {log_folder}')
    
if __name__ == '__main__':
    log_retrieval('retrieved_chunks', 'bm25_chunks', 'combined_chunks', 'response')
