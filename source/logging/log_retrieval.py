'''
Author: Nguyen Truong Duy
Purpose: This module is used to retrieve logs for utilization in the analysis (retrieval)
Latest update: 06-03-2025
'''

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time
from icecream import ic
import json

def log_retrieval(contextual_results, bm25_results, combined_results, reranked_results, query, response):
    os.makedirs('logs', exist_ok=True)
    os.makedirs('logs/retrieval', exist_ok=True)
    
    log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    log_path = os.path.join('logs/retrieval', f'{log_time}.json')
    
    
    def preprocess_reranked_results(reranked_results):
        '''
        An utility function to preprocess the reranked results
        '''
        lst_reranked_text = []
        for result in reranked_results:
            text = result.text
            lst_reranked_text.append(text)
        return lst_reranked_text
    
    def preprocess_contextual_results(contextual_results):
        '''
        An utility function to preprocess the contextual results
        '''
        lst_contextual_text = []
        for result in contextual_results.source_nodes:
            text_result = result.text
            lst_contextual_text.append(text_result)
        return lst_contextual_text
        
    def preprocess_bm25_results(bm25_results):
        '''
        An utility function to preprocess the bm25 results
        '''
        lst_bm25_text = []
        for result in bm25_results:
            text = result.original_content
            lst_bm25_text.append(text)
        return lst_bm25_text
    
    def preprocess_combined_results(combined_results):
        '''
        An utility function to preprocess the combined results
        '''
        lst_combined_text = []
        for result in combined_results:
            text = result.text
            lst_combined_text.append(text)
        return lst_combined_text
    
    log_content = {
        'contextual_results': preprocess_contextual_results(contextual_results),
        'bm25_results': preprocess_bm25_results(bm25_results),
        'combined_results': preprocess_combined_results(combined_results),
        'reranked_results': preprocess_reranked_results(reranked_results),
        'query': query,
        'response': response
    }                                              
    
    with open(log_path, 'w') as json_file:
        json.dump(log_content, json_file, indent=4, ensure_ascii=False) 
        
    
    ic('logging retrieval done')
    
if __name__ == '__main__':
    log_retrieval('retrieved_chunks', 'bm25_chunks', 'combined_chunks', 'response')