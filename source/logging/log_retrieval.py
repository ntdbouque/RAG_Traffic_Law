import os
import time
from icecream import ic

LOGS_DIR = './logs'

def ensure_log_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)

    log_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
    log_folder = os.path.join(LOGS_DIR, log_time)
    os.makedirs(log_folder, exist_ok=True)
    return log_folder

def save_contextual_results(log_folder, contextual_results):
    if not contextual_results:
        return
    contextual_results.sort(key=lambda x: x.score, reverse=True)
    lines = [
        f'{i+1}:\nID: {result.id_}\nScore: {result.score}\nText: {result.text}\n' + ('-'*100)
        for i, result in enumerate(contextual_results)
    ]
    with open(os.path.join(log_folder, 'contextual_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def save_bm25_results(log_folder, bm25_results):
    if not bm25_results:
        return
    bm25_results.sort(key=lambda x: x.score, reverse=True)
    lines = [
        f'{i+1}:\nID: {result.doc_id}\nScore: {result.score}\nText: {result.original_content}\n' + ('-'*100)
        for i, result in enumerate(bm25_results)
    ]
    with open(os.path.join(log_folder, 'bm25_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def save_combined_results(log_folder, combined_results):
    if not combined_results:
        return
    combined_results.sort(key=lambda x: x.score, reverse=True)
    lines = [
        f'{i+1}:\nID: {result.node.id_}\nScore: {result.score}\nText: {result.text}\n' + ('-'*100)
        for i, result in enumerate(combined_results)
    ]
    with open(os.path.join(log_folder, 'combined_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def save_reranked_results(log_folder, reranked_results):
    if not reranked_results:
        return
    reranked_results.sort(key=lambda x: x.score, reverse=True)
    lines = [
        f'{i+1}:\nID: {result.node.id_}\nScore: {result.score}\nText: {result.node.text}\n' + ('-'*100)
        for i, result in enumerate(reranked_results)
    ]
    with open(os.path.join(log_folder, 'reranked_results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def save_query_and_response(log_folder, query, response):
    if query:
        with open(os.path.join(log_folder, 'query.txt'), 'w', encoding='utf-8') as f:
            f.write(query.query_str)
    if response:
        with open(os.path.join(log_folder, 'response.txt'), 'w', encoding='utf-8') as f:
            f.write(response)

def log_retrieval(contextual_results=None, bm25_results=None, combined_results=None, reranked_results=None, query=None, response=None):
    log_folder = ensure_log_dir()
    save_contextual_results(log_folder, contextual_results)
    save_bm25_results(log_folder, bm25_results)
    save_combined_results(log_folder, combined_results)
    save_reranked_results(log_folder, reranked_results)
    save_query_and_response(log_folder, query, response)
    ic(f'Logging retrieval done in {log_folder}')
