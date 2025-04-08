'''
Author: Tran Van Luong, Nguyen Duy
Purpose: Evaluate the performance of Retriever system
'''


import os
import sys
import uuid
import json
from tqdm import tqdm
from icecream import ic
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from source.rag.retrieval import RetrievalPipeline
from source.settings import setting as ConfigSetting
from source.rag.retrieval import RetrievalPipeline
from source.reader.llama_parse_reader import parse_multiple_files
from source.rag.ingest import split_documents

from llama_index.core.extractors.metadata_extractors import KeywordExtractor
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from random import sample
from typing import List

from source.constants import (
    METADATA_PROMPT,
)


class Evaluator():
    '''
    Evaluator class to evaluate the performance of RAG system
    '''
    def __init__(self, setting: ConfigSetting): # type: ignore
        self.setting = setting
        ic(self.setting)    
        self.retrieval_pipeline = RetrievalPipeline(self.setting)
        self.llm = OpenAI(model=self.setting.model_name)
        self.keyword_extractor = KeywordExtractor(
                        nodes=1,
                        llm = self.llm,
                        prompt_template = METADATA_PROMPT
                )
    
    def postprocess_output(self, response: str) -> List[str]:
        return [i.strip() for i in response.split('\n')]
        
    def generate_questions(self, splitted_articles, prompt_template: str):
        lst_articles = []
    
        chapter_id = 1
        for chapter_lv in splitted_articles:
            article_id = 1
            for article_lv in chapter_lv:
                dct_article = {
                    'article_content': article_lv.text,
                    'article_id': f'chapter{chapter_id}_article{article_id}'
                }
                lst_articles.append(dct_article)
                
                article_id += 1
            chapter_id += 1
            
            
        dct_QA = {}
        selected_laws = sample(lst_articles, 50) 
        for dct_article in tqdm(selected_laws):
            messages = [
                ChatMessage(
                    role='system',
                    content="tôi đang cần đánh giá khả năng truy xuất văn bản trên tài liệu luật giao thông đường bộ của hệ thống Traffic Law RAG, bạn có thể giúp tôi được hay không",
                ),
                ChatMessage(
                    role='user',
                    content=prompt_template.format(law_text=dct_article['article_content']),
                    )
            ]
            response = self.llm.chat(messages).message.content
            lst_questions = self.postprocess_output(response)
            dct_QA[dct_article['article_id']] = lst_questions
            

        with open('source/evaluator/questions.json', 'w') as f:
            json.dump(dct_QA, f, indent=4, ensure_ascii=False)
            

    def query_rag_system(self, questions: List[str], rag_system, groundtruth: List[str]):
        results = []
        for i, question in enumerate(questions):
            top_answers = RetrievalPipeline.hybrid_rag_search(question)[:3] 
            is_correct = any(answer in groundtruth[i] for answer in top_answers)
            result = {
                "question_id": i,
                "question": question,
                "answers": top_answers,
                "groundtruth": groundtruth[i],
                "true_or_false": is_correct
            }
            results.append(result)
        return results

    def save_results_to_json(self, results, filename='evaluation_results.json'):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)

    def create_synthesis_dataset(self, prompt_template: str, folder_dir: str):
        raw_documents = parse_multiple_files(folder_dir)
        split_chapter, split_articles = split_documents(self.keyword_extractor, raw_documents)
        self.generate_questions(split_articles, prompt_template)
        
    def gpt_evaluate(self):
        #results = self.query_rag_system(questions, self.retrieval_pipeline, groundtruth)
        #self.save_results_to_json(results)
        pass
        
if __name__ == '__main__':
    prompt_template = '''cho tôi 5 câu hỏi để đánh giá khả năng truy xuất của hệ thống truy vấn dữ liệu luật giao thông dựa vào đoạn văn bản bên dưới: \n
                        {law_text}
                        Định dạng đầu ra:
                        câu 1: [câu 1]
                        câu 2: [câu 2]
                        câu 3: [câu 3]
                        câu 4: [câu 4]
                        câu 5: [câu 5]
                        vui lòng không đưa ra bất kì thông tin nào khác
                        '''
    folder_dir = "/workspace/competitions/Sly/RAG_Traffic_Law/sample"
    setting = ConfigSetting
    evaluator = Evaluator(setting)
    evaluator.create_synthesis_dataset(prompt_template, folder_dir)


