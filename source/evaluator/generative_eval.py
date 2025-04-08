'''
Purpose:This script is used to generate questions for evaluating the RAG system on the Traffic Law dataset and evaluation Traffic Law RAG
        It uses the OpenAI API to generate questions based on a given prompt template.
Author: Truong Duy/Tran Van Luong
Last modified:  07-03-2025
'''
import sys
import json
from tqdm import tqdm
from icecream import ic
from pathlib import Path
from dotenv import load_dotenv
import requests

sys.path.append(str(Path(__file__).parent.parent.parent))
load_dotenv(override=True)

from source.settings import setting
from source.database.elastic import ElasticSearch

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

import json
import random
from tqdm import tqdm


QUESTION_GENERATION_PROMPT = '''
Hãy tạo 3 câu hỏi để đánh giá khả năng truy xuất thông tin của hệ thống truy vấn dữ liệu luật giao thông, dựa trên đoạn văn bản sau:

{law_text}

Định dạng đầu ra:
câu 1: [câu hỏi 1]
câu 2: [câu hỏi 2]
câu 3: [câu hỏi 3]

Chỉ trả về các câu hỏi theo đúng định dạng trên, không thêm bất kỳ thông tin nào khác.
'''

SYS_EVAL_PROMPT = '''
Bạn là một trợ lý chuyên đánh giá khả năng truy xuất thông tin của hệ thống Retrieval-Augmented Generation (RAG).
Tôi sẽ cung cấp một câu hỏi, một câu trả lời do hệ thống RAG tạo ra, và một đoạn văn bản tham chiếu.
Nhiệm vụ của bạn là đánh giá xem câu trả lời có đủ bao quát và chính xác theo nội dung của đoạn tham chiếu hay không.
'''

EVAL_PROMPT = '''
Câu hỏi: {question}
Câu trả lời: {answer}
Đoạn tham chiếu: {reference}

Dựa trên đoạn tham chiếu, câu trả lời có đủ bao quát và chính xác để đáp ứng câu hỏi không?
Vui lòng chỉ trả lời [Có] hoặc [Không].
Ví dụ: Có
'''
    

class QuestionGenerator:
    def __init__(self, output_path = None): # type: ignore 
        '''
        Initialize the QuestionGenerator with the given retriever and prompt template.
        
        Args:
            retriever (ElasticSearchRetriever): An instance of ElasticSearchRetriever for fetching documents
            prompt_template (str): The template used to prompt the model for generating questions
        '''
        self.setting = setting
        self.llm = OpenAI(model=self.setting.model_name)
        ic(self.llm)
        self.es_client = ElasticSearch(self.setting.elastic_search_url, self.setting.elastic_search_index_name)
        self.output_path = output_path
        
    def generate_questions(self, num_articles: int = 5):
        '''
        Generate questions for randomly selected documents from ElasticSearch.

        Args:
            num_questions (int): The number of documents to generate questions for.

        Returns:
            dict: A dictionary where keys are article IDs and values are lists of generated questions.
        '''
        lst_gt_ques = []
        
        def get_random_original_content():
            random_node = random.choice(self.es_client.get_all_nodes())
            return random_node['_source']['original_content']
        
        
        for i in tqdm(range(num_articles), desc="Generating questions..."):
            original_content = get_random_original_content()
            messages = [
                ChatMessage(
                    role='system',
                    content="Tôi đang cần đánh giá khả năng truy xuất văn bản trên tài liệu luật giao thông đường bộ của hệ thống Traffic Law RAG, bạn có thể giúp tôi được hay không?"
                ),
                ChatMessage(
                    role='user',
                    content=QUESTION_GENERATION_PROMPT.format(law_text=original_content)
                )
            ]
            response = self.llm.chat(messages).message.content
            lst_questions = self.postprocess_output(response)
            
            doc_id = f'document_{i}'
            dct_QA = {
                'doc_id': doc_id,
                'questions': lst_questions,
                'gt': original_content,
            }
            
            lst_gt_ques.append(dct_QA)
        
        return lst_gt_ques

    def postprocess_output(self, output: str):
        '''
        Post-process the generated output into a list of questions.
        
        Args:
            output (str): The raw output from the language model.

        Returns:
            list: A list of questions extracted from the output.
        '''
    
        return [question.strip() for question in output.split('\n') if question.strip()]
    
    def generate_eval_data(self, num_articles: int = 5):
        '''
        Perform query retrieval and evaluation for a number of generated questions.

        Args:
            num_questions (int): The number of questions to generate and evaluate.

        Returns:
            list: A list of dictionaries containing questions, answers, and groundtruth.
        '''
        
        def send_query(text, api_key=None):
            headers = {"Content-Type": "application/json"}
            data = {"message": text, "api_key": api_key}
            resp = requests.post(
                "http://localhost:4045/v1/complete", json=data, headers=headers
            )
            return resp.text
        
        results = []
        lst_gt_ques = self.generate_questions(num_articles=num_articles)  
        ic(lst_gt_ques)
        ic(len(lst_gt_ques))

        for dct_gt_ques in tqdm(lst_gt_ques, total=len(lst_gt_ques), desc="Querying RAG system)"):
            answers = []

            for question in dct_gt_ques['questions']:
                answer = send_query(question)
                answers.append(answer)
                
            results.append({
                "idx": dct_gt_ques['doc_id'],
                "question": dct_gt_ques['questions'],
                "answer": answers,
                "groundtruth":dct_gt_ques['gt']
            })
            
        with open(self.output_path, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
           

if __name__ == "__main__":
    output_path = '/workspace/competitions/Sly/RAG_Traffic_Law/source/evaluator/dict/evaluation_results.json'
    question_generator = QuestionGenerator(output_path=output_path)

    question_generator.generate_eval_data(num_articles=10)
    
