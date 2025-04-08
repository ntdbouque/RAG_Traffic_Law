import sys
import json
from tqdm import tqdm
from icecream import ic
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
load_dotenv(override=True)

from source.settings import setting

from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage


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

llm = OpenAI(model=setting.model_name)
    
def gpt_evaluate(json_path: str, output_path: str):
    '''
    Evaluate the generated questions based on the groundtruth and save to output_path.
    
    Args:
        json_path (str): The path to the JSON file containing the generated questions and groundtruth.
        output_path (str): The path to save the evaluation results.
    '''
    
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    lst_qa_gt_results = []
        
    for dct_gt_qa in data:
        idx = dct_gt_qa['idx']
        questions = dct_gt_qa['question']
        answers = dct_gt_qa['answer']
        groundtruth = dct_gt_qa['groundtruth']
        for question, answer in zip(questions, answers):
            messages = [
                ChatMessage(
                    role='system',
                    content=SYS_EVAL_PROMPT
                ),
                ChatMessage(
                    role='user',
                    content=EVAL_PROMPT.format(question=question, answer=answer, reference=groundtruth)
                )
            ]
            response = llm.chat(messages).message.content

            lst_qa_gt_results.append({
                'idx': idx,
                'question': question,
                'answer': answer,
                'groundtruth': groundtruth,
                'gpt eval': response
            })
        
    with open(output_path, 'w') as f:
        json.dump(lst_qa_gt_results, f, indent=4, ensure_ascii=False)  
    ic('Output saved to', output_path)
        
if __name__ == '__main__':
    json_file = '/workspace/competitions/Sly/RAG_Traffic_Law/source/evaluator/dict/evaluation_results.json'
    output_path = '/workspace/competitions/Sly/RAG_Traffic_Law/source/evaluator/dict/evaluation_gpt.json'
    gpt_evaluate(json_file, output_path)