'''
Author: Nguyen Truong Duy
Purpose: Read file csv va dinh dang ve cau truc list[Document] -> document, list[list[Document]] - chunk
Latest Update: 19/03/2025
'''

import os
import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).parent.parent.parent))


from icecream import ic
from typing import List
from llama_index.core import Document
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


context_prompt = 'tôi sẽ cung cấp cho bạn một điều trích từ một chương của một bộ luật/nghị định/thông tư. Tôi cần bạn cung cấp chunking cho tôi \
thành những phần nhỏ hơn ví dụ như theo điều, theo khoản (tối đa 300 từ cho mỗi chunk nhỏ), đồng thời tôi cần bạn cho tôi biết vị trí của chunk đó trong \
toàn một điều. Cấu trúc câu trả lời sẽ như sau:\
Chunk: {chunk}\
Locate: {location}\
Ví dụ:\
Chunk: {nội dung}\
Locate: \
- Điểm: a,b,c,d (nếu không có thì N/A)\
- Khoản: 1\
- Điều: 3\
'

chunk_prompt = 'Dưới đây là một nội dung của một điều\
{chunk}. Hãy giữ nguyên nội dung và không chỉnh sửa bất kì điều gì'


def reformat_into_small_chunk_level(file_path, output_path):
    client = OpenAI(os.getenv('OPENAI_API_KEY'))
    df=pd.read_csv(file_path)
    
    def preprocess_chunks(chunks):
        return lst_preprocessed_chunks
        
    for i , row in df.iterrows():
        chunk = row['chunk']
        title = row['chunk_title']

        chunk = chunk + '\n' + title
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": context_prompt
                },
                {
                    "role": "user",
                    "content": chunk_prompt.format(chunk = chunk)
                }
            ]
        )

        chunks = completion.choices[0].message.content
    
    return new_file_path

def parse_and_format_csv_chunk_lv(file_pth: str):
    '''
        Read the content of a csv file and format it to list[Document], list[list[Document]] in small chunk level

    Args:
        file_path: Path to the csv file
    Returns:
        list[Document]: List of document from the csv file
        list[list[Document]]: List of chunks of document
    '''
    

def parse_and_format_csv(file_path: str):
    '''
    Read the content of a csv file and format it to list[Document], list[list[Document]]

    Args:
        file_path: Path to the csv file
    Returns:
        list[Document]: List of document from the csv file
        list[list[Document]]: List of chunks of document
    '''
    df = pd.read_csv(file_path)
    ic(df.shape)
    documents = []
    
    splitted_documents: List[Document] = []
    splitted_chunks: List[List[Document]] = []
    
    unique_context_df = df['context'].unique()

    # Duyệt qua các context duy nhất để tạo các Document
    for context in unique_context_df:
        document_lv_row = df[df['context'] == context].iloc[0]
        
        document_lv_node = Document(
            text=document_lv_row['context'],
            metadata={
                'chapter_title': document_lv_row['context_title'],
            }
        )
        splitted_documents.append(document_lv_node)
        
        # Tạo danh sách chunk cho từng document
        chunk_list: List[Document] = []
        for _, chunk_lv_row in df[df['context'] == context].iterrows():
            chunk_lv_node = Document(
                text=chunk_lv_row['chunk'],
                metadata={
                    'article_title': chunk_lv_row['chunk_title'],
                    'chapter_title': chunk_lv_row['context_title'],
                    'ten_luat': 'Quy Dinh Ve Hoat Dong Dao Tao Va Sat Hach Lai Xe',
                    'so_hieu': '1160/2024/NĐ-CP',
                    'loai_van_ban': 'Nghi Dinh',
                    'noi_ban_hanh': 'Chinh Phu',
                    'nguoi_ky': 'Tran Hong Ha',
                    'ngay_ban_hanh': '18/12/2024',
                    'ngay_hieu_luc': '01/01/2025',
                    'ngay_cong_bao': '07/01/2025',
                    'so_cong_bao': '23-24',
                    'tinh_trang': 'con_hieu_luc'
                }
            )
            chunk_list.append(chunk_lv_node)
        
        # Thêm list chunk vào splitted_chunks
        splitted_chunks.append(chunk_list)

    return splitted_documents, splitted_chunks