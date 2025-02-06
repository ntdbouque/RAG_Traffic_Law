'''
Author: Nguyen Truong Duy
Purpose: Building a reader, ready to ingest data to Qdrant and ElasticSearch server
Latest Update: 27/01/2025
'''

import os
import re
from llama_index.core.schema import TextNode
from llama_index.core.extractors.metadata_extractors import TitleExtractor, KeywordExtractor
from llama_index.llms.openai import OpenAI
import sys
from pathlib import Path
import json
from llama_index.core import Document
from copy import deepcopy

sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

load_dotenv()

# metadata - tieu de, so cua luat, ngay thang an hanh, dia diem ban hanh


SPECIAL_CASE = ['phần', 'chương', 'mục', 'tiểu mục', 'điều']

METADATA_PROMPT = '''
Từ đoạn văn bản dưới đây, hãy trích xuất các thông tin sau một cách rõ ràng và chính xác:

1. Tiêu đề: tiêu đề của bộ luật (ví dụ Luật Dân Sự)
2. Số của luật: Số hiệu hoặc mã số của bộ luật  
3. Ngày tháng năm ban hành: Ngày, tháng, năm mà luật được ban hành hoặc thông qua (định dạng DD/MM/YY).  
4. Địa điểm ban hành: Địa điểm hoặc cơ quan ban hành luật (nếu có thông tin).  

Nếu không tìm thấy một trong các thông tin trên, hãy ghi rõ 'Không có thông tin'.

Văn bản:
'{context_str}'

Định dạng đầu ra:
- Tiêu đề: [Tiêu đề]
- Luật số: [Số hiệu]
- Ngày tháng năm ban hành: [Ngày tháng năm]
- Địa điểm ban hành: [Địa điểm]
'''

    
def pre_process(lst_lines: list[str]) -> list[str]:
    '''
    Preprocess list of lines, include convert to lower case, remove blank lines, remove page break,
    remove special character and reformat for separate chapter, return clean list of separate lines. 

    Args:
        lst_lines (List[str]): A list of lines in document is preprocessed 
    '''
    # lower case
    lst_lines= [line.lower() for line in lst_lines]
    
    # remove blank lines
    lst_lines = [line.strip() for line in lst_lines if line.strip()]
    
    # remove page break
    lst_lines = [line for line in lst_lines if not re.match(r'^-{3,}$', line)]
    
    # remove special character without heading
    for i, line in enumerate(lst_lines):
        if not any(f'# {special_case}' in line for special_case in SPECIAL_CASE) and line.startswith('#'):
            lst_lines[i] = line.replace('#', '').strip()
    
    # format chapter
    formated_lines = []
    i = 0
    while i < len(lst_lines):
        line = lst_lines[i]
        
        if re.match(r"# chương [ivxldm]+", line):
            chapter_title = line
            if i+1 < len(lst_lines) and not any(f"# {special_case}" in lst_lines[i+1] for special_case in SPECIAL_CASE):
                chapter_title += " " + ': ' + lst_lines[i+1].replace('#', '').strip()
                i += 1
            formated_lines.append(chapter_title)
        else:
            formated_lines.append(line)
        i += 1

    return formated_lines

def extract_document_metadata(lst_lines):
    '''
    Extract metadata from 7 first lines of document using Gpt-4o-mini, predefined prompt.
    Results are heading, law number, public date, public place. 

    Args:
        lst_lines: all lines in a document
    '''
    llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    intro = lst_lines[:7]
    intro = ' '.join(intro)
    node = TextNode(text=intro)
    
    metadata_extractor = KeywordExtractor(
                                    nodes=1,
                                    llm = llm,
                                    prompt_template = METADATA_PROMPT
                        )
    metadata = metadata_extractor.extract([node])
    
    raw_data = metadata[0]['excerpt_keywords']
    
    dct_metadata = {}
    for line in raw_data.split('\n'):
           if ':' in line:
                key,value = line.split(': ',1)
                key = key.replace('- ', '').strip()
                value = value.strip()
                dct_metadata[key] = value
           
    return dct_metadata

def extract_chapter_indices(lst_lines):
    '''
    Find indices which start of a chapter in a document

    Args:
        lst_lines: all lines from a document
    '''
    lst_chapters_indices = []
    pattern = r"# chương [ivxlcdm]+.*"
    for i, line in enumerate(lst_lines):
        if re.match(pattern, line):
            lst_chapters_indices.append(i)
    return lst_chapters_indices

def extract_chapter_content(lst_document_lines, chapter_indices):
    '''
    Find content of each chapter based on all lines of document and extracted chapter start indices

    Args:
        lst_document_lines: all lines of document
        chapter_indices: start indices of each chapter
    '''
    chapters_content = {}
    num_chapters = len(chapter_indices)

    for i in range(num_chapters):
        start = chapter_indices[i]
        end = chapter_indices[i+1] if i+1 < num_chapters else len(lst_document_lines)
        
        chapter_title = lst_document_lines[start]
        chapter_lines = lst_document_lines[start+1:end]
    
        article_indices = extract_article_indices(chapter_lines)
        article_contents = extract_article_contents(chapter_lines, article_indices)
        
        chapters_content[chapter_title] = article_contents
    
    return chapters_content 
        
def extract_article_indices(chapter_lines):
    '''
    Similar to extract_chapter_indices function. However, it extracts start indices of each artical, 
    which belong to a chapter.
    '''
    article_indices = []
    pattern = r"# điều \d+"
    for i, line in enumerate(chapter_lines):
        if re.match(pattern, line):
            article_indices.append(i)
    return article_indices

def extract_article_contents(chapter_lines, article_indices):
    '''
    Similar to extract_chapter_content function. However, it extract article content, which belong to a chapter
    '''
    article_contents = {}
    num_articles = len(article_indices)
    
    for i in range(num_articles):
        start = article_indices[i]
        end = article_indices[i+1] if i+1 < num_articles else len(chapter_lines)
        
        article_title = chapter_lines[start]
        article_lines = '\n'.join(chapter_lines[start+1:end])
        
        article_contents[article_title] = article_lines
    return article_contents
 
def split_documents(raw_documents):
    '''
    Split document into chunk following by section, extract document metadata
    
    Args: 
        raw_documents (List[str]): list of all line in document, which is not preprocessed

    Return:
        splitted_chapters (List[Document]): list of each chapter content
        splitted_articles (List[List[Document]]): List of list of each article, group by chapter.

    '''

    splitted_articles = []
    splitted_chapters = []
    for doc in raw_documents: # duyệt 
        lst_lines = doc.text.split('\n')
        
        lst_lines = pre_process(lst_lines) # step 2
        # with open('/workspace/competitions/Sly/RAG_tool/src/readers/section_reader/test.txt', 'w') as f:
        #     for line in lst_lines:
        #         f.write(line+'\n')
            
        chapter_indices = extract_chapter_indices(lst_lines)  # step 3.2
        
        document_metadata = extract_document_metadata(lst_lines) # step 3.1
        
        document_content = extract_chapter_content(lst_lines, chapter_indices) # step 4
        
        metadata = {}
        metadata['title'] = document_metadata['Tiêu đề']
        
        for chapter_title, chapter_content in document_content.items(): # loop chapter
            chapter_metadata = deepcopy(metadata)
            chapter_metadata['chapter_title'] = chapter_title
            lst_chapter = []
    
            chapter_text = '\n\n'.join(chapter_content.values())
            
            splitted_chapters.append(
                Document(
                    metadata=chapter_metadata, 
                    text=chapter_text
                )
            )
    
            for article_title, article_content in chapter_content.items(): # loop article
                article_metadata = deepcopy(metadata)
                article_metadata['chapter_title'] = chapter_title
                article_metadata['article_title'] = article_title
    
                node = Document(
                    metadata=article_metadata, 
                    text = article_content
                )
                lst_chapter.append(node)
                
            splitted_articles.append(lst_chapter)
                
    return splitted_chapters, splitted_articles
    
if __name__ == '__main__':
    # # Đường dẫn tệp
    # document_path = '/workspace/competitions/Sly/RAG_tool/logs/v31/parse/doc_0.txt'
    # output_json_path = '/workspace/competitions/Sly/RAG_tool/data/test_parser/chapters.json'

    # # Xử lý văn bản và trích xuất dữ liệu
    # result = split(document_path)

    # # Lưu vào file JSON
    # os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    # with open(output_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)

    # print(f"Dữ liệu đã được lưu vào {output_json_path}")
    
    documents = reformat('/workspace/competitions/Sly/RAG_tool/logs/v31/parse/doc_0.txt')
    print(documents)