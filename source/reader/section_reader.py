'''
Author: Nguyen Truong Duy
Purpose: Building a reader, ready to ingest data to Qdrant and ElasticSearch server
Latest Update: 09/02/2025
'''

import os
import re
from llama_index.core.schema import TextNode
from typing import List, Tuple
import sys
from pathlib import Path
import json
from llama_index.core import Document
from copy import deepcopy
from icecream import ic

sys.path.append(str(Path(__file__).parent.parent.parent))

from source.constants import (
    METADATA_PROMPT,
    SPECIAL_CASE
)
from dotenv import load_dotenv

load_dotenv()

import fitz  

def pre_process(lst_lines: list[str]) -> list[str]:
    '''
    Preprocess list of lines, include convert to lower case, remove blank lines, remove page break,
    remove special character and reformat for separate chapter, return clean list of separate lines. 

    Args:
        lst_lines (List[str]): A list of lines in document is preprocessed 
    '''
    lst_lines= [line.lower() for line in lst_lines]
    lst_lines = [line.strip() for line in lst_lines if line.strip()]
    lst_lines = [line for line in lst_lines if not re.match(r'^-{3,}$', line)]
    
    seen_articles = set()
    for i, line in enumerate(lst_lines):
        match = re.match(r"^điều\s+(\d+)", line, re.IGNORECASE)
        if match:
            article_number = match.group(1)  
            if article_number in seen_articles:
                continue  
            seen_articles.add(article_number)
            lst_lines[i] = "# " + line  
        elif "điều" in line and not re.search(r"^điều\s+\d+", line, re.IGNORECASE):
            continue 
    for i, line in enumerate(lst_lines):
        if not any(f'# {special_case}' in line for special_case in SPECIAL_CASE) and line.startswith('#'):
            lst_lines[i] = line.replace('#', '').strip()
    
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

def extract_document_metadata(metadata_extractor, lst_lines):
    '''
    Extract metadata from 7 first lines of document using Gpt-4o-mini, predefined prompt.
    Results are heading, law number, public date, public place. 

    Args:
        lst_lines: all lines in a document
    '''
    intro = lst_lines[:7]
    intro = ' '.join(intro)
    node = TextNode(text=intro)
    
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
 
def split_documents(extractor, raw_documents):
    '''
    Split document into chunk following by section, extract document metadata
    
    Args: 
        extractor: a keyword extractor of LLamaIndex
        raw_documents (List[str]): list of all line in document, which is not preprocessed

    Return:
        splitted_chapters (List[Document]): list of each chapter content
        splitted_articles (List[List[Document]]): List of list of each article, group by chapter.

    '''

    splitted_articles = []
    splitted_chapters = []
    for doc in raw_documents:  
        lst_lines = doc.text.split('\n')
        lst_lines = pre_process(lst_lines) 
            
        chapter_indices = extract_chapter_indices(lst_lines) 
        
        document_metadata = extract_document_metadata(extractor, lst_lines) 
        
        document_content = extract_chapter_content(lst_lines, chapter_indices) 
        
        metadata = {}
        metadata['title'] = document_metadata['Tiêu đề']
        
        for chapter_title, chapter_content in document_content.items(): 
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
    
            for article_title, article_content in chapter_content.items():
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

    
################# --- TEST PDF PARSE --- #################
def read_pdf_to_text(pdf_path: str) -> list[str]:
    """
    Read text from a PDF file and return as a list of lines.
    """
    doc = fitz.open(pdf_path)
    lines = []
    for page in doc:
        text = page.get_text("textblocks")  
        lines.extend(text.split("\n"))
    return lines

def convert_to_markdown(lines: list[str]) -> str:
    """
    Convert preprocessed text lines to Markdown format, handling both chapters and articles.
    
    Args:
        lines (list[str]): List of text lines from a legal document.

    Returns:
        str: Markdown formatted text.
    """
    markdown_lines = []
    lines= [line.lower() for line in lines]
    for i, line in enumerate(lines):
        chapter_match = re.match(r"^(Chương\s+[IVXLCDM]+)(\.?)\s*(.*)", line, re.IGNORECASE)
        if chapter_match:
            chapter_number, dot, chapter_title = chapter_match.groups()
            if chapter_title.strip():
                markdown_lines.append(f"# {chapter_number}: {chapter_title.strip()}") 
            else:
                markdown_lines.append(f"# {chapter_number}") 
            continue  

        article_match = re.match(r"^(Điều\s+\d+)(\.?)\s*(.*)", line, re.IGNORECASE)
        if article_match:
            article_number, dot, content = article_match.groups()
            if dot and content.strip():  
                markdown_lines.append(f"## {article_number}. {content.strip()}") 
            elif dot:  
                markdown_lines.append(f"## {article_number}.") 
            else:  
                markdown_lines.append(line) 
            continue  
        markdown_lines.append(line)

    return "\n".join(markdown_lines)


def parse_pdf(pdf_path: str, output_txt_path: str):
    lines = read_pdf_to_text(pdf_path)
    markdown_text = convert_to_markdown(lines)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

def read_txt_to_lines(txt_path: str) -> list[str]:
    """
    Đọc tệp TXT và trả về danh sách các dòng văn bản.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()] 

def split_docs(lines: list[str]):
    """
    Trích xuất danh sách các chương và điều luật từ văn bản.
    
    Trả về:
        - splitted_chapters (list[Document]): Danh sách các chương
        - splitted_articles (list[list[Document]]): Danh sách các điều luật theo chương
    """
    splitted_chapters = []
    splitted_articles = []

    current_chapter = None
    current_chapter_articles = []
    chapter_metadata = {}

    for i, line in enumerate(lines):
        chapter_match = re.match(r"^# (Chương\s+[IVXLCDM]+)", line, re.IGNORECASE)
        article_match = re.match(r"^## (Điều\s+\d+)\.?\s*(.*)", line, re.IGNORECASE)

        if chapter_match:  
            if current_chapter:
                splitted_chapters.append(Document(metadata=chapter_metadata, text=current_chapter))
                splitted_articles.append(current_chapter_articles)

            chapter_number = chapter_match.group(1)
            chapter_title = lines[i + 1].strip() if (i + 1 < len(lines)) else ""

            chapter_metadata = {"chapter_title": f"{chapter_number}: {chapter_title}"}
            current_chapter = f"{chapter_number}: {chapter_title}" 
            current_chapter_articles = []

        elif article_match and current_chapter: 
            article_number = article_match.group(1)
            article_title = article_match.group(2).strip() if article_match.group(2) else ""

            article_metadata = deepcopy(chapter_metadata)
            article_metadata["article_title"] = f"{article_number}. {article_title}"

            article_content = []
            j = i + 1
            while j < len(lines) and not lines[j].startswith("##"): 
                article_content.append(lines[j])
                j += 1

            current_chapter_articles.append(Document(metadata=article_metadata, text="\n".join(article_content)))

    if current_chapter:
        splitted_chapters.append(Document(metadata=chapter_metadata, text=current_chapter))
        splitted_articles.append(current_chapter_articles)

    return splitted_chapters, splitted_articles

def process_txt(txt_path: str):
    """
    Xử lý tệp TXT và trả về danh sách các chương và điều luật.
    """
    lines = read_txt_to_lines(txt_path)
    return split_docs(lines)


if __name__ == '__main__':
    from source.reader.llama_parse_reader import parse_multiple_files
    raw_documents = parse_multiple_files(folder_dir=r'/workspace/competitions/Sly/RAG_Traffic_Law/sample')# list[Document, Document]

    from llama_index.core.extractors.metadata_extractors import KeywordExtractor
    from llama_index.llms.openai import OpenAI

    llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    metadata_extractor = KeywordExtractor(
                                nodes=1,
                                llm = llm,
                                prompt_template = METADATA_PROMPT
                        )

    splitted_chapters, splitted_articles = split_documents(metadata_extractor, raw_documents)
    ic(splitted_articles)
    output_file = "/workspace/competitions/Sly/RAG_Traffic_Law/sample/output2.txt"
    with open(output_file, 'w') as f:
        for articles in splitted_articles:
            for article in articles:
                f.write(f"{article.metadata['chapter_title']}\n")
                f.write(f"{article.metadata['article_title']}\n")
                f.write(f"{article.text}\n\n")
    
    
    ### --- TEST PDF PARSE --- ###
    # input_pdf = "/workspace/competitions/Sly/RAG_Traffic_Law/sample/23_2008_QH12_82203.pdf"  
    # output_txt = "/workspace/competitions/Sly/RAG_Traffic_Law/sample/splitted_articles_2.txt"
    # parse_pdf(input_pdf, output_txt)
    # splitted_chapters, splitted_articles = process_txt(output_txt)
    # ic(splitted_articles)
    # output_split_chapter = "/workspace/competitions/Sly/RAG_Traffic_Law/sample/output_split_chapters_2.txt"
    # with open(output_split_chapter, 'w') as f:
    #     for articles in splitted_chapters:
    #         f.write(f"{articles.metadata['chapter_title']}\n")
    # output_split_article = "/workspace/competitions/Sly/RAG_Traffic_Law/sample/output_split_articles_2.txt"
    # with open(output_split_article, 'w') as f:
    #     for articles in splitted_articles:
    #         for article in articles:
    #             f.write(f"{article.metadata['chapter_title']}\n")
    #             f.write(f"{article.metadata['article_title']}\n")
    #             f.write(f"{article.text}\n\n")

    
    
    