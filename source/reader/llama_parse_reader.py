'''
Author: Nguyen Truong Duy
Purpose: Parse toàn bộ file bằng LLamaParser
Latest Update: 06/02/2025
'''

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from tqdm import tqdm
from icecream import ic
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
from source.reader.utils import get_files_from_folder, get_extractor

def parse_multiple_files(folder_dir: str) -> list[Document]:
    '''
    Read the content of multiple files

    Args:
        folder_dir: Path to folder directory containing files
    Returns:
        list[Document]: List of document from all files
    '''

    valid_files = get_files_from_folder(folder_dir)
    ic(valid_files)
    file_extractor = get_extractor()

    documents = SimpleDirectoryReader(
        input_files = valid_files,
        file_extractor = file_extractor,
    ).load_data(show_progress=True)
    
    return documents

# if __name__ == '__main__':
#     folder_dir = r'C:\Users\duy\Desktop\TrafficLaw\sample'
#     documents = parse_multiple_files(folder_dir)
#     print(documents)