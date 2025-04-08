'''
Author: Nguyen Truong Duy
Purpose: 
    + Kiểm tra file extension được hỗ trợ
    + Lấy tất cả những valid file từ folder đưa vào
Latest Update: 06/02/2025
'''
import os
import sys
from icecream import ic
from pathlib import Path
from dotenv import load_dotenv
from typing import Union

sys.path.append(str(Path(__file__).parent.parent.parent))
from source.constants import SUPPORTED_FILE_EXTENSIONS
from llama_index.readers.llama_parse import LlamaParse

load_dotenv()

def check_valid_extension(file_path: Union[str, Path]) -> bool:
    '''
    Check if the file extension is supported

    Args:
        file_path (str): File path to check
    Returns:
        bool: True if the file extension is supported, False otherwise
    '''
    return Path(file_path).suffix in SUPPORTED_FILE_EXTENSIONS

def get_files_from_folder(files_or_folder: str) -> list[str]:
    '''
    Get valid files from folder directory

    Args:
        folder_dir (str): Path to folder directory
    Returns:
        list[str]: List of valid file paths
    '''

    files = []
    for file_or_folder in files_or_folder:  
        if Path(file_or_folder).is_dir():
            for file_path in Path(folder_dir).rglob('*'):
                if check_valid_extension(file_path):
                    files.append(str(file_path.resolve()))
        else:
            if check_valid_extension(file_or_folder):
                files.append(str(Path(file_or_folder).resolve()))
            else:
                ic(f"File extension not supported: {file_or_folder}")
    return files

def get_extractor():
    return {
        '.pdf': LlamaParse(
            result_type = 'markdown', api_key = os.getenv('LLAMA_PARSE_API_KEY'),
            split_by_page = False,
            num_workers = 9,
            max_pages = 100,
        )
    }

if __name__ == '__main__':
    folder_dir = r'C:\Users\duy\Desktop\TrafficLaw\sample'
    files = get_files_from_folders(folder_dir)
    print(files)