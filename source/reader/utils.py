'''
Author: Nguyen Truong Duy
Purpose: Usage function in reading documents
Latest Update: 27/01/2025
'''

def check_valid_extension(file_path: str | Path) -> bool:
    '''
    Check if the file extension is supported

    Args:
        file_path (str): File path to check
    Returns:
        bool: True if the file extension is supported, False otherwise
    '''
    return Path(file_path).suffix in SUPPORTED_FILE_EXTENSIONS


def get_files_from_folders(folder_dir: str) -> list[str]:
    '''
    Get valid files from folder directory

    Args:
        folder_dir (str): Path to folder directory
    Returns:
        list[str]: List of valid file paths
    '''

    files = []

    if Path(folder_dir).is_dir():
        files.extend(
            [
                str(file_path.resolve())
                for file_path in Path(folder_dir).rglob('*')
                if check_valid_extension(file_path) 
            ]
        )

    return files

def get_extractor():
    return {
        '.pdf': LLamaParse(
            result_type = 'markdown', api_key = os.getenv('LLAMA_PARSE_API_KEY'),
            split_by_page = 'False',
            continous_mode = True,
            max_pages = 10
        )
    }