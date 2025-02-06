'''
Author: Nguyen Truong Duy
Purpose: Parsing multiple files in specific documents
Latest Update: 27/01/2025
'''

def parse_multiple_files(folder_dir: str) -> list[Document]:
    '''
    Read the content of multiple files

    Args:
        folder_dir: Path to folder directory containing files
    Returns:
        list[Document]: List of document from all files
    '''

    valid_files = get_files_from_folder(folder_dir)

    file_extractor = get_extractor()