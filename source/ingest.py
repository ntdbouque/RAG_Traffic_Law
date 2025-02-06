'''
Author: Nguyen Truong Duy
Purpose: Ingest Data to Qdrant and ElasticSearch server
Latest Update: 27/01/2025
'''

def add_contextual_content(
    chapter: Document, 
    articles: list[Document],
    doc_id: int
)

def run_ingest(
    folder_dir: str | Path,
    type: Literal['origin, contextual', 'both'] = 'contextual'
) -> None:
    '''
    Run the ingest process for Retrieval Augmented Generation System

    Args:
        folder_dir (str | Path): The folder directory containing documents
        type: Literal['origin, contextual', 'both']: The type to ingest. Default `contextual`
    '''

    raw_documents = parse_multiple_files(folder_dir)

    splitted_chapters, splitted_articles = split_documents(raw_documents)

    contextual_documents, contextual_documents_metadata = 

