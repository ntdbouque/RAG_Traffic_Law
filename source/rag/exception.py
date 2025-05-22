### Payload
'''
chapter_uuid: str # không để làm gì, sinh ra chơi == None

text: str # contextual content + original_content
article_uuid: str # uuid
article_id: str # định vị vị trí

'''
### action:
'''
doc_id # uuid
article_id # định vị vị trí
original_content
contextual_content
'''

import os
import sys
from pathlib import Path
import uuid
from tqdm import tqdm
sys.path.append(str(Path(__file__).parent.parent.parent))

from source.rag.ingest import DocumentIngestionPipeline
from source.settings import setting
from source.schemas import DocumentMetadata

import pandas as pd
from llama_index.core import Document 

ingestor = DocumentIngestionPipeline(setting)

def format_data_to_ingest(file_path):
    contextual_documents : list[Document] = []
    contextual_documents_metadata: list[DocumentMetadata] = []
    df = pd.read_csv(file_path)
    
    for i, row in tqdm(df.iterrows(), total = df.shape[0]):
        chunk = row['Chunk']
        context = row['Context']
        position = row['Position']
        # dieu = 
        # khoan = 
        
        new_chunk = context + '\n' + chunk
        article_uuid = str(uuid.uuid4())
        
        
        contextual_documents.append(
            Document(
                    text=new_chunk,
                    metadata = dict(
                        chapter_id = '',
                        chapter_uuid = '',
                        article_id = position,
                        article_uuid = article_uuid,
                        article_content = chunk,    
                        contextualized_article_content = ''
                    )
                )
        )
        
        contextual_documents_metadata.append(
            DocumentMetadata(
                        new_chunk = new_chunk,
                        chapter_id = '',
                        chapter_uuid = '',
                        article_id = position,
                        article_uuid = article_uuid,
                        article_content = chunk,
                        contextualized_article_content = context,
                ),
        )

    return contextual_documents, contextual_documents_metadata

if __name__ == '__main__':
    contextual_documents, contextual_documents_metadata = format_data_to_ingest(file_path='/workspace/competitions/Sly/RAG_Traffic_Law_experiment/data/finalll/35_2024.csv')
    ingestor.ingest_data_qdrant(contextual_documents)
    ingestor.ingest_data_elastic(contextual_documents_metadata)
    
    contextual_documents, contextual_documents_metadata = format_data_to_ingest(file_path='/workspace/competitions/Sly/RAG_Traffic_Law_experiment/data/finalll/36_2025 (1).csv')
    ingestor.ingest_data_qdrant(contextual_documents)
    ingestor.ingest_data_elastic(contextual_documents_metadata)
    
    contextual_documents, contextual_documents_metadata = format_data_to_ingest(file_path='/workspace/competitions/Sly/RAG_Traffic_Law_experiment/data/finalll/100_2019_ND_CP.csv')
    ingestor.ingest_data_qdrant(contextual_documents)
    ingestor.ingest_data_elastic(contextual_documents_metadata)
    
    contextual_documents, contextual_documents_metadata = format_data_to_ingest(file_path='/workspace/competitions/Sly/RAG_Traffic_Law_experiment/data/finalll/160_2024 (3).csv')
    ingestor.ingest_data_qdrant(contextual_documents)
    ingestor.ingest_data_elastic(contextual_documents_metadata)
    
    contextual_documents, contextual_documents_metadata = format_data_to_ingest(file_path='/workspace/competitions/Sly/RAG_Traffic_Law_experiment/data/finalll/161_2024 (2).csv')
    ingestor.ingest_data_qdrant(contextual_documents)
    ingestor.ingest_data_elastic(contextual_documents_metadata)
    
    contextual_documents, contextual_documents_metadata = format_data_to_ingest(file_path='/workspace/competitions/Sly/RAG_Traffic_Law_experiment/data/finalll/165_2024.csv')
    ingestor.ingest_data_qdrant(contextual_documents)
    ingestor.ingest_data_elastic(contextual_documents_metadata)
    
    contextual_documents, contextual_documents_metadata = format_data_to_ingest(file_path='/workspace/competitions/Sly/RAG_Traffic_Law_experiment/data/finalll/168_2024.csv')
    ingestor.ingest_data_qdrant(contextual_documents)
    ingestor.ingest_data_elastic(contextual_documents_metadata)
    