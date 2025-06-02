from pydantic import BaseModel

class RAGType:
    """
    RAG type schema.

    Attributes:
        CONTEXTUAL (str): Contextual RAG type.
    """

    CONTEXTUAL = "contextual"

class DocumentMetadata(BaseModel):
    """
    Document metadata schema.

    Attributes:
        new_chunk (str): article preprend with contextualized article content
        chapter_id (str): chapter index number
        chapter_uuid (str): chapter uuid
        article_id (str): article index number 
        article_uuid (str): article uuid
        original_content (str): Original content of the document.
        contextualized_article_content (str): Contextualized content of the document which will be prepend to the original content.
    """
    new_chunk: str
    chapter_id: str
    chapter_uuid: str
    article_id: str
    article_uuid: str
    article_content: str
    contextualized_article_content: str

class ElasticSearchResponse(BaseModel):
    """
    ElasticSearch response schema.

    Attributes:
        doc_id (str): Document ID.
        content (str): Content of the document.
        contextualized_content (str): Contextualized content of the document.
        score (float): Score of the document.
    """

    doc_id: str
    original_content: str
    contextual_content: str
    article_id: str
    score: float


class QdrantPayload(BaseModel):
    '''
    Qdrant Payload Schema

    Attributes:
        chapter_uuid (str): ch
        new_chunk (str): a chunk is prepend with contextualized article content
        article_uuid (str): article uuid
    '''
    chapter_uuid: str
    text: str
    original_content: str
    article_uuid: str
    article_id: str
