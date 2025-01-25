"""
Author: Nguyen Truong Duy
Purpose: định nghĩa lớp QdrantVectorDatabase kế thừa từ lớp BaseVectorDatabase, gồm các phương thức:
- kiểm tra kết nối Qdrant Server
- kiểm tra sự tồn tại của collection
- tạo collection
- thêm vector
- thêm nhiều vector
- xoá vector
- xoá collection
"""
from qdrant_client.http import models
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from typing import List, Dict, Any
import logging
from tenacity import (
    retry,
    wait_fixed,
    after_log,
    before_sleep_log,
    stop_after_attempt,
    retry_if_exception_type
)

from base import BaseVectorDatabase

logger = logging.getLogger(__file__)


class QdrantVectorDatabase(BaseVectorDatabase):
    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_fixed(5),
        after=after_log(logger, logging.DEBUG),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
        retry=retry_if_exception_type(ConnectionError)
    )

    def __init__(self, url: str, distance: str = models.Distance.COSINE) -> None:
        self.url = url,
        self.client = QdrantClient(url)
        self.distance = distance
        self.test_connection()

        logger.info('Qdrant Client initialized succesfully')

    def test_connection(self):
        '''
        Test the connection with the Qdrant Server
        '''
        try:
            self.client.get_collections()
        except:
            raise ConnectionError('Qdrant Connection Failed')

    def check_collection_exists(self, collection_name):
        return self.client.collection_exists(collection_name)

    def create_collection(self, collection_name: str, vector_size: int):
        if not self.check_collection_exists(collection_name):
            logger.info(f'Creating Collection {collection_name}')
            self.client.create_collection(
                collection_name, 
                vectors_config=models.VectorParams(
                    size=vector_size, distance=self.distance
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=0
                ),
                # Quantization document: https://qdrant.tech/documentation/guides/quantization/#:~:text=Binary%20quantization%20is%20an%20extreme%20case%20of%20scalar,the%20memory%20footprint%20by%20a%20factor%20of%2032.
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True),
                ),
            )

    def add_vector(
        self,
        collection_name: str,
        vector_id: str,
        vector: List[Float],
        payload: QdrantPayload,
    ):
        '''
        Add a vector to the collection

        Args:
            collection_name (str): collection name to add
            vector_id (str): Vector ID
            vector (List[float]): vector embedding
            payload (QdrantPayload): Payload for the vector
        '''
        if not self.check_collection_exists(collection_name):
            self.create_collection(collection_name, len(vector))
        
        self.client.upsert(
            collection_name = collection_name,
            points=[
                models.PointStruct(
                    id=vector_id,
                    payload=payload.model_dump(),
                    vector=vector
                )
            ]
        )

    def add_vectors(
        self,
        collection_name: str,
        vector_ids: list[str],
        vectors: list[list[float]],
        payloads: list[QdrantPayload],
    ):
        '''
        Add multiples vectors to the collection

        Args:
            collection_name (str): collection name to add
            vectors (List[List[float]]): List of vector to add
            vector_ids (List[str]): list of vector id
            payloads (List(QdrantPayload)): List of Qdrant Payload
        '''
        if not self.check_collection_exists(collection_name):
            self.create_collection(collection_name, len(vectors[0]))        

        points = [
            models.PointStruct(
                id = vector_id,
                payload = payload.model_dump(),
                vector = vector
            )
            for vector_id, vector, payload in zip(vector_ids, vectors, payloads)
        ]
        
        self.client.upsert(collection_name=collection_name, points=points)
        
    def delete_vector(self, collection_name: str, document_id:str | UUID):
        '''
        Delete a vector from the collection 

        Args:
            collection_name (str)
            document_id (str|UUID
        '''
        document_id = str(document_id)
        if not self.check_collection_exists(collection_name):
            logger.debug(f"Collection {collection_name} does not exist")
            return

        logger.debug(
            "collection_name: %s - document_id: %s", collection_name, document_id
        )

        self.client.delete(
            collection_name,
            point_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FilterSelector(
                            key='document_id',
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )

    def delete_collection(self, collection_name: str):
        """
        Delete a collection

        Args:
            collection_name (str): Collection name to delete
        """
        if not self.check_collection_exists(collection_name):
            logger.debug(f"Collection {collection_name} does not exist")
            return

        success = self.client.delete_collection(collection_name)

        if success:
            logger.debug(f"Collection {collection_name} deleted successfully!")

    
if __name__ == '__main__':
    url = "http://localhost:6333"
    db = QdrantVectorDatabase(url)
    