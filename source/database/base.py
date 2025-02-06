'''
Author: Nguyen Truong Duy
Purpose: Định nghĩa một lớp BaseVectorDatabase, sử dụng module ABC trong Python. 
Giúp xây dựng một chuẩn chung cho các vector database, bao gồm:
    - kiểm tra tính tồn tại của collection,
    - kiểm tra kết nối đến server, 
    - tạo collection,
'''

from abc import ABC, abstractmethod

from typing import List, Dict, Any

class BaseVectorDatabase(ABC):
    @abstractmethod
    def check_collection_exists(self, collection_name: str):
        '''
        Check if the collection exists

        Args:
            collection_name (str): Collection name to check
        '''
        pass
    
    @abstractmethod
    def test_connection(self):
        '''
        Test Connection with the server 
        '''

    @abstractmethod
    def create_collection(self, collection_name, vector_size: int):
        '''
        Create a new collection

        Args:
            collection_name (str): Collection name
            vector_size (int): vector size
        '''
        pass

