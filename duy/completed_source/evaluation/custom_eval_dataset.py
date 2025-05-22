'''
Author: Nguyễn Trường Duy
Purpose: multiple class for Legal Query Eval Dataset
Update: 21/05/2024
'''

from typing import Dict, List, Tuple
from llama_index.core.bridge.pydantic import BaseModel
import json


class LegalQuery(BaseModel):
    '''
    an instance of query
    '''
    query: str
    intent: str  # e.g., "lookup", "situation", "procedure", "comparison", "learning"
    complexity: int  # 1 to 3

class LegalQueryEvalDataset(BaseModel):
    queries: Dict[str, LegalQuery]
    corpus: Dict[str, str]
    relevant_docs: Dict[str, List[str]]
    mode: str = "text"

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.dict(), f, indent=4, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
