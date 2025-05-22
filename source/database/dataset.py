from typing import Dict, List, Tuple
from llama_index.core.bridge.pydantic import BaseModel
import json
from llama_index.core.llama_dataset.legacy.embedding import EmbeddingQAFinetuneDataset

# get all node from qdrant database:
def get_nodes_from_collection(collection_name: str):
    '''
    Get all nodes from a qdrant collection
    Args:
        collection_name: a qdrant collection name
    '''
    client = QdrantClient(url="http://localhost:6333")

    qdrant_nodes, _ = client.scroll(
        collection_name="contextual_rag_nckh",
        limit=1489
    )
    nodes = []
    for node in qdrant_nodes:
        nodes.append(TextNode(text=node.payload['text'], id_=node.id))
    return nodes

class LegalQuery(BaseModel):
    id: str
    query: str
    intent: str  # e.g., "lookup", "situation", "procedure", "comparison", "learning"
    complexity: int  # 1 to 3
    law_id: str  # reference to a law chunk or article


class LegalQueryEvalDataset(EmbeddingQAFinetuneDataset):
    """
    Extension of EmbeddingQAFinetuneDataset to support intent, complexity, and law_id.
    """
    queries: Dict[str, LegalQuery]  # enriched queries

    @property
    def query_docid_pairs(self) -> List[Tuple[str, List[str]]]:
        return [
            (q.query, self.relevant_docs[qid])
            for qid, q in self.queries.items()
        ]

    def to_flat_queries(self) -> Dict[str, str]:
        return {qid: q.query for qid, q in self.queries.items()}

    def to_base_format(self) -> EmbeddingQAFinetuneDataset:
        return EmbeddingQAFinetuneDataset(
            queries=self.to_flat_queries(),
            corpus=self.corpus,
            relevant_docs=self.relevant_docs,
            mode=self.mode
        )

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.dict(), f, indent=4, ensure_ascii=False)


    @classmethod
    def from_json(cls, path: str) -> "LegalQueryEvalDataset":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Parse enriched queries
        queries_raw = {
            qid: LegalQuery(**qdata) for qid, qdata in data["queries"].items()
        }

        return cls(
            queries=queries_raw,
            corpus=data["corpus"],
            relevant_docs=data["relevant_docs"],
            mode=data.get("mode", "text")
        )
