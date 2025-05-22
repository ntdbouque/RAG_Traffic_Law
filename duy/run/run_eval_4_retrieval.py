import os
import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from dotenv import load_dotenv

import asyncio
from duy.completed_source.evaluate import evaluate_retrieval, save_json
from duy.completed_source.custom_dataset import LegalQueryEvalDataset
from duy.completed_source.baseline_retrieval import get_base_retriever, get_gpt_reranker

import nest_asyncio
nest_asyncio.apply()

load_dotenv(override=True)

if __name__ == '__main__':
    async def main():
        qa_dataset = LegalQueryEvalDataset.base_from_json("duy/data/eval_dataset_4_retrieval.json")
        print("Length of queries eval:", len(qa_dataset.queries))
        metrics = ["hit_rate", "mrr"]
        retriever = get_base_retriever()
        gpt_reranker = get_gpt_reranker()
        node_postprocessors = [gpt_reranker]
        des_results_json = './duy/data/eval_results_4_retrieval.json'

        eval_results = await evaluate_retrieval(qa_dataset, metrics, retriever, node_postprocessors)
        save_json(eval_results, des_results_json)

    asyncio.run(main())