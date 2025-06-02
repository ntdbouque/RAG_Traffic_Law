import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from source.rag.ingest import DocumentIngestionPipeline
from source.settings import setting

def main(folder_path: str):
    """
    Khởi chạy quá trình ingestion từ thư mục chứa các file CSV
    """
    ingestor = DocumentIngestionPipeline(setting)
    ingestor.run_ingest_from_csv(folder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ingest data from CSV to Qdrant and Elasticsearch")
    parser.add_argument(
        "--csv_folder",
        type=str,
        required=True,
        help="Đường dẫn tới thư mục chứa các file .csv"
    )

    args = parser.parse_args()
    main(args.csv_folder)
