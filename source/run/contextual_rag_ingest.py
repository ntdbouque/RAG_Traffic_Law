import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from source.rag.ingest import DocumentIngestionPipeline
from source.settings import setting


def load_parser():
    parser = argparse.ArgumentParser(description="Ingest data")
    parser.add_argument(
        "--folder_dir",
        type=str,
        help="Path to the folder containing the documents",
        default='./sample',
    )
    
    parser.add_argument(
        "--ingest_type",
        choices=["origin", "contextual", "both"],
        default='both'
    )
    return parser.parse_args()

def main():
    from icecream import ic
    args = load_parser()

    ingestor = DocumentIngestionPipeline(setting=setting)

    ic(args.folder_dir)
    ic(args.ingest_type)

    ingestor.run_ingest(folder_dir=args.folder_dir, type=args.ingest_type)


if __name__ == "__main__":
    main()
