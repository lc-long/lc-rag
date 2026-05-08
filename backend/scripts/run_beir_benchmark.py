"""Run BEIR benchmark on Nova-RAG's HybridRetriever.

Usage:
    cd backend && uv run python scripts/run_beir_benchmark.py

Datasets tested (5):
    - fiqa: Financial QA
    - scidocs: Scientific paper retrieval
    - scifact: Scientific claim verification
    - nfcorpus: Medical/nutrition retrieval
    - arguana: Argument retrieval
"""
import asyncio
import logging
import os
import sys
import json
from pathlib import Path

# Add backend to path so "from src.core" style imports work
_backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(_backend_dir))
os.chdir(_backend_dir)

from dotenv import load_dotenv
load_dotenv(".env")

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from src.core.retriever.beir_benchmark import NovaRAGRetriever
from src.core.retriever.bm25_index import BM25Indexer
from src.core.storage.vector_store import VectorStore
from src.core.embedder.aliyun_embedder import AliyunEmbedder
from src.core.retriever.evaluator import RAGEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nova_rag")

# BEIR datasets to test
BEIR_DATASETS = [
    "fiqa",
    "scidocs",
    "scifact",
    "nfcorpus",
    "arguana",
]

# Output dir for downloaded datasets
DATA_FOLDER = Path.home() / ".cache" / "beir" / "datasets"


async def init_retriever():
    """Initialize HybridRetriever with actual components."""
    embedder = AliyunEmbedder()
    vector_store = VectorStore()
    bm25_indexer = BM25Indexer()

    return NovaRAGRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_indexer=bm25_indexer,
    )


def download_dataset(dataset: str, data_folder: Path) -> bool:
    """Download and prepare a BEIR dataset."""
    try:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/garbe/ir-datasets/{dataset}"
        logger.info(f"Downloading {dataset} from {url}...")

        from beir.util import download_and_unzip
        download_and_unzip(url, str(data_folder))

        return True
    except Exception as e:
        logger.error(f"Failed to download {dataset}: {e}")
        return False


async def run_benchmark():
    """Run BEIR benchmark across multiple datasets."""
    logger.info("=" * 60)
    logger.info("Nova-RAG BEIR Benchmark Evaluation")
    logger.info("=" * 60)

    retriever = await init_retriever()
    evaluator = EvaluateRetrieval(retriever, k_values=[1, 3, 5, 10, 100])

    all_results = {}

    for dataset in BEIR_DATASETS:
        logger.info(f"\n--- Testing {dataset} ---")

        dataset_path = DATA_FOLDER / dataset

        # Try to load existing dataset, download if not found
        if not dataset_path.exists():
            if not download_dataset(dataset, DATA_FOLDER):
                logger.warning(f"Skipping {dataset} - download failed")
                continue

        try:
            # Load dataset using GenericDataLoader
            data_loader = GenericDataLoader(data_folder=str(dataset_path))
            corpus, queries, qrels = data_loader.load(split="test")

            logger.info(f"  Corpus: {len(corpus)} docs | Queries: {len(queries)} | Qrels: {len(qrels)}")

            if len(queries) == 0:
                logger.warning(f"  No queries found in {dataset}, skipping")
                continue

            # Run retrieval
            logger.info(f"  Running retrieval...")
            results = evaluator.retrieve(corpus, queries, top_k=100)

            # Evaluate
            ndcg, map, recall, precision = EvaluateRetrieval.evaluate(
                qrels, results,evaluator.k_values
            )

            logger.info(f"  Results:")
            for k in [1, 3, 5, 10]:
                logger.info(f"    NDCG@{k}: {ndcg.get(f'NDCG@{k}', 0):.4f}")
                logger.info(f"    Recall@{k}: {recall.get(f'Recall@{k}', 0):.4f}")

            all_results[dataset] = {
                "ndcg": {k: round(ndcg.get(f'NDCG@{k}', 0), 4) for k in [1, 3, 5, 10]},
                "map": round(map.get('MAP', 0), 4),
                "recall": {k: round(recall.get(f'Recall@{k}', 0), 4) for k in [1, 3, 5, 10]},
                "precision": {k: round(precision.get(f'P@{k}', 0), 4) for k in [1, 3, 5, 10]},
                "num_docs": len(corpus),
                "num_queries": len(queries),
            }

        except Exception as e:
            logger.error(f"  Error evaluating {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print("\n" + "=" * 60)
    print("BEIR BENCHMARK SUMMARY")
    print("=" * 60)

    for dataset, metrics in all_results.items():
        print(f"\n{dataset.upper()}:")
        print(f"  NDCG@10: {metrics['ndcg'][10]:.4f}")
        print(f"  MAP: {metrics['map']:.4f}")
        print(f"  Recall@10: {metrics['recall'][10]:.4f}")
        print(f"  (Corpus: {metrics['num_docs']} docs, {metrics['num_queries']} queries)")

    # Calculate averages
    if all_results:
        avg_ndcg10 = sum(r['ndcg'][10] for r in all_results.values()) / len(all_results)
        avg_map = sum(r['map'] for r in all_results.values()) / len(all_results)
        avg_recall10 = sum(r['recall'][10] for r in all_results.values()) / len(all_results)

        print(f"\n--- AVERAGE ACROSS {len(all_results)} DATASETS ---")
        print(f"  Avg NDCG@10: {avg_ndcg10:.4f}")
        print(f"  Avg MAP: {avg_map:.4f}")
        print(f"  Avg Recall@10: {avg_recall10:.4f}")

        # Save to JSON for resume use
        output = {
            "datasets_tested": list(all_results.keys()),
            "average_metrics": {
                "ndcg@10": round(avg_ndcg10, 4),
                "map": round(avg_map, 4),
                "recall@10": round(avg_recall10, 4),
            },
            "per_dataset": all_results,
        }

        output_path = Path(__file__).parent.parent / "beir_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    asyncio.run(run_benchmark())