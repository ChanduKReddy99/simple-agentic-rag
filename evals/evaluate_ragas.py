"""RAGAS evaluation runner for the Agentic RAG workflow.

This file runs the agent against a golden dataset, scores answers and retrieved
contexts, and writes JSON reports for local, CI, or nightly quality checks.
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

from app.agent.graph import AgenticRAGGraph
from app.core.config import get_settings
from app.core.logging import configure_logging, log_detail, log_section, log_step_finish, log_step_start
from app.rag.embeddings import build_embeddings
from app.rag.retriever import RetrieverService
from app.rag.vectorstore import build_vectorstore


DEFAULT_DATASET_PATH = Path("evals/golden_dataset.json")
DEFAULT_REPORT_PATH = Path("reports/ragas_report.json")
DEFAULT_REPORTS_DIR = Path("reports")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for the local Agentic RAG pipeline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-type", choices=["local", "ci", "nightly"], default="local")
    parser.add_argument("--timestamped-report", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, run_name="ragas_evaluation")
    output_path = args.output or _default_output_path(args.run_type, timestamped=args.timestamped_report)

    payload = run_ragas_evaluation(
        dataset_path=args.dataset,
        output_path=output_path,
        limit=args.limit,
        run_type=args.run_type,
        show_progress=args.show_progress,
    )
    print(json.dumps(payload["scores"], indent=2))
    print(f"Saved detailed RAGAS results to {output_path}")


def run_ragas_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_path: Path | None = None,
    limit: int | None = None,
    run_type: str = "local",
    timestamped_report: bool = False,
    show_progress: bool = False,
) -> dict[str, Any]:
    settings = get_settings()
    output_path = output_path or _default_output_path(run_type, timestamped=timestamped_report)
    log_detail("run_type", run_type)
    log_detail("dataset", str(dataset_path))
    log_detail("report", str(output_path))

    log_step_start("load_eval_cases")
    records = _load_eval_records(dataset_path=dataset_path, limit=limit)
    log_step_finish("load_eval_cases", f"cases={len(records)}")

    log_step_start("build_rag_agent")
    agent = _build_agent()
    log_step_finish("build_rag_agent")

    log_step_start("generate_answers", f"cases={len(records)}")
    evaluated_records = [_run_case(agent, record) for record in records]
    log_step_finish("generate_answers")

    log_step_start("load_ragas_dependencies")
    Dataset, evaluate, metric_bundle = _load_ragas_dependencies()
    log_step_finish("load_ragas_dependencies")

    dataset = Dataset.from_list(evaluated_records)
    metrics = _select_metrics(records, metric_bundle)
    log_detail("metrics", ", ".join(_metric_name(metric) for metric in metrics))

    log_step_start("run_ragas_metrics")
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=agent.llm,
        embeddings=build_embeddings(settings),
        raise_exceptions=False,
        show_progress=show_progress,
    )
    log_step_finish("run_ragas_metrics")

    payload = {
        "metadata": {
            "run_type": run_type,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": str(dataset_path),
            "case_count": len(records),
            "metrics": [_metric_name(metric) for metric in metrics],
            "chat_model": settings.openai_chat_model,
            "embedding_model": settings.openai_embedding_model,
            "retrieval_k": settings.retrieval_k,
        },
        "scores": _result_to_dict(result),
        "records": evaluated_records,
    }
    log_step_start("write_report", str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_step_finish("write_report", str(output_path))
    return payload

def _load_ragas_dependencies():
    try:
        from datasets import Dataset
        from ragas import evaluate

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Importing .* from 'ragas.metrics' is deprecated.*",
                category=DeprecationWarning,
            )
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation dependencies are not installed correctly. "
            "Run `uv sync --group eval` and try again. "
            f"Original import error: {exc}"
        ) from exc

    return Dataset, evaluate, {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }

def _build_agent() -> AgenticRAGGraph:
    settings = get_settings()
    embeddings = build_embeddings(settings)
    vectorstore = build_vectorstore(settings, embeddings)
    retriever = RetrieverService(vectorstore=vectorstore, settings=settings)
    return AgenticRAGGraph(retriever=retriever, settings=settings)


def _load_eval_records(dataset_path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")

    raw_records = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise ValueError("Golden dataset must be a JSON array of eval cases")

    records = raw_records[:limit] if limit is not None else raw_records
    if not records:
        raise ValueError("At least one eval case is required")
    for index, record in enumerate(records, start=1):
        if not record.get("question"):
            raise ValueError(f"Eval case {index} is missing question")
    return records


def _select_metrics(records: list[dict[str, Any]], metric_bundle: dict[str, Any]) -> list[Any]:
    metrics = [
        metric_bundle["faithfulness"],
        metric_bundle["answer_relevancy"],
    ]
    if all(record.get("ground_truth") for record in records):
        metrics.extend(
            [
                metric_bundle["context_precision"],
                metric_bundle["context_recall"],
            ]
        )
    return metrics


def _run_case(agent: AgenticRAGGraph, record: dict[str, Any]) -> dict[str, Any]:
    log_step_start("rag_case", record["question"])
    result = agent.invoke(record["question"])
    evaluated = {
        "id": record.get("id"),
        "user_input": record["question"],
        "response": result.get("answer", ""),
        "retrieved_contexts": result.get("contexts", []),
        "tags": record.get("tags", []),
    }
    if record.get("ground_truth"):
        evaluated["reference"] = record["ground_truth"]
    log_detail("route", result.get("route", "unknown"))
    log_detail("contexts", len(evaluated["retrieved_contexts"]))
    log_step_finish("rag_case", record["question"])
    return evaluated


def _result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
        numeric_columns = frame.select_dtypes(include="number").columns
        return {column: float(frame[column].mean()) for column in numeric_columns}
    if isinstance(result, dict):
        return result
    return dict(result)


def _metric_name(metric: Any) -> str:
    return getattr(metric, "name", metric.__class__.__name__)


def _default_output_path(run_type: str, timestamped: bool = False) -> Path:
    if not timestamped:
        return DEFAULT_REPORT_PATH
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_REPORTS_DIR / f"ragas_{run_type}_{timestamp}.json"


if __name__ == "__main__":
    main()
