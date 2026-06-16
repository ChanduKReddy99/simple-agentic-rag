"""Small structured logging helpers used by scripts and the API.

The helpers keep console/file logs readable by marking major workflow sections,
step starts, step finishes, and key details during ingestion, evaluation, and
chat execution.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
_CURRENT_LOG_PATH: Path | None = None


def configure_logging(level: str = "INFO", run_name: str | None = None, log_to_file: bool = True) -> Path | None:
    global _CURRENT_LOG_PATH
    if _CURRENT_LOG_PATH is not None:
        if run_name:
            log_section(run_name)
            log_detail("log_file", str(_CURRENT_LOG_PATH))
        return _CURRENT_LOG_PATH

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_path = _build_log_path(run_name) if log_to_file and run_name else None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    for handler in handlers:
        handler.addFilter(_NoiseFilter())

    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    if run_name:
        log_section(run_name)
        if log_path is not None:
            log_detail("log_file", str(log_path))
    _quiet_noisy_dependency_loggers()
    _CURRENT_LOG_PATH = log_path
    return log_path


def log_section(name: str) -> None:
    logging.info("==== %s ====", name)


def log_step_start(step: str, detail: str | None = None) -> None:
    message = f"[START] {step}"
    if detail:
        message = f"{message} | {detail}"
    logging.info(message)


def log_step_finish(step: str, detail: str | None = None) -> None:
    message = f"[DONE]  {step}"
    if detail:
        message = f"{message} | {detail}"
    logging.info(message)


def log_step_skip(step: str, reason: str) -> None:
    logging.info("[SKIP]  %s | %s", step, reason)


def log_detail(name: str, value: str | int | float | bool | None) -> None:
    logging.info("        %s: %s", name, value)


def _build_log_path(run_name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in run_name.lower())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{safe_name}_{timestamp}.log"


def _quiet_noisy_dependency_loggers() -> None:
    for logger_name in ("httpx", "httpcore", "openai", "chromadb"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


class _NoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        noisy_messages = (
            "LLM returned 1 generations instead of requested",
        )
        return not any(noisy_message in message for noisy_message in noisy_messages)
