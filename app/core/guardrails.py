"""Deterministic input and output safety checks.

This module blocks obvious prompt-injection attempts, secret-revealing requests,
and generated answers that look like credentials before they reach users.
"""

from dataclasses import dataclass, field
import re
from typing import Any


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


PROMPT_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(all\s+)?(previous|prior|above)\s+instructions\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\s+(dan|developer|system|root)\b", re.IGNORECASE),
    re.compile(r"\bbypass\s+(the\s+)?(safety|guardrails|policy|instructions)\b", re.IGNORECASE),
]

SECRET_REQUEST_PATTERN = re.compile(
    r"\b(reveal|show|print|dump|expose|leak|tell\s+me|give\s+me|what\s+is)\b"
    r".{0,80}\b(api[_\s-]?key|secret|password|token|private\s+key|system\s+prompt|developer\s+message)\b",
    re.IGNORECASE | re.DOTALL,
)

SENSITIVE_OUTPUT_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"-----BEGIN (OPENSSH|RSA|DSA|EC|PRIVATE) PRIVATE KEY-----"),
    re.compile(r"\b[A-Z0-9_]*(API_KEY|SECRET|PASSWORD|TOKEN)\s*[:=]\s*['\"]?[^'\"\s]+", re.IGNORECASE),
]


def validate_question(question: str) -> GuardrailResult:
    normalized = " ".join(question.strip().split())
    if not normalized:
        return _blocked("empty_question", "Please ask a non-empty question.")

    if _has_excessive_control_chars(question):
        return _blocked(
            "invalid_control_characters",
            "I can't process that request because it contains unsupported control characters.",
        )

    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(normalized):
            return _blocked(
                "prompt_injection",
                "I can't follow requests that try to override system or developer instructions.",
            )

    if SECRET_REQUEST_PATTERN.search(normalized):
        return _blocked(
            "secret_exfiltration",
            "I can't reveal secrets, credentials, system prompts, or developer messages.",
        )

    return GuardrailResult(allowed=True)


def validate_answer(answer: str) -> GuardrailResult:
    for pattern in SENSITIVE_OUTPUT_PATTERNS:
        if pattern.search(answer):
            return _blocked(
                "sensitive_output",
                "I generated content that looked like a secret, so I blocked the response.",
            )
    return GuardrailResult(allowed=True)


def _has_excessive_control_chars(value: str) -> bool:
    if not value:
        return False
    control_count = sum(1 for char in value if ord(char) < 32 and char not in "\n\r\t")
    return control_count > 0


def _blocked(reason: str, message: str) -> GuardrailResult:
    return GuardrailResult(
        allowed=False,
        reason=reason,
        message=message,
        metadata={"guardrail_reason": reason},
    )
