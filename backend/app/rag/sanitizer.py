"""Retrieved document text sanitizer and prompt delimiter wrapper.

Trade-off: Regex instruction sanitization prevents indirect prompt injections at ingestion boundary.
Explicit <<<DOCUMENT_CONTEXT>>> delimiters isolate external untrusted tokens from LLM system rules.
"""
import re

INSTRUCTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\b"),
    re.compile(r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\b"),
    re.compile(r"(?i)\bforget\s+(all\s+)?(previous|prior)\b"),
    re.compile(r"(?i)^\s*(system|user|assistant|human|ai)\s*:"),
    re.compile(r"(?i)<\s*\|\s*(im_start|im_end|system|user|assistant)\s*\|\s*>"),
    re.compile(r"(?i)\[\s*(system|inst|instruction)\s*\]"),
    re.compile(r"(?i)\byou\s+are\s+now\b"),
    re.compile(r"(?i)\bnew\s+instructions\s*:"),
    re.compile(r"(?i)\b(override|bypass)\s+rules\b"),
    re.compile(r"(?i)\b(act\s+as|pretend\s+to\s+be)\s+(an?\s+)?(unfiltered|jailbroken|admin|developer|evil)\b"),
    re.compile(r"(?i)\bprint\s+(your|the)\s+(system\s+)?prompt\b"),
]


def is_instruction_line(line: str) -> bool:
    """Check if a line matches known prompt injection or adversarial instruction patterns."""
    stripped = line.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in INSTRUCTION_PATTERNS)


def sanitize_document_text(
    text: str,
    filename: str = "document",
    page_number: int | None = None,
) -> str:
    """Strip adversarial instruction lines and wrap surviving text in clear context delimiters."""
    lines = text.splitlines()
    clean_lines = [ln for ln in lines if not is_instruction_line(ln)]
    cleaned_body = "\n".join(clean_lines).strip()

    page_suffix = f" p.{page_number}" if page_number is not None else ""
    header = f"<<<DOCUMENT_CONTEXT: [{filename}{page_suffix}]>>>"
    footer = "<<<END_DOCUMENT_CONTEXT>>>"

    return f"{header}\n{cleaned_body}\n{footer}"
