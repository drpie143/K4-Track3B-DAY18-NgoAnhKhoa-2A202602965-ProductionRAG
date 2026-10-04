from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

_API_KEY_VALID = True


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full", "combined"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    global _API_KEY_VALID
    if OPENAI_API_KEY and _API_KEY_VALID:
        try:
            from openai import OpenAI, AuthenticationError
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            content = resp.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            if "AuthenticationError" in type(e).__name__ or "401" in str(e):
                _API_KEY_VALID = False
            print(f"  ⚠️  OpenAI summarize fallback: {e}")

    # Extractive fallback (không cần API)
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    return ". ".join(sentences[:2]) + "." if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    global _API_KEY_VALID
    if OPENAI_API_KEY and _API_KEY_VALID:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi bằng tiếng Việt mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng có dấu hỏi chấm ở cuối."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            content = resp.choices[0].message.content
            if content:
                questions = content.strip().split("\n")
                cleaned = [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()]
                if cleaned:
                    return cleaned[:n_questions]
        except Exception as e:
            if "AuthenticationError" in type(e).__name__ or "401" in str(e):
                _API_KEY_VALID = False
            print(f"  ⚠️  OpenAI HyQA fallback: {e}")

    # Extractive fallback
    import re
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    global _API_KEY_VALID
    if OPENAI_API_KEY and _API_KEY_VALID:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết 1 câu ngắn bằng tiếng Việt mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
            )
            content = resp.choices[0].message.content
            if content:
                context = content.strip()
                return f"{context}\n\n{text}"
        except Exception as e:
            if "AuthenticationError" in type(e).__name__ or "401" in str(e):
                _API_KEY_VALID = False
            print(f"  ⚠️  OpenAI contextual fallback: {e}")

    # Simple fallback
    prefix = f"Trích từ tài liệu {document_title}. " if document_title else ""
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    global _API_KEY_VALID
    if OPENAI_API_KEY and _API_KEY_VALID:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về định dạng JSON thuần không markdown: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
            )
            content = resp.choices[0].message.content
            if content:
                return json.loads(content)
        except Exception as e:
            if "AuthenticationError" in type(e).__name__ or "401" in str(e):
                _API_KEY_VALID = False
            print(f"  ⚠️  OpenAI metadata fallback: {e}")

    return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    global _API_KEY_VALID
    if OPENAI_API_KEY and _API_KEY_VALID:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và trả về kết quả dưới dạng JSON object hợp lệ:
{
  "summary": "tóm tắt 2-3 câu bằng tiếng Việt",
  "questions": ["câu hỏi 1?", "câu hỏi 2?", "câu hỏi 3?"],
  "context": "1 câu mô tả vị trí và nội dung của đoạn văn trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi"}
}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                response_format={"type": "json_object"},
                max_tokens=400,
            )
            content = resp.choices[0].message.content
            if content:
                return json.loads(content)
        except Exception as e:
            if "AuthenticationError" in type(e).__name__ or "401" in str(e):
                _API_KEY_VALID = False
            print(f"  ⚠️  Enrichment API fallback: {e}")

    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    summary = ". ".join(sentences[:2]) + "." if sentences else text
    context_prefix = f"Trích từ tài liệu {source}." if source else "Tài liệu nội bộ."
    return {
        "summary": summary,
        "questions": [f"{s.rstrip('.')}?" for s in sentences[:3]],
        "context": context_prefix,
        "metadata": {"topic": "general", "entities": [], "category": "policy", "language": "vi"},
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
