"""PDF 텍스트 추출·청킹 (Phase 2)."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Literal

import fitz

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_SEPARATORS = ["\n\n제", "\n\n", "\n", " ", ""]


def extract_text_with_pymupdf(pdf_path: str | Path) -> str:
    """PyMuPDF(fitz)로 PDF 전체 페이지에서 텍스트를 이어 붙여 반환한다."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path.resolve()}")

    parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            parts.append(page.get_text())

    return "\n".join(parts).strip()


def clean_financial_text(text: str) -> str:
    """금융 문서 텍스트를 최소한으로 정리한다."""
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[str]:
    """텍스트를 chunk_size/chunk_overlap 기준으로 분할한다."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text:
        return []

    splitter = _build_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or DEFAULT_SEPARATORS,
    )
    chunks = splitter.split_text(text)
    return [chunk for chunk in chunks if chunk and chunk.strip()]


def chunks_to_records(
    chunks: list[str],
    source: str | Path,
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """문자열 청크 리스트를 id/text/length/metadata 스키마의 dict 리스트로 만든다."""
    path = Path(source)
    base_meta: dict[str, Any] = {"source": path.name}
    if extra_metadata:
        base_meta.update(extra_metadata)

    out: list[dict[str, Any]] = []
    record_id = 0
    for chunk in chunks:
        text = chunk.strip() if chunk else ""
        if not text:
            continue
        out.append(
            {
                "id": record_id,
                "text": text,
                "length": len(text),
                "metadata": {**base_meta},
            }
        )
        record_id += 1
    return out


def chunk_by_articles(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[str]:
    """`제N조` 경계를 우선으로 나누고, 긴 조항은 Recursive 분할로 이어 붙인다."""
    if not text.strip():
        return []
    parts = re.split(r"(?=제\s*\d+\s*조)", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    if not parts:
        return split_text_into_chunks(text, chunk_size, chunk_overlap, separators)
    out: list[str] = []
    for part in parts:
        if len(part) <= chunk_size:
            out.append(part)
        else:
            out.extend(split_text_into_chunks(part, chunk_size, chunk_overlap, separators))
    return out


def process_pdf(
    pdf_path: str | Path,
    *,
    chunking_method: Literal["langchain", "articles"] = "langchain",
    enrich_metadata: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """PDF 경로에서 추출·정리·분할 후 청크 레코드 리스트를 반환한다."""
    if enrich_metadata:
        raise NotImplementedError(
            "enrich_metadata=True는 Phase 5 metadata_filter 연동 후 지원합니다."
        )
    path = Path(pdf_path)
    raw = extract_text_with_pymupdf(path)
    cleaned = clean_financial_text(raw)
    if not cleaned:
        return []

    if chunking_method == "langchain":
        pieces = split_text_into_chunks(cleaned, chunk_size, chunk_overlap, separators)
    elif chunking_method == "articles":
        pieces = chunk_by_articles(cleaned, chunk_size, chunk_overlap, separators)
    else:
        raise ValueError(f"chunking_method must be 'langchain' or 'articles', got {chunking_method!r}")

    return chunks_to_records(pieces, source=path, extra_metadata=extra_metadata)


def _build_text_splitter(chunk_size: int, chunk_overlap: int, separators: list[str]):
    """가능하면 LangChain splitter를 쓰고, 없으면 내부 splitter로 대체한다.

    LangChain text_splitters는 transformers/torch를 끌어올 수 있어, Windows에서
    DLL 로드(OSError 등)로 import 자체가 실패할 수 있다. 그때도 동작하도록 폴백한다.
    """
    _lc_import_errors = (ImportError, OSError, RuntimeError)
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )
    except _lc_import_errors:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
            )
        except _lc_import_errors:
            return _LocalTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )


class _LocalTextSplitter:
    """LangChain 미설치 환경용 고정 길이 splitter."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks
