from pathlib import Path

import fitz
import pytest

from src.chunking import (
    chunk_by_articles,
    chunks_to_records,
    clean_financial_text,
    extract_text_with_pymupdf,
    process_pdf,
    split_text_into_chunks,
)


def _write_minimal_pdf(path: Path, body: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), body)
    doc.save(str(path))
    doc.close()


def test_extract_text_with_pymupdf_roundtrip(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _write_minimal_pdf(pdf_path, "Hello PyMuPDF")
    text = extract_text_with_pymupdf(pdf_path)
    assert "Hello PyMuPDF" in text


def test_extract_text_with_pymupdf_missing_file(tmp_path):
    missing = tmp_path / "nope.pdf"
    with pytest.raises(FileNotFoundError, match="PDF not found"):
        extract_text_with_pymupdf(missing)


def test_extract_text_with_pymupdf_empty_pages(tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(pdf_path))
    doc.close()
    assert extract_text_with_pymupdf(pdf_path) == ""


def test_clean_financial_text_normalizes_whitespace():
    raw = "제1조   목적\r\n\r\n\r\n  이 약관은\t서비스 이용 조건을 정한다.  "
    cleaned = clean_financial_text(raw)
    assert cleaned == "제1조 목적\n\n 이 약관은 서비스 이용 조건을 정한다."


def test_split_text_into_chunks_basic_overlap():
    text = "A" * 1200
    chunks = split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 2
    assert len(chunks[0]) == 1000
    assert len(chunks[1]) == 400


def test_split_text_into_chunks_invalid_params():
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        split_text_into_chunks("abc", chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap must be >= 0"):
        split_text_into_chunks("abc", chunk_overlap=-1)
    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        split_text_into_chunks("abc", chunk_size=10, chunk_overlap=10)


def test_chunks_to_records_schema_and_source():
    records = chunks_to_records(
        ["첫번째 청크", "  두번째  "],
        source="/path/to/약관.pdf",
    )
    assert len(records) == 2
    assert records[0] == {
        "id": 0,
        "text": "첫번째 청크",
        "length": len("첫번째 청크"),
        "metadata": {"source": "약관.pdf"},
    }
    assert records[1]["id"] == 1
    assert records[1]["text"] == "두번째"
    assert records[1]["length"] == len("두번째")
    assert records[1]["metadata"]["source"] == "약관.pdf"


def test_chunks_to_records_skips_empty_and_extra_metadata():
    records = chunks_to_records(
        ["a", "", "  ", "b"],
        source="doc.pdf",
        extra_metadata={"doc_type": "terms"},
    )
    assert len(records) == 2
    assert records[0]["id"] == 0
    assert records[1]["id"] == 1
    assert records[0]["metadata"] == {"source": "doc.pdf", "doc_type": "terms"}


def test_process_pdf_langchain(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    _write_minimal_pdf(pdf_path, "Hello PyMuPDF process")
    records = process_pdf(pdf_path, chunking_method="langchain")
    assert len(records) >= 1
    assert records[0]["metadata"]["source"] == "doc.pdf"
    assert set(records[0]) == {"id", "text", "length", "metadata"}
    assert "Hello" in records[0]["text"]


def test_process_pdf_passes_extra_metadata(tmp_path):
    pdf_path = tmp_path / "meta.pdf"
    _write_minimal_pdf(pdf_path, "Hello")
    records = process_pdf(
        pdf_path,
        chunking_method="langchain",
        extra_metadata={"doc_type": "terms"},
    )
    assert records[0]["metadata"] == {"source": "meta.pdf", "doc_type": "terms"}


def test_process_pdf_enrich_metadata_raises(tmp_path):
    pdf_path = tmp_path / "x.pdf"
    _write_minimal_pdf(pdf_path, "hi")
    with pytest.raises(NotImplementedError, match="Phase 5"):
        process_pdf(pdf_path, enrich_metadata=True)


def test_chunk_by_articles_splits_on_article_headers():
    text = "전문\n제1조 목적은 이하와 같다.\n제2조 정의는 다음과 같다."
    chunks = chunk_by_articles(text, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) >= 2
    assert any("제1조" in c for c in chunks)
    assert any("제2조" in c for c in chunks)


def test_process_pdf_articles_method(tmp_path):
    pdf_path = tmp_path / "articles.pdf"
    _write_minimal_pdf(pdf_path, "제1조 앞쪽 제2조 뒤쪽")
    records = process_pdf(pdf_path, chunking_method="articles")
    assert len(records) >= 1
    assert records[0]["metadata"]["source"] == "articles.pdf"


@pytest.mark.parametrize(
    ("filename", "expected_keyword"),
    [
        ("Terms_sheet_persnal_MERGE_09923.pdf", "KB국민"),
        ("kbcard_20241230.pdf", "WE:SH"),
    ],
)
def test_extract_text_with_pymupdf_sample_pdfs(filename, expected_keyword):
    repo_root = Path(__file__).resolve().parents[1]
    pdf_path = repo_root / "data" / "pdfs" / filename
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found: {pdf_path}")

    text = extract_text_with_pymupdf(pdf_path)
    assert text.strip() != ""
    assert expected_keyword in text


def test_extract_text_with_pymupdf_all_pdfs_in_data_folder():
    repo_root = Path(__file__).resolve().parents[1]
    pdf_dir = repo_root / "data" / "pdfs"
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        pytest.skip(f"No PDF files found in {pdf_dir}")

    for pdf_path in pdf_paths:
        text = extract_text_with_pymupdf(pdf_path)
        assert text.strip() != "", f"Extracted empty text from {pdf_path.name}"


def preview_extracted_texts(preview_chars: int = 400) -> None:
    """data/pdfs 내 PDF 추출 결과를 사람이 읽기 쉽게 출력한다."""
    repo_root = Path(__file__).resolve().parents[1]
    pdf_dir = repo_root / "data" / "pdfs"
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"[INFO] No PDF files found in {pdf_dir}")
        return

    print(f"[INFO] Checking {len(pdf_paths)} PDF file(s) in: {pdf_dir}")
    for pdf_path in pdf_paths:
        text = extract_text_with_pymupdf(pdf_path)
        preview = text[:preview_chars].replace("\n", " ")
        status = "OK" if text.strip() else "EMPTY"
        print("=" * 80)
        print(f"FILE   : {pdf_path.name}")
        print(f"STATUS : {status}")
        print(f"LENGTH : {len(text)}")
        print(f"PREVIEW: {preview}")
    print("=" * 80)


if __name__ == "__main__":
    preview_extracted_texts()


def preview_one_chunk(
    pdf_filename: str = "Terms_sheet_persnal_MERGE_09923.pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    preview_chars: int = 400,
) -> None:
    """지정한 PDF에서 첫 번째 chunk 1개를 출력한다."""
    repo_root = Path(__file__).resolve().parents[1]
    pdf_path = repo_root / "data" / "pdfs" / pdf_filename
    if not pdf_path.exists():
        print(f"[INFO] PDF not found: {pdf_path}")
        return

    text = extract_text_with_pymupdf(pdf_path)
    chunks = _split_text_with_overlap(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        print("[INFO] No chunks generated.")
        return

    first_chunk = chunks[0]
    print("=" * 80)
    print(f"FILE         : {pdf_filename}")
    print(f"TOTAL_CHUNKS : {len(chunks)}")
    print(f"FIRST_LEN    : {len(first_chunk)}")
    print(f"FIRST_CHUNK  : {first_chunk[:preview_chars].replace(chr(10), ' ')}")
    print("=" * 80)


def compare_raw_vs_cleaned_chunks(
    pdf_filename: str = "Terms_sheet_persnal_MERGE_09923.pdf",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    preview_chars: int = 300,
    max_pairs: int = 3,
) -> None:
    """원문 추출 텍스트와 정리 텍스트의 chunk를 나란히 비교 출력한다."""
    repo_root = Path(__file__).resolve().parents[1]
    pdf_path = repo_root / "data" / "pdfs" / pdf_filename
    if not pdf_path.exists():
        print(f"[INFO] PDF not found: {pdf_path}")
        return

    raw_text = extract_text_with_pymupdf(pdf_path)
    cleaned_text = clean_financial_text(raw_text)
    raw_chunks = _split_text_with_overlap(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    cleaned_chunks = _split_text_with_overlap(
        cleaned_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print("=" * 100)
    print(f"FILE            : {pdf_filename}")
    print(f"RAW_TEXT_LEN    : {len(raw_text)}")
    print(f"CLEAN_TEXT_LEN  : {len(cleaned_text)}")
    print(f"RAW_CHUNKS      : {len(raw_chunks)}")
    print(f"CLEAN_CHUNKS    : {len(cleaned_chunks)}")
    print("-" * 100)

    pair_count = min(max_pairs, len(raw_chunks), len(cleaned_chunks))
    if pair_count == 0:
        print("[INFO] No chunks to compare.")
        print("=" * 100)
        return

    for idx in range(pair_count):
        raw_preview = raw_chunks[idx][:preview_chars].replace("\n", " ")
        cleaned_preview = cleaned_chunks[idx][:preview_chars].replace("\n", " ")
        print(f"[CHUNK #{idx}]")
        print(f"RAW_LEN     : {len(raw_chunks[idx])}")
        print(f"CLEAN_LEN   : {len(cleaned_chunks[idx])}")
        print(f"RAW_PREVIEW : {raw_preview}\n\n")
        print(f"CLEAN_PREVIEW: {cleaned_preview}")
        print("-" * 100)
    print("=" * 100)


def compare_first_chunk_same_window(
    pdf_filename: str = "Terms_sheet_persnal_MERGE_09923.pdf",
    chunk_size: int = 1000,
    preview_chars: int = 500,
) -> None:
    """첫 번째 chunk 범위(0~chunk_size)를 동일하게 맞춰 원문/정리본을 비교한다."""
    repo_root = Path(__file__).resolve().parents[1]
    pdf_path = repo_root / "data" / "pdfs" / pdf_filename
    if not pdf_path.exists():
        print(f"[INFO] PDF not found: {pdf_path}")
        return

    raw_text = extract_text_with_pymupdf(pdf_path)
    cleaned_text = clean_financial_text(raw_text)

    raw_first = raw_text[:chunk_size].strip()
    cleaned_first = cleaned_text[:chunk_size].strip()

    print("=" * 100)
    print(f"FILE                : {pdf_filename}")
    print(f"WINDOW              : 0 ~ {chunk_size}")
    print(f"RAW_FIRST_LEN       : {len(raw_first)}")
    print(f"CLEANED_FIRST_LEN   : {len(cleaned_first)}")
    print("-" * 100)
    print(f"RAW_FIRST_PREVIEW   : {raw_first[:preview_chars].replace(chr(10), ' ')}")
    print("-" * 100)
    print(f"CLEAN_FIRST_PREVIEW : {cleaned_first[:preview_chars].replace(chr(10), ' ')}")
    print("=" * 100)


def _split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """LangChain 없이 간단히 고정 길이 chunk를 만든다."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text:
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
