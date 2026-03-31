from unittest.mock import MagicMock

from src.embedding import FinancialEmbeddingStore


def test_add_documents_skips_empty_and_calls_vectorstore():
    store = FinancialEmbeddingStore.__new__(FinancialEmbeddingStore)
    store.vectorstore = MagicMock()
    store.vectorstore.add_documents.return_value = ["0"]

    chunks = [
        {"id": 0, "text": "hello", "metadata": {"source": "a.pdf"}},
        {"id": 1, "text": "   ", "metadata": {}},
    ]
    out = store.add_documents(chunks, metadata={"layer": "test"})
    assert out == ["0"]
    store.vectorstore.add_documents.assert_called_once()
    kwargs = store.vectorstore.add_documents.call_args.kwargs
    docs = kwargs["documents"]
    assert len(docs) == 1
    assert docs[0].page_content == "hello"
    assert docs[0].metadata["source"] == "a.pdf"
    assert docs[0].metadata["layer"] == "test"
    assert docs[0].metadata["chunk_id"] == 0


def test_add_documents_empty_returns_empty_list():
    store = FinancialEmbeddingStore.__new__(FinancialEmbeddingStore)
    store.vectorstore = MagicMock()
    assert store.add_documents([]) == []
    store.vectorstore.add_documents.assert_not_called()
