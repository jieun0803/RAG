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


def test_similarity_search_delegates_to_vectorstore():
    store = FinancialEmbeddingStore.__new__(FinancialEmbeddingStore)
    store.vectorstore = MagicMock()
    store.vectorstore.similarity_search.return_value = ["doc1"]

    out = store.similarity_search("이자율", k=3)
    assert out == ["doc1"]
    store.vectorstore.similarity_search.assert_called_once_with("이자율", k=3)


def test_get_collection_info_and_load_existing_vectorstore():
    store = FinancialEmbeddingStore.__new__(FinancialEmbeddingStore)
    store.collection_name = "financial_documents"
    store.persist_dir = "vectordb"
    store.client = MagicMock()
    collection = MagicMock()
    collection.count.return_value = 2
    store.client.get_or_create_collection.return_value = collection

    info = store.get_collection_info()
    assert info["collection_name"] == "financial_documents"
    assert info["persist_dir"] == "vectordb"
    assert info["count"] == 2
    assert store.load_existing_vectorstore() is True
