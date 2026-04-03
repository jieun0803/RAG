from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.embedding import FinancialEmbeddingStore
from src.retriever import (
    FinancialDocumentRetriever,
    _document_to_source,
    enhance_query_with_finance_terms,
)


@patch("src.retriever.RetrievalQA")
@patch("src.retriever.ChatUpstage")
def test_retriever_init_with_vectorstore(mock_chat: MagicMock, mock_rqa: MagicMock) -> None:
    mock_rqa.from_chain_type.return_value = MagicMock(name="qa_chain")
    vs = MagicMock()
    vs.as_retriever.return_value = MagicMock(name="retriever")

    r = FinancialDocumentRetriever(vs)

    assert r.vectorstore is vs
    vs.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
    mock_chat.assert_called_once()
    assert r.llm is mock_chat.return_value
    assert r.retriever is vs.as_retriever.return_value
    assert r.qa_chain is mock_rqa.from_chain_type.return_value
    mock_rqa.from_chain_type.assert_called_once()
    call_kw = mock_rqa.from_chain_type.call_args.kwargs
    assert call_kw["chain_type"] == "stuff"
    assert call_kw["return_source_documents"] is True
    assert call_kw["retriever"] is r.retriever


@patch("src.retriever.RetrievalQA")
@patch("src.retriever.ChatUpstage")
def test_retriever_init_with_embedding_store(
    mock_chat: MagicMock, mock_rqa: MagicMock
) -> None:
    mock_rqa.from_chain_type.return_value = MagicMock()

    class _StubStore(FinancialEmbeddingStore):
        def __init__(self) -> None:
            self.vectorstore = MagicMock()
            self.vectorstore.as_retriever.return_value = MagicMock()

    store = _StubStore()
    r = FinancialDocumentRetriever(store)
    assert r.vectorstore is store.vectorstore


@patch("src.retriever.RetrievalQA")
@patch("src.retriever.ChatUpstage")
def test_answer_question(mock_chat: MagicMock, mock_rqa: MagicMock) -> None:
    qa = MagicMock()
    qa.invoke.return_value = {
        "result": "답변 본문",
        "source_documents": [
            Document(page_content="청크 텍스트", metadata={"source": "a.pdf"}),
        ],
    }
    mock_rqa.from_chain_type.return_value = qa

    vs = MagicMock()
    vs.as_retriever.return_value = MagicMock()
    r = FinancialDocumentRetriever(vs)
    out = r.answer_question("이자율은?")

    assert out["question"] == "이자율은?"
    assert out["enhanced_query"] == "이자율은? 금리 연체이자"
    assert out["answer"] == "답변 본문"
    assert len(out["sources"]) == 1
    assert out["sources"][0]["page_content"] == "청크 텍스트"
    assert out["sources"][0]["metadata"]["source"] == "a.pdf"
    qa.invoke.assert_called_once_with({"query": "이자율은? 금리 연체이자"})


@patch("src.retriever.RetrievalQA")
@patch("src.retriever.ChatUpstage")
def test_answer_question_skips_enhancement(
    mock_chat: MagicMock, mock_rqa: MagicMock
) -> None:
    qa = MagicMock()
    qa.invoke.return_value = {"result": "x", "source_documents": []}
    mock_rqa.from_chain_type.return_value = qa
    vs = MagicMock()
    vs.as_retriever.return_value = MagicMock()
    r = FinancialDocumentRetriever(vs)
    out = r.answer_question("이자율은?", enhance_query=False)

    assert "enhanced_query" not in out
    qa.invoke.assert_called_once_with({"query": "이자율은?"})


def test_enhance_query_with_finance_terms_adds_related_words() -> None:
    assert "금리" in enhance_query_with_finance_terms("이자는 어떻게 되나요?")
    assert enhance_query_with_finance_terms("  ") == ""


def test_document_to_source_plain_object() -> None:
    m = MagicMock()
    m.page_content = "x"
    m.metadata = {"k": 1}
    assert _document_to_source(m) == {"page_content": "x", "metadata": {"k": 1}}
