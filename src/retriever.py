"""검색 + LLM 답변 (Phase 4)."""

from __future__ import annotations

from typing import Any

from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_upstage import ChatUpstage

from src.embedding import FinancialEmbeddingStore

_RAG_SYSTEM = """다음 문맥(context)에만 근거하여 질문에 답하세요.
문맥에 근거가 없으면 "약관·문서에서 해당 내용을 찾을 수 없습니다."라고 답하고, 없는 사실을 지어내지 마세요.
----------------
{context}"""

RAG_CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(_RAG_SYSTEM),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

# (trigger가 질문에 포함될 때) 검색 recall을 높이기 위해 붙이는 동의어·관련어
_FINANCE_EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("이자", ("금리", "연체이자")),
    ("금리", ("이자율", "이자")),
    ("수수료", ("수수료율",)),
    ("약관", ("계약조항",)),
    ("해지", ("계약해지", "중도해지")),
    ("대출", ("원리금", "상환")),
)


def enhance_query_with_finance_terms(question: str) -> str:
    """질문에 금융 동의어·관련 용어를 덧붙여 벡터 검색에 쓸 쿼리를 만든다. LLM 답변 문장은 바꾸지 않는다."""
    q = question.strip()
    if not q:
        return q
    to_add: list[str] = []
    seen: set[str] = set()
    for trigger, terms in _FINANCE_EXPANSIONS:
        if trigger not in q:
            continue
        for t in terms:
            if t in q or t in seen:
                continue
            seen.add(t) // 중복 방지
            to_add.append(t) // 질문 뒤에 추가 예정
    if not to_add:
        return q
    return f"{q} {' '.join(to_add)}"


class FinancialDocumentRetriever:
    """벡터스토어 + LLM으로 질의 응답을 구성한다."""

    def __init__(
        self,
        store: Any,
        *,
        search_k: int = 5,
        model: str = "solar-pro2",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        api_key: str | None = None,
        enable_query_enhancement: bool = True,
    ) -> None:
        """`store`는 `FinancialEmbeddingStore`(`.vectorstore` 사용) 또는 LangChain 벡터스토어."""
        if isinstance(store, FinancialEmbeddingStore):
            self.vectorstore = store.vectorstore
        else:
            self.vectorstore = store

        self.enable_query_enhancement = enable_query_enhancement
        self.llm = ChatUpstage(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": search_k})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": RAG_CHAT_PROMPT},
        )

    def answer_question(
        self,
        question: str,
        *,
        enhance_query: bool | None = None,
    ) -> dict[str, Any]:
        """질문에 대해 답변과 검색 출처를 반환한다. 검색 단계에만 쿼리 보강을 적용한다."""
        use_enhance = (
            self.enable_query_enhancement if enhance_query is None else enhance_query
        )
        q_retrieval = (
            enhance_query_with_finance_terms(question) if use_enhance else question.strip()
        )
        raw = self.qa_chain.invoke({"query": q_retrieval})
        docs = raw.get("source_documents") or []
        out: dict[str, Any] = {
            "question": question,
            "answer": raw.get("result") or "",
            "sources": [_document_to_source(d) for d in docs],
        }
        if q_retrieval != question.strip():
            out["enhanced_query"] = q_retrieval
        return out


def _document_to_source(doc: Document | Any) -> dict[str, Any]:
    if isinstance(doc, Document):
        return {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata or {}),
        }
    page_content = getattr(doc, "page_content", str(doc))
    metadata = getattr(doc, "metadata", None) or {}
    return {"page_content": page_content, "metadata": dict(metadata)}
