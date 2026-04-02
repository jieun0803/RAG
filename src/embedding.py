"""임베딩 + 벡터스토어 관리 (Phase 3)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class FinancialEmbeddingStore:
    """금융 문서 임베딩/저장을 담당하는 스토어 객체 뼈대."""

    def __init__(
        self,
        persist_dir: str | Path = "./vectordb",
        collection_name: str = "financial_documents",
        embedding_model: str = "solar-embedding-1-large-passage",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        self.api_key = self._require_api_key()
        self.embeddings = self._create_embeddings()
        self.client = self._create_persistent_client()
        self.vectorstore = self._create_vectorstore()

    def _require_api_key(self) -> str:
        api_key = os.getenv("UPSTAGE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("UPSTAGE_API_KEY is not set in environment.")
        if api_key == "your_upstage_api_key_here":
            raise ValueError("UPSTAGE_API_KEY is placeholder value.")
        return api_key

    def _create_embeddings(self):
        try:
            from langchain_upstage import UpstageEmbeddings
        except ImportError as exc:
            raise ImportError("langchain_upstage is required for embeddings.") from exc

        return UpstageEmbeddings(
            model=self.embedding_model,
            api_key=self.api_key,
        )

    def _create_persistent_client(self):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("chromadb is required for vector storage.") from exc

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self.persist_dir))

    def _create_vectorstore(self):
        chroma_cls = self._resolve_chroma_class()
        return chroma_cls(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

    def _resolve_chroma_class(self):
        try:
            from langchain_chroma import Chroma

            return Chroma
        except ImportError:
            try:
                from langchain_community.vectorstores import Chroma

                return Chroma
            except ImportError as exc:
                raise ImportError(
                    "Install langchain-chroma or langchain-community for Chroma integration."
                ) from exc

    def add_documents(
        self,
        chunks: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """청크 dict 리스트를 Document로 변환해 벡터스토어에 추가한다."""
        try:
            from langchain_core.documents import Document
        except ImportError:
            from langchain.schema import Document

        extra = dict(metadata or {})
        documents: list[Any] = []
        ids: list[str] = []

        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            meta: dict[str, Any] = dict(chunk.get("metadata") or {})
            meta.update(extra)
            if "id" in chunk:
                meta.setdefault("chunk_id", chunk["id"])
            documents.append(Document(page_content=text, metadata=meta))
            ids.append(str(chunk.get("id", len(ids))))

        if not documents:
            return []

        try:
            return self.vectorstore.add_documents(documents=documents, ids=ids)
        except TypeError:
            return self.vectorstore.add_documents(documents=documents)

    def similarity_search(self, query: str, k: int = 5):
        """질의와 유사한 문서를 상위 k개 반환한다."""
        return self.vectorstore.similarity_search(query, k=k)

    def get_collection_info(self) -> dict[str, Any]:
        """현재 컬렉션의 기본 정보를 반환한다."""
        collection = self.client.get_or_create_collection(name=self.collection_name)
        return {
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "count": collection.count(),
        }

    def load_existing_vectorstore(self) -> bool:
        """기존 컬렉션에 문서가 하나 이상 있으면 True를 반환한다."""
        info = self.get_collection_info()
        return info["count"] > 0
