import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain.schema import Document


def _fake_docs(n: int = 5) -> list:
    """Generate n mock LangChain Documents for testing."""
    return [
        Document(
            page_content=f"Chunk number {i} about programming.",
            metadata={"source": "test.txt", "language": "python"},
        )
        for i in range(n)
    ]


@pytest.fixture()
def mock_embeddings():
    """Patch HuggingFaceEmbeddings so no model download happens in tests."""
    with patch("app.vector_store.HuggingFaceEmbeddings") as mock_cls:
        # Return a MagicMock that behaves like an embeddings object
        instance = MagicMock()
        # embed_documents must return a list of same-length float vectors
        instance.embed_documents.return_value = [
            [0.1] * 384 for _ in range(10)
        ]
        instance.embed_query.return_value = [0.1] * 384
        mock_cls.return_value = instance
        yield instance


@pytest.fixture()
def vs_manager(tmp_path, mock_embeddings):
    """Create a VectorStoreManager backed by a temp directory."""
    with patch("app.vector_store.CHROMA_DB_DIR", tmp_path):
        import importlib, app.vector_store as vs_module
        importlib.reload(vs_module)
        manager = vs_module.VectorStoreManager()
        yield manager


class TestCollectionExistence:
    def test_new_language_does_not_exist(self, vs_manager):
        assert not vs_manager.collection_exists("python")

    def test_exists_after_ingest(self, vs_manager):
        vs_manager.ingest("python", _fake_docs())
        assert vs_manager.collection_exists("python")

    def test_different_languages_are_independent(self, vs_manager):
        vs_manager.ingest("python", _fake_docs())
        assert not vs_manager.collection_exists("rust")


class TestIngest:
    def test_ingest_creates_collection(self, vs_manager):
        vs_manager.ingest("javascript", _fake_docs(3))
        assert "javascript" in vs_manager.list_ingested_languages()

    def test_reingest_replaces_collection(self, vs_manager):
        vs_manager.ingest("python", _fake_docs(3))
        # Re-ingest with different content — should not raise, just replace
        vs_manager.ingest("python", _fake_docs(7))
        assert vs_manager.collection_exists("python")

    def test_multiple_languages_coexist(self, vs_manager):
        vs_manager.ingest("python", _fake_docs(2))
        vs_manager.ingest("rust", _fake_docs(2))
        ingested = vs_manager.list_ingested_languages()
        assert "python" in ingested
        assert "rust" in ingested


class TestGetRetriever:
    def test_raises_if_not_ingested(self, vs_manager):
        with pytest.raises(RuntimeError, match="does not exist"):
            vs_manager.get_retriever("haskell")

    def test_returns_retriever_after_ingest(self, vs_manager):
        vs_manager.ingest("python", _fake_docs(5))
        retriever = vs_manager.get_retriever("python")
        # A LangChain retriever must have an invoke / get_relevant_documents method
        assert hasattr(retriever, "invoke") or hasattr(
            retriever, "get_relevant_documents"
        )