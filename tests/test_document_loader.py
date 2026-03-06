import pytest
from pathlib import Path
from unittest.mock import patch
from langchain_core.documents import Document
import app.document_loader as dl

# Helper
def _make_lang_dir(tmp_path: Path, language: str, files: dict) -> Path:
    """
    Create a docs/<language>/ directory inside tmp_path and populate it.

    Args:
        tmp_path: pytest's temporary directory fixture
        language: subfolder name
        files: {filename: content} mapping
    """
    lang_dir = tmp_path / language
    lang_dir.mkdir(parents=True)
    for name, content in files.items():
        (lang_dir / name).write_text(content, encoding="utf-8")
    return lang_dir


# Tests
class TestGetAvailableLanguages:
    def test_returns_sorted_language_names(self, tmp_path):
        # Create two language folders in tmp_path (acts as fake docs/ dir)
        (tmp_path / "rust").mkdir()
        (tmp_path / "python").mkdir()

        # Patch DOCS_DIR on the already-imported module object.
        with patch.object(dl, "DOCS_DIR", tmp_path):
            langs = dl.get_available_languages()

        assert langs == ["python", "rust"]

    def test_returns_empty_if_no_subdirs(self, tmp_path):
        with patch.object(dl, "DOCS_DIR", tmp_path):
            langs = dl.get_available_languages()

        assert langs == []

    def test_ignores_files_at_root(self, tmp_path):
        # A loose file in docs/ should not appear as a language
        (tmp_path / "readme.txt").write_text("hi")

        with patch.object(dl, "DOCS_DIR", tmp_path):
            langs = dl.get_available_languages()

        assert langs == []

    def test_returns_only_directories(self, tmp_path):
        # Mix of a file and a directory, only the directory should appear
        (tmp_path / "python").mkdir()
        (tmp_path / "stray_file.txt").write_text("ignore")

        with patch.object(dl, "DOCS_DIR", tmp_path):
            langs = dl.get_available_languages()

        assert langs == ["python"]

class TestLoadDocumentsForLanguage:
    def test_loads_and_chunks_txt_file(self, tmp_path):
        # 2600 chars, well above CHUNK_SIZE (800) so we expect multiple chunks
        content = "Hello world. " * 200
        _make_lang_dir(tmp_path, "python", {"intro.txt": content})

        with patch.object(dl, "DOCS_DIR", tmp_path):
            docs = dl.load_documents_for_language("python")

        assert len(docs) > 1                          # content was chunked
        assert all(isinstance(d, Document) for d in docs)
        assert all(d.metadata["language"] == "python" for d in docs)

    def test_metadata_includes_source(self, tmp_path):
        _make_lang_dir(tmp_path, "js", {"overview.md": "# JS\nSome content here."})

        with patch.object(dl, "DOCS_DIR", tmp_path):
            docs = dl.load_documents_for_language("js")

        sources = {d.metadata["source"] for d in docs}
        assert any("overview.md" in s for s in sources)

    def test_raises_if_language_not_found(self, tmp_path):
        # tmp_path is empty — no "nonexistent" subfolder exists
        with patch.object(dl, "DOCS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                dl.load_documents_for_language("nonexistent")

    def test_raises_if_no_supported_files(self, tmp_path):
        # Create a folder with only an unsupported file type
        lang_dir = tmp_path / "empty_lang"
        lang_dir.mkdir()
        (lang_dir / "image.png").write_bytes(b"\x89PNG")

        with patch.object(dl, "DOCS_DIR", tmp_path):
            with pytest.raises(ValueError, match="No supported files"):
                dl.load_documents_for_language("empty_lang")

    def test_loads_multiple_extensions(self, tmp_path):
        _make_lang_dir(tmp_path, "rust", {
            "basics.txt": "Rust ownership model explained.",
            "advanced.md": "# Lifetimes\nLifetimes prevent dangling references.",
        })

        with patch.object(dl, "DOCS_DIR", tmp_path):
            docs = dl.load_documents_for_language("rust")

        filenames = {d.metadata["filename"] for d in docs}
        assert "basics.txt" in filenames
        assert "advanced.md" in filenames

    def test_chunk_metadata_preserves_language(self, tmp_path):
        # Every chunk produced from a language's files must carry that language in its metadata
        content = "Python is great. " * 100
        _make_lang_dir(tmp_path, "python", {"guide.txt": content})

        with patch.object(dl, "DOCS_DIR", tmp_path):
            docs = dl.load_documents_for_language("python")

        assert all(d.metadata.get("language") == "python" for d in docs)

    def test_single_small_file_produces_at_least_one_chunk(self, tmp_path):
        # Even a file smaller than CHUNK_SIZE should produce exactly one Document
        _make_lang_dir(tmp_path, "lua", {"intro.txt": "Lua is a lightweight language."})

        with patch.object(dl, "DOCS_DIR", tmp_path):
            docs = dl.load_documents_for_language("lua")

        assert len(docs) >= 1
        assert "Lua" in docs[0].page_content