"""ChromaDB local com funcao de embedding OpenAI (chave apenas via ambiente)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 120) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def persist_directory_size_bytes(persist_directory: str | Path) -> int:
    root = Path(persist_directory)
    if not root.is_dir():
        return 0
    total = 0
    for f in root.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def list_knowledge_files(knowledge_dir: Path) -> list[Path]:
    if not knowledge_dir.is_dir():
        return []
    found: set[Path] = set()
    for pattern in ("*.txt", "*.md"):
        found.update(knowledge_dir.glob(pattern))
    return sorted(found, key=lambda p: p.name.lower())


def make_openai_embedding_function(api_key: str, model_name: str) -> embedding_functions.OpenAIEmbeddingFunction:
    return embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=model_name)


def get_chroma_client(persist_directory: str) -> chromadb.PersistentClient:
    path = Path(persist_directory)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def get_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    ef: embedding_functions.OpenAIEmbeddingFunction,
) -> Collection:
    return client.get_or_create_collection(name=collection_name, embedding_function=ef)


def reset_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    ef: embedding_functions.OpenAIEmbeddingFunction,
) -> Collection:
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name, embedding_function=ef)


def index_knowledge_dir(collection: Collection, knowledge_dir: Path) -> tuple[int, int]:
    """Le arquivos .txt/.md, faz upsert por chunks. Retorna (num_arquivos, num_chunks)."""
    files = list_knowledge_files(knowledge_dir)
    if not files:
        return (0, 0)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for fp in files:
        text = fp.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            digest = hashlib.sha256(f"{fp.name}:{i}:{chunk[:120]}".encode("utf-8")).hexdigest()[:32]
            ids.append(digest)
            documents.append(chunk)
            metadatas.append({"source": fp.name, "chunk_index": int(i)})

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return (len(files), len(ids))


def query_similar(
    collection: Collection,
    query_text: str,
    n_results: int,
) -> dict:
    return collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
