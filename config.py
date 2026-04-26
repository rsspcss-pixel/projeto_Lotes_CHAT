import os


def _normalize_secret_env(value: str | None) -> str:
    """Remove espacos e aspas envolventes comuns em ficheiros .env."""
    if not value:
        return ""
    s = value.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _optional_int(name: str, default: str | None) -> int | None:
    raw = os.getenv(name, default)
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return None
    return int(raw)


FLOWISE_API_URL = _required("FLOWISE_API_URL")
FLOWISE_API_TOKEN = _required("FLOWISE_API_TOKEN")
REQUEST_CONNECT_TIMEOUT_SECONDS = _optional_int("REQUEST_CONNECT_TIMEOUT_SECONDS", "10")
REQUEST_READ_TIMEOUT_SECONDS = _optional_int("REQUEST_READ_TIMEOUT_SECONDS", "600")
DUCKDB_DATABASE_PATH = os.getenv("DUCKDB_DATABASE_PATH", "data/compras.duckdb").strip() or "data/compras.duckdb"
DUCKDB_SOURCE_DIR = os.getenv("DUCKDB_SOURCE_DIR", "data").strip() or "data"

# ChromaDB embutido + embeddings OpenAI (configure apenas via ambiente; nao armazene segredos no codigo)
OPENAI_API_KEY = _normalize_secret_env(os.getenv("OPENAI_API_KEY"))
OPENAI_EMBEDDING_MODEL = (
    os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip() or "text-embedding-3-small"
)
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "data/chroma").strip() or "data/chroma"
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "negociacao_conhecimento").strip() or "negociacao_conhecimento"
KNOWLEDGE_TXT_DIR = os.getenv("KNOWLEDGE_TXT_DIR", "data/documentos_negociacao").strip() or "data/documentos_negociacao"
