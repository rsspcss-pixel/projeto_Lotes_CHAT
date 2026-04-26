import json
import time
import uuid
from pathlib import Path

import duckdb
import pandas as pd
import requests
import streamlit as st

import chroma_rag

FLOWISE_READY = True
FLOWISE_CONFIG_ERROR = ""
FLOWISE_API_URL = ""
FLOWISE_API_TOKEN = ""
REQUEST_CONNECT_TIMEOUT_SECONDS = 10
REQUEST_READ_TIMEOUT_SECONDS = 600
DUCKDB_DATABASE_PATH = "data/compras.duckdb"
DUCKDB_SOURCE_DIR = "data"
OPENAI_API_KEY = ""
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_PERSIST_DIRECTORY = "data/chroma"
CHROMA_COLLECTION_NAME = "negociacao_conhecimento"
KNOWLEDGE_TXT_DIR = "data/documentos_negociacao"
ABNT_PROMPT_PREFIX = (
    "Responda em portugues do Brasil, com redacao formal e clara, seguindo estilo tecnico "
    "compativel com ABNT. Use acentuacao correta, termos precisos e, quando fizer sentido, "
    "apresente em topicos objetivos."
)

try:
    from config import (
        CHROMA_COLLECTION_NAME,
        CHROMA_PERSIST_DIRECTORY,
        DUCKDB_DATABASE_PATH,
        DUCKDB_SOURCE_DIR,
        FLOWISE_API_TOKEN,
        FLOWISE_API_URL,
        KNOWLEDGE_TXT_DIR,
        OPENAI_API_KEY,
        OPENAI_EMBEDDING_MODEL,
        REQUEST_CONNECT_TIMEOUT_SECONDS,
        REQUEST_READ_TIMEOUT_SECONDS,
    )
except Exception as exc:
    FLOWISE_READY = False
    FLOWISE_CONFIG_ERROR = str(exc)


def extract_answer(data: object) -> str:
    if isinstance(data, dict):
        if data.get("text"):
            return str(data["text"])
        if data.get("answer"):
            return str(data["answer"])
        if data.get("response"):
            return str(data["response"])
    return repair_mojibake(str(data))


def repair_mojibake(text: str) -> str:
    markers = ("Ã", "Â", "â€", "â€™", "â€œ", "â€”")
    if not any(marker in text for marker in markers):
        return text
    try:
        repaired = text.encode("latin1").decode("utf-8")
        if repaired:
            return repaired
    except UnicodeError:
        pass
    return text


def iterate_sse_chunks(response: requests.Response):
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith("data:"):
            continue
        payload = raw_line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            decoded = json.loads(payload)
            if isinstance(decoded, dict):
                if decoded.get("text"):
                    yield repair_mojibake(str(decoded["text"]))
                elif decoded.get("token"):
                    yield repair_mojibake(str(decoded["token"]))
                elif isinstance(decoded.get("data"), str):
                    yield repair_mojibake(decoded["data"])
            elif isinstance(decoded, str):
                yield repair_mojibake(decoded)
        except json.JSONDecodeError:
            yield repair_mojibake(payload)


def fake_stream(answer: str, placeholder):
    built = ""
    for part in answer.split(" "):
        if st.session_state.cancel_requested:
            break
        built = f"{built} {part}".strip()
        placeholder.markdown(built)
        time.sleep(0.006)
    return built or answer


def query_flowise(question: str, placeholder):
    if not FLOWISE_READY:
        return "Flowise nao configurado. Ajuste o .env para habilitar o chat."

    headers = {"Authorization": f"Bearer {FLOWISE_API_TOKEN}"}
    formatted_question = f"{ABNT_PROMPT_PREFIX}\n\nPergunta: {question}"
    payload = {"question": formatted_question, "streaming": True}
    timeout = (REQUEST_CONNECT_TIMEOUT_SECONDS, REQUEST_READ_TIMEOUT_SECONDS)

    with requests.post(
        FLOWISE_API_URL,
        headers=headers,
        json=payload,
        timeout=timeout,
        stream=True,
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            full_text = ""
            for chunk in iterate_sse_chunks(response):
                if st.session_state.cancel_requested:
                    break
                if chunk:
                    full_text += chunk
                    placeholder.markdown(full_text)
            if full_text.strip():
                return repair_mojibake(full_text)

        # Fallback sem streaming real: reaproveita a mesma resposta para evitar nova requisicao.
        try:
            answer = extract_answer(response.json())
        except ValueError:
            answer = extract_answer(response.text)
        return fake_stream(answer, placeholder)


def ensure_state():
    if "chats" not in st.session_state:
        st.session_state.chats = []
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "cancel_requested" not in st.session_state:
        st.session_state.cancel_requested = False
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "duckdb_query" not in st.session_state:
        st.session_state.duckdb_query = (
            "SELECT categoria, fornecedor, SUM(consumo_medio_6m) AS consumo_6m_total,\n"
            "AVG(custo_aquisicao_unitario) AS custo_medio\n"
            "FROM vw_produtos_demo\n"
            "GROUP BY 1, 2\n"
            "ORDER BY consumo_6m_total DESC\n"
            "LIMIT 20;"
        )
    if not st.session_state.chats:
        create_new_chat()
    if "chroma_ingest" not in st.session_state:
        st.session_state.chroma_ingest = {
            "ultima_em": None,
            "duracao_s": None,
            "arquivos": None,
            "trechos": None,
            "substituiu_indice": None,
            "modo": None,
            "ok": None,
            "mensagem": "",
        }
    if "chroma_query" not in st.session_state:
        st.session_state.chroma_query = {
            "ultima_em": None,
            "duracao_s": None,
            "pedidos": None,
            "retornados": None,
            "dist_media": None,
            "consulta_preview": "",
            "ok": None,
            "erro": None,
        }


def create_new_chat():
    chat_id = str(uuid.uuid4())
    title = f"Chat {len(st.session_state.chats) + 1}"
    st.session_state.chats.append({"id": chat_id, "title": title, "messages": []})
    st.session_state.active_chat_id = chat_id


def set_active_chat(chat_id: str):
    st.session_state.active_chat_id = chat_id


def get_active_chat():
    for chat in st.session_state.chats:
        if chat["id"] == st.session_state.active_chat_id:
            return chat
    create_new_chat()
    return st.session_state.chats[-1]


def find_last_user_message(messages):
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return None


def get_dataset_files() -> list[Path]:
    source_dir = Path(DUCKDB_SOURCE_DIR)
    cosmeticos = sorted(source_dir.glob("insumos_cosmeticos_*.csv"))
    return cosmeticos or sorted(source_dir.glob("produtos_*.csv"))


@st.cache_resource
def get_duckdb_connection(database_path: str):
    db_path = Path(database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(database=str(db_path))


@st.cache_resource
def get_chroma_persistent_client(persist_directory: str):
    return chroma_rag.get_chroma_client(persist_directory)


def _sql_array(paths: list[Path]) -> str:
    escaped = [str(p).replace("\\", "/").replace("'", "''") for p in paths]
    return ", ".join(f"'{item}'" for item in escaped)


def materialize_demo_view(connection: duckdb.DuckDBPyConnection, selected_files: list[Path]):
    files_sql = _sql_array(selected_files)
    connection.execute(
        f"""
        CREATE OR REPLACE VIEW vw_produtos_demo AS
        SELECT *
        FROM read_csv_auto([{files_sql}], union_by_name=true, sample_size=-1)
        """
    )


def render_chat_tab():
    active_chat = get_active_chat()
    chat_scroll_container = st.container(height=540, border=False)
    with chat_scroll_container:
        for message in active_chat["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if st.session_state.pending_prompt and not st.session_state.is_processing:
            prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None
            st.session_state.cancel_requested = False
            st.session_state.is_processing = True
            answer = ""

            if len(active_chat["messages"]) == 0:
                active_chat["title"] = (prompt[:32] + "...") if len(prompt) > 35 else prompt

            active_chat["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    answer = query_flowise(prompt, placeholder)
                    if st.session_state.cancel_requested:
                        answer = "Requisicao cancelada no frontend."
                        placeholder.markdown(answer)
                except requests.exceptions.Timeout:
                    answer = (
                        "Flowise demorou para responder e excedeu o timeout configurado. "
                        "Aumente REQUEST_READ_TIMEOUT_SECONDS no .env ou revise os nos do fluxo."
                    )
                    placeholder.markdown(answer)
                except Exception as exc:
                    answer = f"Erro ao consultar Flowise: {exc}"
                    placeholder.markdown(answer)
                finally:
                    st.session_state.is_processing = False

            active_chat["messages"].append({"role": "assistant", "content": answer})


def render_duckdb_tab():
    st.subheader("Demo DuckDB + pandas")
    all_files = get_dataset_files()
    if not all_files:
        st.warning(
            "Nenhum dataset encontrado em 'data/'. "
            "Execute `python generate_mock_data.py` para gerar os arquivos ficticios."
        )
        return

    default_files = all_files[:3]
    selected_files = st.multiselect(
        "Arquivos usados na demonstracao (recomendado: 3 arquivos)",
        options=all_files,
        default=default_files,
        format_func=lambda p: p.name,
    )
    if not selected_files:
        st.info("Selecione ao menos um arquivo para consultar no DuckDB.")
        return

    connection = get_duckdb_connection(DUCKDB_DATABASE_PATH)
    materialize_demo_view(connection, selected_files)

    total_rows = connection.execute("SELECT COUNT(*) FROM vw_produtos_demo").fetchone()[0]
    total_fornecedores = connection.execute("SELECT COUNT(DISTINCT fornecedor) FROM vw_produtos_demo").fetchone()[0]
    custo_medio = connection.execute("SELECT AVG(custo_aquisicao_unitario) FROM vw_produtos_demo").fetchone()[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Linhas no conjunto", f"{total_rows:,}".replace(",", "."))
    c2.metric("Fornecedores unicos", f"{total_fornecedores:,}".replace(",", "."))
    c3.metric("Custo medio", f"R$ {custo_medio:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    tab_tabelas, tab_predefinidas, tab_sql_custom = st.tabs(
        ["Tabelas", "Consultas pre-definidas", "SQL customizado"]
    )

    with tab_tabelas:
        st.caption("Estrutura e amostra da view `vw_produtos_demo`.")
        schema_df = connection.execute("DESCRIBE vw_produtos_demo").df()
        st.dataframe(schema_df, use_container_width=True)
        preview_df: pd.DataFrame = connection.execute("SELECT * FROM vw_produtos_demo LIMIT 20").df()
        st.dataframe(preview_df, use_container_width=True)

    with tab_predefinidas:
        consultas_predefinidas = {
            "Top categorias por impacto financeiro anual": """
                SELECT
                    categoria,
                    SUM(consumo_medio_12m * custo_aquisicao_unitario) AS impacto_anual
                FROM vw_produtos_demo
                GROUP BY 1
                ORDER BY 2 DESC
                LIMIT 10
            """,
            "Fornecedores com maior volume de consumo (6m)": """
                SELECT
                    fornecedor,
                    SUM(consumo_medio_6m) AS consumo_total_6m,
                    AVG(custo_aquisicao_unitario) AS custo_medio
                FROM vw_produtos_demo
                GROUP BY 1
                ORDER BY 2 DESC
                LIMIT 15
            """,
            "Insumos criticos com maior custo unitario": """
                SELECT
                    insumo,
                    categoria,
                    fornecedor,
                    criticidade_insumo,
                    custo_aquisicao_unitario
                FROM vw_produtos_demo
                WHERE lower(criticidade_insumo) = 'alta'
                ORDER BY custo_aquisicao_unitario DESC
                LIMIT 20
            """,
        }
        consulta_nome = st.selectbox(
            "Escolha uma consulta pre-definida",
            options=list(consultas_predefinidas.keys()),
        )
        st.code(consultas_predefinidas[consulta_nome].strip(), language="sql")
        if st.button("Executar consulta pre-definida", key="run_predefined_query"):
            try:
                df_predef = connection.execute(consultas_predefinidas[consulta_nome]).df()
                st.dataframe(df_predef, use_container_width=True)
                if not df_predef.empty:
                    st.download_button(
                        "Baixar resultado (CSV)",
                        data=df_predef.to_csv(index=False).encode("utf-8"),
                        file_name="resultado_predefinido_duckdb.csv",
                        mime="text/csv",
                        key="download_predefined_query",
                    )
            except Exception as exc:
                st.error(f"Falha ao executar consulta pre-definida: {exc}")

    with tab_sql_custom:
        st.caption("Consulta SQL livre no DuckDB com retorno em pandas DataFrame.")
        st.session_state.duckdb_query = st.text_area(
            "SQL de consulta",
            value=st.session_state.duckdb_query,
            height=190,
        )
        if st.button("Executar SQL customizado", key="run_custom_query"):
            try:
                df_result = connection.execute(st.session_state.duckdb_query).df()
                st.dataframe(df_result, use_container_width=True)
                if not df_result.empty:
                    st.download_button(
                        "Baixar resultado (CSV)",
                        data=df_result.to_csv(index=False).encode("utf-8"),
                        file_name="resultado_duckdb.csv",
                        mime="text/csv",
                        key="download_custom_query",
                    )
            except Exception as exc:
                st.error(f"Falha ao executar SQL customizado no DuckDB: {exc}")


def _fmt_bytes(num: int) -> str:
    if num < 1024:
        return f"{num} B"
    if num < 1024**2:
        return f"{num / 1024:.1f} KB"
    if num < 1024**3:
        return f"{num / 1024**2:.1f} MB"
    return f"{num / 1024**3:.2f} GB"


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _openai_key_plausible(key: str) -> bool:
    """Alinha com config: chave OpenAI de API costuma comecar com sk-."""
    k = (key or "").strip()
    if len(k) >= 2 and k[0] == k[-1] and k[0] in "\"'":
        k = k[1:-1].strip()
    return len(k) > 8 and k.startswith("sk-")


def _openai_401_hint(message: str) -> str:
    m = message.lower()
    if "401" in message or "invalid_api_key" in m or "incorrect api key" in m:
        return (
            " Confirme **OPENAI_API_KEY** em `streamlit/.env.chroma` (ou `.env`): deve ser uma chave de API da OpenAI "
            "(prefixo `sk-`), criada em https://platform.openai.com/api-keys — **nao** use `FLOWISE_API_TOKEN` nem outras chaves."
        )
    return ""


def render_chroma_tab():
    st.subheader("ChromaDB (diagnostico local)")
    st.caption(
        "Base vetorial embutida no mesmo processo do Streamlit, com persistencia em disco. "
        "Embeddings gerados pela OpenAIEmbeddingFunction do Chroma (nao usa o embedding padrao interno)."
    )

    knowledge_dir = Path(KNOWLEDGE_TXT_DIR)
    txt_files = chroma_rag.list_knowledge_files(knowledge_dir)
    st.markdown(
        f"Pasta de conhecimento: `{KNOWLEDGE_TXT_DIR}` — **{len(txt_files)}** arquivo(s) `.txt` / `.md`."
    )
    if txt_files:
        with st.expander("Arquivos detectados para indexacao"):
            for fp in txt_files:
                st.text(fp.name)

    st.markdown(
        f"Persistencia: `{CHROMA_PERSIST_DIRECTORY}` · Colecao: `{CHROMA_COLLECTION_NAME}` · "
        f"Modelo de embedding: `{OPENAI_EMBEDDING_MODEL}`"
    )

    if not OPENAI_API_KEY:
        st.warning(
            "Para indexar e consultar, defina **OPENAI_API_KEY** no ambiente (por exemplo em `streamlit/.env.chroma` "
            "referenciado pelo Docker Compose). Nao armazene chaves no codigo."
        )
        return

    if not _openai_key_plausible(OPENAI_API_KEY):
        st.error(
            "**OPENAI_API_KEY** nao parece uma chave da API OpenAI (esperado prefixo `sk-`, por exemplo `sk-proj-...`). "
            "O valor atual pode ser o token do Flowise ou outro segredo — a ingestao e a consulta vao falhar com erro 401."
        )
        st.info(
            "Crie uma chave em [platform.openai.com/api-keys](https://platform.openai.com/api-keys), coloque-a em "
            "`docker/streamlit/.env.chroma` como `OPENAI_API_KEY=sk-...` (sem aspas), guarde o ficheiro e execute "
            "`docker compose up -d streamlit --force-recreate` na pasta `docker`."
        )
        return

    client = get_chroma_persistent_client(CHROMA_PERSIST_DIRECTORY)

    try:
        ef = chroma_rag.make_openai_embedding_function(OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL)
        collection = chroma_rag.get_collection(client, CHROMA_COLLECTION_NAME, ef)
    except Exception as exc:
        st.error(f"Nao foi possivel abrir a colecao Chroma: {exc}")
        return

    st.markdown("**Painel: Estatísticas · Ingestão · Consulta**")
    col_est, col_ing, col_q = st.columns(3)
    ingest = st.session_state.chroma_ingest
    qstat = st.session_state.chroma_query
    persist_sz = chroma_rag.persist_directory_size_bytes(CHROMA_PERSIST_DIRECTORY)
    n_trechos = collection.count()

    with col_est:
        st.markdown("**Estatísticas**")
        st.metric("Trechos na colecao", n_trechos)
        st.metric("Arquivos fonte (.txt/.md)", len(txt_files))
        st.metric("Disco (pasta Chroma)", _fmt_bytes(persist_sz))
        st.caption(f"Colecao `{CHROMA_COLLECTION_NAME}` · modelo `{OPENAI_EMBEDDING_MODEL}`")

    with col_ing:
        st.markdown("**Ingestão**")
        st.metric("Ultima duracao (s)", ingest["duracao_s"] if ingest["duracao_s"] is not None else "—")
        c1, c2 = st.columns(2)
        c1.metric("Arquivos", ingest["arquivos"] if ingest["arquivos"] is not None else "—")
        c2.metric("Trechos gravados", ingest["trechos"] if ingest["trechos"] is not None else "—")
        subst = ingest["substituiu_indice"]
        subst_lbl = "Sim" if subst is True else "Nao" if subst is False else "—"
        modo = ingest.get("modo") or "—"
        modo_lbl = {"ingerir": "Ingerir (upsert)", "recriar_e_ingerir": "Recriar e ingerir"}.get(modo, modo)
        st.caption(
            f"Horario: {_fmt_ts(ingest['ultima_em'])} · Modo: {modo_lbl} · Indice substituido: {subst_lbl}"
        )
        if ingest.get("ok") is False and ingest.get("mensagem"):
            st.caption(f"Erro: {ingest['mensagem'][:120]}")

    with col_q:
        st.markdown("**Consulta**")
        st.metric("Ultima duracao (s)", qstat["duracao_s"] if qstat["duracao_s"] is not None else "—")
        c3, c4 = st.columns(2)
        c3.metric("Trechos pedidos", qstat["pedidos"] if qstat["pedidos"] is not None else "—")
        c4.metric("Trechos retornados", qstat["retornados"] if qstat["retornados"] is not None else "—")
        dist_m = qstat["dist_media"]
        st.metric("Distancia media", f"{dist_m:.4f}" if isinstance(dist_m, (int, float)) else "—")
        preview = (qstat.get("consulta_preview") or "").strip()
        if preview:
            st.caption(f"Texto: {preview[:80]}{'…' if len(preview) > 80 else ''}")
        st.caption(f"Horario: {_fmt_ts(qstat['ultima_em'])}")
        if qstat.get("erro"):
            st.caption(f"Erro: {qstat['erro'][:160]}")

    st.divider()
    st.markdown("**Ingestão de documentos**")
    st.caption(
        "**Ingerir documentos** — abre a coleção e faz *upsert* dos trechos dos ficheiros da pasta "
        "(conteúdo alterado é atualizado; vetores antigos de ficheiros apagados podem ficar até recriar).  \n"
        "**Recriar e ingerir** — apaga a coleção, cria outra vazia e indexa tudo de novo (índice alinhado só com os ficheiros atuais)."
    )
    b_ingest, b_recreate = st.columns(2)
    with b_ingest:
        do_ingest = st.button("Ingerir documentos", type="primary", use_container_width=True, key="chroma_ingest_upsert")
    with b_recreate:
        do_recreate = st.button(
            "Recriar e ingerir",
            type="secondary",
            use_container_width=True,
            key="chroma_ingest_recreate",
        )

    def _executar_ingestao(recreate: bool, modo_label: str) -> None:
        t0 = time.perf_counter()
        try:
            ef_ing = chroma_rag.make_openai_embedding_function(OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL)
            if recreate:
                collection = chroma_rag.reset_collection(client, CHROMA_COLLECTION_NAME, ef_ing)
            else:
                collection = chroma_rag.get_collection(client, CHROMA_COLLECTION_NAME, ef_ing)
            n_files, n_chunks = chroma_rag.index_knowledge_dir(collection, knowledge_dir)
            elapsed = time.perf_counter() - t0
            st.session_state.chroma_ingest = {
                "ultima_em": time.time(),
                "duracao_s": round(elapsed, 3),
                "arquivos": n_files,
                "trechos": n_chunks,
                "substituiu_indice": recreate,
                "modo": modo_label,
                "ok": True,
                "mensagem": "",
            }
            if n_files == 0:
                st.warning("Nenhum .txt/.md encontrado na pasta configurada.")
            else:
                acao = "Colecao recriada e indexada" if recreate else "Documentos ingeridos (upsert)"
                st.success(f"{acao}: {n_files} arquivo(s), {n_chunks} trecho(s) gravados nesta operacao.")
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            st.session_state.chroma_ingest = {
                "ultima_em": time.time(),
                "duracao_s": round(elapsed, 3),
                "arquivos": None,
                "trechos": None,
                "substituiu_indice": recreate,
                "modo": modo_label,
                "ok": False,
                "mensagem": str(exc),
            }
            st.error(f"Falha na ingestao: {exc}{_openai_401_hint(str(exc))}")

    if do_ingest:
        _executar_ingestao(recreate=False, modo_label="ingerir")
    if do_recreate:
        _executar_ingestao(recreate=True, modo_label="recriar_e_ingerir")

    if do_ingest or do_recreate:
        collection = chroma_rag.get_collection(client, CHROMA_COLLECTION_NAME, ef)

    st.divider()
    st.markdown("**Consulta por similaridade**")

    query_text = st.text_input(
        "Texto da busca",
        placeholder="Ex.: prazo de validade no recebimento, dwell time, custo de armazenagem extra",
        key="chroma_query_text",
    )
    n_results = st.slider("Quantidade de trechos retornados", min_value=1, max_value=15, value=5, key="chroma_n")

    if st.button("Executar busca", key="chroma_run_query"):
        if not query_text.strip():
            st.info("Informe um texto para buscar.")
        elif collection.count() == 0:
            st.session_state.chroma_query = {
                "ultima_em": time.time(),
                "duracao_s": None,
                "pedidos": n_results,
                "retornados": 0,
                "dist_media": None,
                "consulta_preview": query_text.strip()[:200],
                "ok": False,
                "erro": "Colecao vazia",
            }
            st.warning("Colecao vazia. Execute a indexacao acima.")
        else:
            t0 = time.perf_counter()
            try:
                raw = chroma_rag.query_similar(collection, query_text.strip(), n_results)
                elapsed = time.perf_counter() - t0
                docs_batch = raw.get("documents") or []
                metas_batch = raw.get("metadatas") or []
                dists_batch = raw.get("distances") or []
                docs = docs_batch[0] if docs_batch else []
                metas = metas_batch[0] if metas_batch else []
                dists = dists_batch[0] if dists_batch else []
                dist_nums = [d for d in dists if isinstance(d, (int, float))]
                dist_media = sum(dist_nums) / len(dist_nums) if dist_nums else None
                st.session_state.chroma_query = {
                    "ultima_em": time.time(),
                    "duracao_s": round(elapsed, 3),
                    "pedidos": n_results,
                    "retornados": len(docs),
                    "dist_media": dist_media,
                    "consulta_preview": query_text.strip()[:200],
                    "ok": True,
                    "erro": None,
                }
                if not docs:
                    st.info("Nenhum trecho retornado.")
                else:
                    for i, doc in enumerate(docs):
                        meta = metas[i] if i < len(metas) else {}
                        dist = dists[i] if i < len(dists) else None
                        src = meta.get("source", "?")
                        chunk_i = meta.get("chunk_index", "?")
                        dist_s = f"{dist:.4f}" if isinstance(dist, (int, float)) else str(dist)
                        st.markdown(f"**Trecho {i + 1}** — `{src}` (parte {chunk_i}) · distancia: `{dist_s}`")
                        st.markdown(doc)
                        st.divider()
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                st.session_state.chroma_query = {
                    "ultima_em": time.time(),
                    "duracao_s": round(elapsed, 3),
                    "pedidos": n_results,
                    "retornados": None,
                    "dist_media": None,
                    "consulta_preview": query_text.strip()[:200],
                    "ok": False,
                    "erro": str(exc),
                }
                st.error(f"Falha na busca: {exc}{_openai_401_hint(str(exc))}")


st.set_page_config(page_title="Assistente de negociacao", page_icon="💬", layout="wide")
ensure_state()

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 15% 20%, #23395d 0%, rgba(35,57,93,0.08) 35%, transparent 55%),
            radial-gradient(circle at 85% 30%, #203354 0%, rgba(32,51,84,0.10) 32%, transparent 56%),
            linear-gradient(135deg, #192841 0%, #152238 40%, #203354 72%, #23395d 100%);
        color: #e6eef8;
    }

    .stApp::before,
    .stApp::after {
        content: "";
        position: fixed;
        left: 0;
        width: 100%;
        pointer-events: none;
        z-index: 0;
    }

    .stApp::before {
        bottom: 0;
        height: 36vh;
        background:
            radial-gradient(120% 90% at 50% 120%, rgba(255,255,255,0.09) 0%, rgba(255,255,255,0.0) 60%),
            radial-gradient(140% 95% at 30% 130%, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.0) 62%),
            radial-gradient(130% 85% at 70% 130%, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.0) 58%);
    }

    .stApp::after {
        bottom: 0;
        height: 24vh;
        opacity: 0.75;
        background:
            radial-gradient(130% 85% at 20% 125%, rgba(255,255,255,0.11) 0%, rgba(255,255,255,0.0) 58%),
            radial-gradient(130% 85% at 80% 125%, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.0) 58%);
    }

    .block-container {
        position: relative;
        z-index: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 1200;
        background: rgba(21, 34, 56, 0.90);
        backdrop-filter: blur(8px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.10);
        padding: 0.35rem 0.25rem 0.65rem 0.25rem;
        margin-bottom: 0.65rem;
    }
    .sticky-header h2 {
        margin: 0 0 0.12rem 0;
        font-size: 1.38rem;
        color: #eaf2ff;
    }
    .sticky-header p {
        margin: 0;
        color: #dbe8ff;
        font-size: 0.94rem;
    }
    [data-testid="stTabs"] {
        position: sticky;
        top: 74px;
        z-index: 1150;
        background: rgba(21, 34, 56, 0.86);
        backdrop-filter: blur(6px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 0.28rem;
    }
    [data-testid="stSidebar"] > div:first-child {
        position: fixed;
        top: 0;
        left: 0;
        height: 100vh;
        width: inherit;
        background: rgba(21, 34, 56, 0.96);
        border-right: 1px solid rgba(255, 255, 255, 0.10);
    }
    [data-testid="stSidebarContent"] {
        height: 100vh;
        overflow-y: auto;
        overflow-x: hidden;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sticky-header">
        <h2>Bate-papo e analise de compras</h2>
        <p>Chat com Flowise + demonstracao DuckDB e pandas para consultas de negociacao.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("Conversas")
    st.button("Novo chat", on_click=create_new_chat, use_container_width=True)
    if st.button("Retry ultima pergunta", use_container_width=True, disabled=st.session_state.is_processing):
        last_user = find_last_user_message(get_active_chat()["messages"])
        if last_user:
            st.session_state.pending_prompt = last_user
            st.rerun()
    if st.button("Cancelar requisicao", use_container_width=True, disabled=not st.session_state.is_processing):
        st.session_state.cancel_requested = True
    st.divider()
    for chat in st.session_state.chats:
        st.button(
            chat["title"],
            key=f"chat_{chat['id']}",
            on_click=set_active_chat,
            args=(chat["id"],),
            use_container_width=True,
            type="primary" if chat["id"] == st.session_state.active_chat_id else "secondary",
        )
    st.divider()
    st.caption(f"DuckDB: {DUCKDB_DATABASE_PATH}")
    st.caption(f"Fonte CSV: {DUCKDB_SOURCE_DIR}")
    st.caption(f"Chroma: {CHROMA_PERSIST_DIRECTORY}")

if not FLOWISE_READY:
    st.warning(
        "Chat indisponivel: configure FLOWISE_API_URL e FLOWISE_API_TOKEN no .env. "
        f"Detalhe tecnico: {FLOWISE_CONFIG_ERROR}"
    )

tab_chat, tab_analytics, tab_chroma = st.tabs(
    ["Chat", "Analytics (DuckDB + pandas)", "ChromaDB (diagnostico)"]
)
with tab_chat:
    render_chat_tab()
with tab_analytics:
    render_duckdb_tab()
with tab_chroma:
    render_chroma_tab()

with tab_chat:
    user_prompt = st.chat_input("Digite sua mensagem", disabled=not FLOWISE_READY)
    if user_prompt:
        st.session_state.pending_prompt = user_prompt
        st.rerun()
