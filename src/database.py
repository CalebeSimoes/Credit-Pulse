import os
import re
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


class UnsafeQueryError(ValueError):
    """Disparada quando a query viola as regras de segurança read-only."""
    pass


class QueryExecutionError(RuntimeError):
    """Disparada quando o banco rejeita a query (sintaxe, coluna inexistente, etc.)."""
    pass


FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "TRUNCATE", "CREATE", "REPLACE", "GRANT", "REVOKE",
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "EXEC"
)

# \b garante palavra inteira: bloqueia "UPDATE", mas não bloqueia uma coluna "updated_at"
FORBIDDEN_PATTERN = re.compile(
    r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b",
    flags=re.IGNORECASE,
)


def get_engine(db_url: str | None = None):
    """
    Cria a conexão com o banco via SQLAlchemy.
    Exemplos de DATABASE_URL:
    - SQLite:     sqlite:///data/creditpulse.db
    - PostgreSQL: postgresql+psycopg2://usuario:senha@localhost:5432/creditpulse
    """
    url = db_url or os.getenv("DATABASE_URL", "sqlite:///data/creditpulse_bcb.db")
    return create_engine(url)


def validate_sql_safety(sql_query: str) -> str:
    """
    Segunda camada de guardrail (defesa em profundidade além do Pydantic):
    1. Bloqueia múltiplos comandos separados por ponto e vírgula (;).
    2. Remove comentários SQL antes de inspecionar o comando real.
    3. Garante início com SELECT ou WITH.
    4. Bloqueia comandos destrutivos mesmo escondidos em CTEs ou subqueries.
    """
    cleaned = sql_query.strip().rstrip(";").strip()

    if not cleaned:
        raise UnsafeQueryError("A query SQL está vazia.")

    # Bloqueia empilhamento (ex: SELECT 1; DROP TABLE fato_indicadores_credito)
    if ";" in cleaned:
        raise UnsafeQueryError("Múltiplos comandos SQL na mesma query não são permitidos.")

    # Limpa comentários (-- e /* */) para evitar bypass do filtro
    no_comments = re.sub(r"--.*?$", "", cleaned, flags=re.MULTILINE)
    no_comments = re.sub(r"/\*.*?\*/", "", no_comments, flags=re.DOTALL).strip()

    if not re.match(r"^(SELECT|WITH)\b", no_comments, flags=re.IGNORECASE):
        raise UnsafeQueryError("Apenas consultas de leitura (SELECT ou WITH) são permitidas.")

    match = FORBIDDEN_PATTERN.search(no_comments)
    if match:
        raise UnsafeQueryError(
            f"Comando proibido detectado na query: '{match.group(0).upper()}'."
        )

    return cleaned


def execute_read_only_query(sql_query: str, db_url: str | None = None) -> pd.DataFrame:
    """Valida a query, executa no Star Schema e devolve um DataFrame."""
    safe_sql = validate_sql_safety(sql_query)
    engine = get_engine(db_url)

    try:
        with engine.connect() as conn:
            return pd.read_sql_query(text(safe_sql), conn)
    except (SQLAlchemyError, pd.errors.DatabaseError) as exc:
        raise QueryExecutionError(f"Erro ao executar SQL no banco: {exc}") from exc