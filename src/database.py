import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError
from sqlglot import ErrorLevel, exp

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = Path(
    os.getenv("CREDITPULSE_DB_PATH", str(BASE_DIR / "data" / "creditpulse_bcb.db"))
)

ALLOWED_TABLES = frozenset(
    {
        "fato_indicadores_credito",
        "fato_macro_mensal",
        "dim_modalidades",
        "dim_tempo",
    }
)

DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_MAX_ROWS = 1000

_FORBIDDEN_NODE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Create",
    "Alter",
    "Command",
    "Merge",
    "Transaction",
    "Commit",
    "Rollback",
    "Set",
    "Pragma",
    "Attach",
    "Detach",
    "LoadData",
)
FORBIDDEN_EXPRESSIONS = tuple(
    node for name in _FORBIDDEN_NODE_NAMES if (node := getattr(exp, name, None)) is not None
)

BLOCKED_FUNCTIONS = frozenset({"load_extension", "sqlite_version", "sqlite_source_id"})


class UnsafeQueryError(ValueError):
    """Levantada quando a query viola as regras de segurança da sandbox SQL."""


class QueryExecutionError(RuntimeError):
    """Levantada quando o SQLite falha ao executar ou interrompe a query."""


def validate_sql_query(
    sql_query: str,
    allowed_tables: frozenset[str] = ALLOWED_TABLES,
) -> str:
    if not sql_query or not sql_query.strip():
        raise UnsafeQueryError("Query vazia não é permitida.")

    try:
        statements = [
            stmt
            for stmt in sqlglot.parse(
                sql_query, read="sqlite", error_level=ErrorLevel.RAISE
            )
            if stmt is not None
        ]
    except ParseError as exc:
        raise QueryExecutionError(f"Erro de sintaxe SQL: {exc}") from exc

    if len(statements) != 1:
        raise UnsafeQueryError("Apenas uma instrução SQL por consulta é permitida.")

    tree = statements[0]

    if not isinstance(tree, exp.Query):
        raise UnsafeQueryError("Apenas consultas de leitura (SELECT / WITH) são permitidas.")

    for node in tree.walk():
        if isinstance(node, FORBIDDEN_EXPRESSIONS):
            raise UnsafeQueryError(f"Operação não permitida na sandbox: {type(node).__name__}.")

    # Bloqueia funções internas ou table-valued pragmas (ex.: pragma_table_info)
    for func in tree.find_all((exp.Anonymous, exp.Func)):
        func_name = (func.name or "").lower()
        if func_name in BLOCKED_FUNCTIONS or func_name.startswith(("pragma_", "sqlite_")):
            raise UnsafeQueryError(f"Função bloqueada na sandbox: {func_name}.")

    # Coleta nomes de CTEs para não confundi-los com tabelas reais do banco
    cte_names = {
        cte.alias_or_name.lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }

    referenced_tables = list(tree.find_all(exp.Table))
    for table in referenced_tables:
        if table.db and table.db.lower() != "main":
            raise UnsafeQueryError(f"Acesso a bancos externos não permitido: '{table.db}'.")

        table_name = (table.name or "").lower()
        if table_name and table_name not in allowed_tables and table_name not in cte_names:
            raise UnsafeQueryError(f"Tabela não autorizada: '{table.name}'.")

    return sql_query.strip()


def _connect_read_only(db_path: Path, timeout_seconds: float) -> sqlite3.Connection:
    resolved = db_path.resolve()
    if not resolved.exists():
        raise QueryExecutionError(f"Banco SQLite não encontrado em: {resolved}")

    # as_uri() trata corretamente caminhos do Windows com espaços
    uri = f"{resolved.as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
    conn.execute("PRAGMA query_only = ON;")

    deadline = time.monotonic() + timeout_seconds

    def _abort_on_timeout() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(_abort_on_timeout, 10_000)
    return conn


def execute_read_only_query(
    sql_query: str,
    db_path: Path = DEFAULT_DB_PATH,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> pd.DataFrame:
    cleaned_sql = validate_sql_query(sql_query)

    conn: sqlite3.Connection | None = None
    try:
        conn = _connect_read_only(db_path, timeout_seconds=timeout_seconds)
        cursor = conn.execute(cleaned_sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows)
        return pd.DataFrame(rows, columns=columns)

    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise QueryExecutionError(
                f"Consulta interrompida por exceder o limite de {timeout_seconds:.1f}s."
            ) from exc
        raise QueryExecutionError(f"Erro ao executar SQL no banco: {exc}") from exc

    except sqlite3.DatabaseError as exc:
        raise QueryExecutionError(f"Erro no banco SQLite: {exc}") from exc

    finally:
        if conn is not None:
            conn.set_progress_handler(None, 0)
            conn.close()