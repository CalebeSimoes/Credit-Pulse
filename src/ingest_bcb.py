import argparse
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "bcb_series.yaml"
DB_PATH = BASE_DIR / "data" / "creditpulse_bcb.db"

INCREMENTAL_LOOKBACK_MONTHS = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("creditpulse.ingest")


class DataQualityError(ValueError):
    """Levantada quando os dados extraídos falham nas checagens de integridade."""


def build_http_session() -> requests.Session:
    """Cria sessão HTTP com retry exponencial para instabilidades na API do BCB."""
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_sgs_series(
    session: requests.Session,
    base_url: str,
    code: int,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Consulta uma série no SGS/BCB e agrega para granularidade mensal (YYYY-MM-01)."""
    url = base_url.format(code=code)
    params = {"formato": "json", "dataInicial": start_date, "dataFinal": end_date}

    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if isinstance(payload, dict) and payload.get("error"):
        raise DataQualityError(f"API BCB retornou erro para série {code}: {payload['error']}")
    if not isinstance(payload, list) or len(payload) == 0:
        raise DataQualityError(f"Série {code} retornou vazia para o período {start_date} a {end_date}.")

    df = pd.DataFrame(payload)
    if "data" not in df.columns or "valor" not in df.columns:
        raise DataQualityError(f"Série {code} sem colunas esperadas ('data', 'valor').")

    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
    df = df.dropna(subset=["data", "valor"])

    if df.empty:
        raise DataQualityError(f"Série {code} ficou sem valores numéricos válidos após limpeza.")

    # Normaliza para o 1º dia do mês (para séries diárias como a Selic 432, usa média mensal)
    df["data_ref"] = df["data"].dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")
    return df.groupby("data_ref", as_index=False)["valor"].mean()


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Garante o Constellation Schema sem destruir tabelas existentes."""
    conn.execute("PRAGMA foreign_keys = ON;")

    # Detecta se o arquivo .db ainda está no schema antigo (antes da separação da fato_macro_mensal)
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(fato_indicadores_credito);").fetchall()
    }
    if existing_cols and "taxa_selic_meta" in existing_cols:
        raise RuntimeError(
            "Schema legado detectado em 'fato_indicadores_credito' (ainda contém colunas macro). "
            "Faça backup ou remova o arquivo .db antigo uma única vez antes de rodar o novo pipeline."
        )

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS dim_modalidades (
            id_modalidade INTEGER PRIMARY KEY,
            nome_modalidade TEXT NOT NULL,
            segmento TEXT NOT NULL,
            tipo_recurso TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS dim_tempo (
            data_ref TEXT PRIMARY KEY,
            ano INTEGER NOT NULL,
            mes INTEGER NOT NULL,
            trimestre INTEGER NOT NULL,
            semestre INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS fato_macro_mensal (
            data_ref TEXT PRIMARY KEY,
            taxa_selic_meta REAL,
            ipca_mensal REAL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (data_ref) REFERENCES dim_tempo(data_ref)
        );

        CREATE TABLE IF NOT EXISTS fato_indicadores_credito (
            data_ref TEXT NOT NULL,
            id_modalidade INTEGER NOT NULL,
            taxa_juros_anual REAL,
            taxa_inadimplencia_90d REAL,
            volume_concessoes_milhoes REAL,
            saldo_carteira_milhoes REAL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (data_ref, id_modalidade),
            FOREIGN KEY (data_ref) REFERENCES dim_tempo(data_ref),
            FOREIGN KEY (id_modalidade) REFERENCES dim_modalidades(id_modalidade)
        );
        """
    )


def resolve_start_date(db_path: Path, config_start_date: str, full_refresh: bool) -> str:
    """Define data inicial: carga completa ou janela incremental de revisão (últimos 3 meses)."""
    if full_refresh or not db_path.exists():
        return config_start_date

    with sqlite3.connect(db_path) as conn:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='fato_indicadores_credito';"
        ).fetchone()
        if not table_exists:
            return config_start_date

        max_date = conn.execute("SELECT MAX(data_ref) FROM fato_indicadores_credito;").fetchone()[0]
        if not max_date:
            return config_start_date

    lookback = pd.Timestamp(max_date) - pd.DateOffset(months=INCREMENTAL_LOOKBACK_MONTHS)
    start_ts = max(lookback, pd.to_datetime(config_start_date, format="%d/%m/%Y"))
    return start_ts.strftime("%d/%m/%Y")


def validate_staging_data(
    df_modalidades: pd.DataFrame,
    df_tempo: pd.DataFrame,
    df_macro: pd.DataFrame,
    df_credito: pd.DataFrame,
) -> None:
    """Valida integridade, unicidade de grão e faixas de sanidade antes de gravar no banco."""
    if df_modalidades.empty or df_tempo.empty or df_macro.empty or df_credito.empty:
        raise DataQualityError("Um ou mais DataFrames de staging estão vazios.")

    if df_modalidades["id_modalidade"].duplicated().any():
        raise DataQualityError("Duplicidade detectada em dim_modalidades.id_modalidade.")

    if df_tempo["data_ref"].duplicated().any():
        raise DataQualityError("Duplicidade detectada em dim_tempo.data_ref.")

    if df_macro["data_ref"].duplicated().any():
        raise DataQualityError("Duplicidade detectada em fato_macro_mensal.data_ref.")

    if df_credito.duplicated(subset=["data_ref", "id_modalidade"]).any():
        raise DataQualityError("Duplicidade de grão (data_ref, id_modalidade) em fato_indicadores_credito.")

    # Checa cobertura de chaves estrangeiras
    tempo_keys = set(df_tempo["data_ref"])
    if not set(df_macro["data_ref"]).issubset(tempo_keys):
        raise DataQualityError("Existem datas em fato_macro_mensal ausentes em dim_tempo.")
    if not set(df_credito["data_ref"]).issubset(tempo_keys):
        raise DataQualityError("Existem datas em fato_indicadores_credito ausentes em dim_tempo.")

    # Checagens de sanidade numérica
    if (df_credito["taxa_juros_anual"].dropna() < 0).any():
        raise DataQualityError("Valor negativo inválido em taxa_juros_anual.")
    inad = df_credito["taxa_inadimplencia_90d"].dropna()
    if ((inad < 0) | (inad > 100)).any():
        raise DataQualityError("Valor fora da faixa [0, 100] em taxa_inadimplencia_90d.")
    if (df_credito["volume_concessoes_milhoes"].dropna() < 0).any():
        raise DataQualityError("Valor negativo inválido em volume_concessoes_milhoes.")
    if "saldo_carteira_milhoes" in df_credito.columns:
        if (df_credito["saldo_carteira_milhoes"].dropna() < 0).any():
            raise DataQualityError("Valor negativo inválido em saldo_carteira_milhoes.")

    # Auditoria de meses com métricas faltantes (aviso operacional, não bloqueia meses parciais recentes)
    null_counts = df_credito[["taxa_juros_anual", "taxa_inadimplencia_90d", "volume_concessoes_milhoes"]].isna().sum()
    if null_counts.any():
        logger.warning("Valores nulos parciais encontrados em fato_indicadores_credito: %s", null_counts.to_dict())


def publish_via_staging(
    db_path: Path,
    df_modalidades: pd.DataFrame,
    df_tempo: pd.DataFrame,
    df_macro: pd.DataFrame,
    df_credito: pd.DataFrame,
) -> None:
    """Carrega dados em tabelas temporárias de staging e publica via UPSERT atômico."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)

    try:
        ensure_schema(conn)

        # Popula staging temporário (vive apenas nesta conexão)
        conn.executescript(
            """
            CREATE TEMP TABLE stg_dim_modalidades (
                id_modalidade INTEGER, nome_modalidade TEXT, segmento TEXT, tipo_recurso TEXT
            );
            CREATE TEMP TABLE stg_dim_tempo (
                data_ref TEXT, ano INTEGER, mes INTEGER, trimestre INTEGER, semestre INTEGER
            );
            CREATE TEMP TABLE stg_fato_macro_mensal (
                data_ref TEXT, taxa_selic_meta REAL, ipca_mensal REAL, updated_at TEXT
            );
            CREATE TEMP TABLE stg_fato_indicadores_credito (
                data_ref TEXT, id_modalidade INTEGER, taxa_juros_anual REAL,
                taxa_inadimplencia_90d REAL, volume_concessoes_milhoes REAL,
                saldo_carteira_milhoes REAL, updated_at TEXT
            );
            """
        )

        df_modalidades.to_sql("stg_dim_modalidades", conn, if_exists="append", index=False)
        df_tempo.to_sql("stg_dim_tempo", conn, if_exists="append", index=False)
        df_macro.to_sql("stg_fato_macro_mensal", conn, if_exists="append", index=False)
        df_credito.to_sql("stg_fato_indicadores_credito", conn, if_exists="append", index=False)

        # Publicação atômica: ou grava tudo ou faz rollback preservando o estado anterior
        conn.execute("BEGIN IMMEDIATE;")

        conn.execute(
            """
            INSERT INTO dim_modalidades (id_modalidade, nome_modalidade, segmento, tipo_recurso)
            SELECT id_modalidade, nome_modalidade, segmento, tipo_recurso
            FROM stg_dim_modalidades
            WHERE true
            ON CONFLICT(id_modalidade) DO UPDATE SET
                nome_modalidade = excluded.nome_modalidade,
                segmento = excluded.segmento,
                tipo_recurso = excluded.tipo_recurso;
            """
        )

        conn.execute(
            """
            INSERT INTO dim_tempo (data_ref, ano, mes, trimestre, semestre)
            SELECT data_ref, ano, mes, trimestre, semestre
            FROM stg_dim_tempo
            WHERE true
            ON CONFLICT(data_ref) DO UPDATE SET
                ano = excluded.ano,
                mes = excluded.mes,
                trimestre = excluded.trimestre,
                semestre = excluded.semestre;
            """
        )

        conn.execute(
            """
            INSERT INTO fato_macro_mensal (data_ref, taxa_selic_meta, ipca_mensal, updated_at)
            SELECT data_ref, taxa_selic_meta, ipca_mensal, updated_at
            FROM stg_fato_macro_mensal
            WHERE true
            ON CONFLICT(data_ref) DO UPDATE SET
                taxa_selic_meta = excluded.taxa_selic_meta,
                ipca_mensal = excluded.ipca_mensal,
                updated_at = excluded.updated_at;
            """
        )

        conn.execute(
            """
            INSERT INTO fato_indicadores_credito (
                data_ref, id_modalidade, taxa_juros_anual, taxa_inadimplencia_90d,
                volume_concessoes_milhoes, saldo_carteira_milhoes, updated_at
            )
            SELECT
                data_ref, id_modalidade, taxa_juros_anual, taxa_inadimplencia_90d,
                volume_concessoes_milhoes, saldo_carteira_milhoes, updated_at
            FROM stg_fato_indicadores_credito
            WHERE true
            ON CONFLICT(data_ref, id_modalidade) DO UPDATE SET
                taxa_juros_anual = excluded.taxa_juros_anual,
                taxa_inadimplencia_90d = excluded.taxa_inadimplencia_90d,
                volume_concessoes_milhoes = excluded.volume_concessoes_milhoes,
                saldo_carteira_milhoes = excluded.saldo_carteira_milhoes,
                updated_at = excluded.updated_at;
            """
        )

        conn.execute("COMMIT;")

    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK;")
        raise
    finally:
        conn.close()


def run_ingestion(full_refresh: bool = False) -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_url = config["api_base_url"]
    end_date = datetime.now().strftime("%d/%m/%Y")
    start_date = resolve_start_date(DB_PATH, config["start_date"], full_refresh=full_refresh)
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    mode = "COMPLETA" if full_refresh or start_date == config["start_date"] else "INCREMENTAL"
    logger.info("Iniciando ingestão BCB (%s) | janela: %s até %s", mode, start_date, end_date)

    session = build_http_session()

    logger.info("1/4 Baixando séries macroeconômicas (Selic e IPCA)...")
    df_selic = fetch_sgs_series(
        session, base_url, config["macro_series"]["taxa_selic_meta"], start_date, end_date
    ).rename(columns={"valor": "taxa_selic_meta"})
    df_ipca = fetch_sgs_series(
        session, base_url, config["macro_series"]["ipca_mensal"], start_date, end_date
    ).rename(columns={"valor": "ipca_mensal"})

    df_macro = pd.merge(df_selic, df_ipca, on="data_ref", how="outer").sort_values("data_ref")
    df_macro["updated_at"] = run_ts

    logger.info("2/4 Baixando séries por modalidade de crédito...")
    dim_modalidades_rows = []
    fato_frames = []

    for mod in config["modalidades"]:
        mod_id = mod["id_modalidade"]
        series_cfg = mod["series"]
        logger.info("   -> Coletando: %s", mod["nome_modalidade"])

        dim_modalidades_rows.append(
            {
                "id_modalidade": mod_id,
                "nome_modalidade": mod["nome_modalidade"],
                "segmento": mod["segmento"],
                "tipo_recurso": mod["tipo_recurso"],
            }
        )

        df_juros = fetch_sgs_series(
            session, base_url, series_cfg["taxa_juros_anual"], start_date, end_date
        ).rename(columns={"valor": "taxa_juros_anual"})
        df_inad = fetch_sgs_series(
            session, base_url, series_cfg["taxa_inadimplencia_90d"], start_date, end_date
        ).rename(columns={"valor": "taxa_inadimplencia_90d"})
        df_conc = fetch_sgs_series(
            session, base_url, series_cfg["volume_concessoes_milhoes"], start_date, end_date
        ).rename(columns={"valor": "volume_concessoes_milhoes"})

        df_mod = (
            df_juros.merge(df_inad, on="data_ref", how="outer")
            .merge(df_conc, on="data_ref", how="outer")
        )

        if "saldo_carteira_milhoes" in series_cfg and series_cfg["saldo_carteira_milhoes"]:
            df_saldo = fetch_sgs_series(
                session, base_url, series_cfg["saldo_carteira_milhoes"], start_date, end_date
            ).rename(columns={"valor": "saldo_carteira_milhoes"})
            df_mod = df_mod.merge(df_saldo, on="data_ref", how="outer")
        else:
            df_mod["saldo_carteira_milhoes"] = None

        df_mod["id_modalidade"] = mod_id
        df_mod["updated_at"] = run_ts
        fato_frames.append(df_mod)

    df_modalidades = pd.DataFrame(dim_modalidades_rows)
    df_credito = pd.concat(fato_frames, ignore_index=True)

    # dim_tempo cobre a união das datas de macro e crédito (evita erro de FK se macro publicar antes)
    all_dates = pd.to_datetime(sorted(set(df_credito["data_ref"]).union(set(df_macro["data_ref"]))))
    df_tempo = pd.DataFrame(
        {
            "data_ref": all_dates.strftime("%Y-%m-%d"),
            "ano": all_dates.year,
            "mes": all_dates.month,
            "trimestre": all_dates.quarter,
            "semestre": (all_dates.month - 1) // 6 + 1,
        }
    )

    logger.info("3/4 Executando validações de integridade em staging...")
    validate_staging_data(df_modalidades, df_tempo, df_macro, df_credito)

    logger.info("4/4 Publicando no SQLite via staging + UPSERT atômico...")
    publish_via_staging(DB_PATH, df_modalidades, df_tempo, df_macro, df_credito)

    with sqlite3.connect(DB_PATH) as conn:
        total_credito = conn.execute("SELECT COUNT(*) FROM fato_indicadores_credito;").fetchone()[0]
        total_macro = conn.execute("SELECT COUNT(*) FROM fato_macro_mensal;").fetchone()[0]

    logger.info(
        "Concluído! Banco: %s | fato_indicadores_credito: %d linhas | fato_macro_mensal: %d linhas",
        DB_PATH,
        total_credito,
        total_macro,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de ingestão BCB/SGS para o CreditPulse.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Força recarga histórica completa desde config.start_date.",
    )
    args = parser.parse_args()
    run_ingestion(full_refresh=args.full)