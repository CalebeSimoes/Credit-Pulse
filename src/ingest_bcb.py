import sqlite3
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "bcb_series.yaml"
DB_PATH = BASE_DIR / "data" / "creditpulse_bcb.db"


def fetch_sgs_series(base_url: str, code: int, start_date: str) -> pd.DataFrame:
    """Consulta uma série temporal no SGS/BCB e retorna DataFrame padronizado por mês."""
    end_date = datetime.now().strftime("%d/%m/%Y")
    url = base_url.format(code=code)
    params = {"formato": "json", "dataInicial": start_date, "dataFinal": end_date}

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = pd.to_numeric(df["valor"], errors="coerce")

    # Normaliza para o primeiro dia do mês (granularidade mensal do Star Schema)
    df["data_ref"] = df["data"].dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m-%d")
    df_monthly = df.groupby("data_ref", as_index=False)["valor"].mean()
    return df_monthly


def build_star_schema(conn: sqlite3.Connection) -> None:
    """Cria as tabelas do Star Schema no SQLite."""
    cursor = conn.cursor()
    cursor.executescript(
        """
        DROP TABLE IF EXISTS fato_indicadores_credito;
        DROP TABLE IF EXISTS dim_modalidades;
        DROP TABLE IF EXISTS dim_tempo;

        CREATE TABLE dim_modalidades (
            id_modalidade INTEGER PRIMARY KEY,
            nome_modalidade TEXT NOT NULL,
            segmento TEXT NOT NULL,
            tipo_recurso TEXT NOT NULL
        );

        CREATE TABLE dim_tempo (
            data_ref TEXT PRIMARY KEY,
            ano INTEGER NOT NULL,
            mes INTEGER NOT NULL,
            trimestre INTEGER NOT NULL,
            semestre INTEGER NOT NULL
        );

        CREATE TABLE fato_indicadores_credito (
            id_fato INTEGER PRIMARY KEY AUTOINCREMENT,
            data_ref TEXT NOT NULL,
            id_modalidade INTEGER NOT NULL,
            taxa_juros_anual REAL,
            taxa_inadimplencia_90d REAL,
            volume_concessoes_milhoes REAL,
            taxa_selic_meta REAL,
            ipca_mensal REAL,
            FOREIGN KEY (data_ref) REFERENCES dim_tempo(data_ref),
            FOREIGN KEY (id_modalidade) REFERENCES dim_modalidades(id_modalidade)
        );
        """
    )
    conn.commit()


def run_ingestion() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_url = config["api_base_url"]
    start_date = config["start_date"]

    print("1/3 Baixando indicadores macroeconômicos (Selic e IPCA)...")
    df_selic = fetch_sgs_series(base_url, config["macro_series"]["taxa_selic_meta"], start_date)
    df_selic.rename(columns={"valor": "taxa_selic_meta"}, inplace=True)

    df_ipca = fetch_sgs_series(base_url, config["macro_series"]["ipca_mensal"], start_date)
    df_ipca.rename(columns={"valor": "ipca_mensal"}, inplace=True)

    df_macro = pd.merge(df_selic, df_ipca, on="data_ref", how="outer")

    print("2/3 Baixando séries por modalidade de crédito...")
    dim_modalidades_rows = []
    fato_frames = []

    for mod in config["modalidades"]:
        mod_id = mod["id_modalidade"]
        print(f"   -> Coletando: {mod['nome_modalidade']}...")

        dim_modalidades_rows.append(
            {
                "id_modalidade": mod_id,
                "nome_modalidade": mod["nome_modalidade"],
                "segmento": mod["segmento"],
                "tipo_recurso": mod["tipo_recurso"],
            }
        )

        df_juros = fetch_sgs_series(base_url, mod["series"]["taxa_juros_anual"], start_date).rename(
            columns={"valor": "taxa_juros_anual"}
        )
        df_inad = fetch_sgs_series(base_url, mod["series"]["taxa_inadimplencia_90d"], start_date).rename(
            columns={"valor": "taxa_inadimplencia_90d"}
        )
        df_conc = fetch_sgs_series(base_url, mod["series"]["volume_concessoes_milhoes"], start_date).rename(
            columns={"valor": "volume_concessoes_milhoes"}
        )

        df_mod = df_juros.merge(df_inad, on="data_ref", how="outer").merge(df_conc, on="data_ref", how="outer")
        df_mod["id_modalidade"] = mod_id
        fato_frames.append(df_mod)

    df_fato = pd.concat(fato_frames, ignore_index=True)
    df_fato = df_fato.merge(df_macro, on="data_ref", how="left")

    # Monta dim_tempo a partir das datas únicas
    datas = pd.to_datetime(df_fato["data_ref"].unique())
    df_tempo = pd.DataFrame({"data_ref": datas.strftime("%Y-%m-%d")})
    df_tempo["ano"] = datas.year
    df_tempo["mes"] = datas.month
    df_tempo["trimestre"] = datas.quarter
    df_tempo["semestre"] = (datas.month - 1) // 6 + 1

    print("3/3 Gravando Star Schema no SQLite...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        build_star_schema(conn)
        pd.DataFrame(dim_modalidades_rows).to_sql("dim_modalidades", conn, if_exists="append", index=False)
        df_tempo.sort_values("data_ref").to_sql("dim_tempo", conn, if_exists="append", index=False)
        df_fato.to_sql("fato_indicadores_credito", conn, if_exists="append", index=False)

    print(f"Concluído! Banco gerado em {DB_PATH} com {len(df_fato)} registros na tabela fato.")


if __name__ == "__main__":
    run_ingestion()