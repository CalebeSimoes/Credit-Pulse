import os
import time
from dataclasses import dataclass
from pathlib import Path

import ollama
import pandas as pd
import yaml

from src.charts import TIME_COLS, friendly_label, infer_chart_spec
from src.database import (
    QueryExecutionError,
    UnsafeQueryError,
    execute_read_only_query,
)
from src.schemas import ChartSpec, SQLResponse

BASE_DIR = Path(__file__).resolve().parent.parent
SEMANTIC_MODEL_PATH = BASE_DIR / "config" / "semantic_layer.yaml"


@dataclass
class AgentResult:
    """Pacote devolvido pelo agente para terminal ou interface Streamlit."""

    question: str
    sql_response: SQLResponse
    df: pd.DataFrame
    chart_spec: ChartSpec
    insight: str
    attempts: int


class CreditPulseAgent:
    def __init__(
        self,
        model: str | None = None,
        semantic_model_path: Path = SEMANTIC_MODEL_PATH,
        max_retries: int = 2,
        verbose: bool = True,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.semantic_data = yaml.safe_load(semantic_model_path.read_text(encoding="utf-8"))
        self.metric_metadata = self._extract_metric_metadata(self.semantic_data)
        self.compact_schema = self._build_compact_schema(self.semantic_data)
        self.max_retries = max_retries
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    @staticmethod
    def _extract_metric_metadata(data: dict) -> dict[str, dict]:
        """Lê tipo (rate/flow/stock), unidade e rótulo diretamente da camada semântica."""
        meta: dict[str, dict] = {}
        for table_info in data.get("tables", {}).values():
            for col_name, col_info in table_info.get("columns", {}).items():
                if col_info.get("role") == "metric":
                    meta[col_name] = {
                        "metric_type": col_info.get("metric_type", "unknown"),
                        "unit": col_info.get("unit", ""),
                        "diff_unit": col_info.get("diff_unit", ""),
                        "label": col_info.get("label", col_name),
                    }
        for metric_name, metric_info in data.get("metrics", {}).items():
            meta[metric_name] = {
                "metric_type": metric_info.get("metric_type", "unknown"),
                "unit": metric_info.get("unit", ""),
                "diff_unit": metric_info.get("diff_unit", ""),
                "label": metric_info.get("label", metric_name),
            }
        return meta

    def _resolve_metric_meta(self, col_name: str) -> dict:
        """Resolve metadados da coluna pelo nome oficial ou correspondência com métricas do YAML."""
        col_low = col_name.lower()
        if col_low in self.metric_metadata:
            return self.metric_metadata[col_low]

        # Suporta aliases gerados pela LLM (ex.: 'media_taxa_juros_anual', 'total_concessoes')
        for canonical_name, info in self.metric_metadata.items():
            if canonical_name in col_low:
                return info

        alias_to_canonical = {
            "juros": "taxa_juros_anual",
            "inadimplencia": "taxa_inadimplencia_90d",
            "selic": "taxa_selic_meta",
            "ipca": "ipca_mensal",
            "concess": "volume_concessoes_milhoes",
            "volume": "volume_concessoes_milhoes",
            "saldo": "saldo_carteira_milhoes",
            "carteira": "saldo_carteira_milhoes",
        }
        for token, canonical_name in alias_to_canonical.items():
            if token in col_low and canonical_name in self.metric_metadata:
                return self.metric_metadata[canonical_name]

        return {
            "metric_type": "unknown",
            "unit": "",
            "diff_unit": "",
            "label": friendly_label(col_name),
        }

    def _build_compact_schema(self, data: dict) -> str:
        """Compacta o semantic_layer.yaml para reduzir tokens e latência em CPU."""
        lines = ["SCHEMA SQLITE (BCB/SGS, desde 2018-01-01):"]

        for table_name, table_info in data["tables"].items():
            alias = table_info.get("alias", "")
            cols = []
            for col_name, col_info in table_info["columns"].items():
                allowed = col_info.get("allowed_values")
                if allowed:
                    cols.append(f"{col_name}={allowed}")
                else:
                    cols.append(col_name)
            header = f"- {table_name} (alias {alias})" if alias else f"- {table_name}"
            lines.append(f"{header}: " + ", ".join(cols))

        lines.append(
            "JOINS: f.data_ref=t.data_ref | f.id_modalidade=m.id_modalidade | "
            "fm.data_ref=t.data_ref | f.data_ref=fm.data_ref"
        )

        lines.append("METRICAS:")
        for m_name, m_info in data["metrics"].items():
            lines.append(f"- {m_name}: {m_info['sql']}")

        lines.append(
            "REGRAS:\n"
            "- Use SEMPRE o nome completo da tabela no FROM/JOIN antes do alias.\n"
            "- Para 'evolucao mensal', retorne apenas data_ref e a metrica ordenados por data_ref. NUNCA faca self-join.\n"
            "- Perguntas so sobre Selic/IPCA: consulte fato_macro_mensal fm (sem fato_indicadores_credito).\n"
            "- NUNCA use SUM() em taxas percentuais; NUNCA some saldo_carteira_milhoes entre meses diferentes.\n"
            "- Para ultimo mes disponivel, use subquery com MAX(data_ref) fechando todos os parenteses."
        )

        lines.append(
            "EXEMPLOS:\n"
            "P: Evolucao mensal do IPCA em 2024\n"
            "SQL: SELECT fm.data_ref, fm.ipca_mensal FROM fato_macro_mensal fm "
            "JOIN dim_tempo t ON fm.data_ref = t.data_ref WHERE t.ano = 2024 ORDER BY fm.data_ref\n"
            "P: Compare juros e concessoes das modalidades no ultimo mes disponivel\n"
            "SQL: SELECT m.nome_modalidade, f.taxa_juros_anual, f.volume_concessoes_milhoes "
            "FROM fato_indicadores_credito f JOIN dim_modalidades m ON f.id_modalidade = m.id_modalidade "
            "WHERE f.data_ref = (SELECT MAX(data_ref) FROM fato_indicadores_credito) "
            "ORDER BY f.taxa_juros_anual DESC\n"
            "P: Volume total de concessoes por ano\n"
            "SQL: SELECT t.ano, SUM(f.volume_concessoes_milhoes) AS volume_total_concessoes "
            "FROM fato_indicadores_credito f JOIN dim_tempo t ON f.data_ref = t.data_ref "
            "GROUP BY t.ano ORDER BY t.ano"
        )

        return "\n".join(lines)

    def _call_ollama(
        self,
        messages: list[dict],
        temperature: float,
        schema: dict | None = None,
        num_predict: int = 512,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": 4096,
            },
        }
        if schema is not None:
            kwargs["format"] = schema
        if any(tag in self.model.lower() for tag in ("qwen3", "deepseek-r1", "gpt-oss")):
            kwargs["think"] = False

        response = ollama.chat(**kwargs)
        return response["message"]["content"].strip()

    def _build_sql_system_prompt(self) -> str:
        return (
            "Especialista em SQLite para dados do Banco Central (BCB). "
            "Siga exatamente o estilo simples dos EXEMPLOS abaixo.\n"
            "- Retorne apenas colunas que existem no SCHEMA.\n"
            "- Feche todos os parenteses e aspas da query SQL.\n\n"
            f"{self.compact_schema}"
        )

    def _generate_sql_and_execute(self, question: str) -> tuple[SQLResponse, pd.DataFrame, int]:
        system_prompt = self._build_sql_system_prompt()
        self._log(f"[Contexto] Prompt SQL compactado: {len(system_prompt)} caracteres")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        last_error = None
        for attempt in range(1, self.max_retries + 2):
            self._log(f"[Etapa 1 - Tentativa {attempt}] Gerando SQL com {self.model}...")
            t0 = time.perf_counter()

            raw_json = self._call_ollama(
                messages=messages,
                schema=SQLResponse.model_json_schema(),
                temperature=0.0,
                num_predict=512,
            )
            self._log(f"   -> SQL gerado em {time.perf_counter() - t0:.1f}s")

            failed_sql = None
            try:
                sql_response = SQLResponse.model_validate_json(raw_json)
                failed_sql = sql_response.sql_query
                df = execute_read_only_query(sql_response.sql_query)
                self._log(f"   -> Query executada com sucesso ({len(df)} linhas retornadas)")
                return sql_response, df, attempt

            except UnsafeQueryError as exc:
                msg = str(exc)
                # Bloqueio imediato para ataques reais (escrita, stacked query, pragma, sqlite_*)
                if "Tabela não autorizada:" not in msg or "sqlite_" in msg.lower():
                    raise
                last_error = exc
                self._log(f"   -> Tabela/alias inválido na tentativa {attempt}: {exc}")
                if failed_sql:
                    self._log(f"   -> SQL gerado: {failed_sql}")
                messages.append({"role": "assistant", "content": raw_json})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Erro: {exc}\n"
                            "Escreva o nome COMPLETO da tabela no FROM/JOIN antes do alias."
                        ),
                    }
                )

            except (ValueError, QueryExecutionError) as exc:
                last_error = exc
                self._log(f"   -> Falha na tentativa {attempt}: {exc}")
                self._log(f"   -> SQL gerado: {failed_sql or raw_json}")
                messages.append({"role": "assistant", "content": raw_json})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Erro na query:\n{exc}\n"
                            "Corrija seguindo o formato simples dos EXEMPLOS e conferindo todos os parenteses."
                        ),
                    }
                )

        raise RuntimeError(
            f"Não foi possível gerar uma query válida após {self.max_retries + 1} tentativas. "
            f"Último erro: {last_error}"
        )

    @staticmethod
    def _format_value(value: float, unit: str) -> str:
        if unit.startswith("%"):
            return f"{value:.2f}{unit}"
        if "R$" in unit:
            return f"R$ {value:,.2f} mi"
        return f"{value:,.2f} {unit}".strip()

    def _build_metric_facts(self, df: pd.DataFrame) -> str:
        """Gera fatos numéricos tipados a partir dos metadados do semantic_layer.yaml."""
        metric_cols = [
            c
            for c in df.select_dtypes(include="number").columns
            if c not in TIME_COLS and not c.startswith("id_")
        ]
        label_cols = [c for c in df.columns if c not in metric_cols and not c.startswith("id_")]

        def _row_label(idx) -> str:
            if not label_cols:
                return f"linha {idx}"
            return " | ".join(str(df.loc[idx, c]) for c in label_cols)

        blocks = [f"Total de registros analisados: {len(df)}"]

        if len(df) == 1:
            row = df.iloc[0]
            items = []
            for c in df.columns:
                if c in metric_cols and pd.notna(row[c]):
                    meta = self._resolve_metric_meta(c)
                    items.append(f"{meta['label']}: {self._format_value(float(row[c]), meta['unit'])}")
                else:
                    items.append(f"{friendly_label(c)}: {row[c]}")
            blocks.append("Resultado único: " + " ; ".join(items))
            return "\n".join(blocks)

        has_time_series = any(t in df.columns for t in ("data_ref", "ano"))

        for col in metric_cols:
            valid = df[col].dropna()
            if valid.empty:
                continue

            meta = self._resolve_metric_meta(col)
            m_type = meta["metric_type"]
            unit = meta["unit"]
            diff_unit = meta["diff_unit"]

            idx_max, idx_min = valid.idxmax(), valid.idxmin()
            val_max, val_min = float(df.loc[idx_max, col]), float(df.loc[idx_min, col])
            diff = val_max - val_min

            lines = [
                f"\nMétrica: {meta['label']} (tipo: {m_type})",
                f"- MAIOR valor: {_row_label(idx_max)} = {self._format_value(val_max, unit)}",
                f"- MENOR valor: {_row_label(idx_min)} = {self._format_value(val_min, unit)}",
            ]

            if m_type == "rate":
                lines.append(f"- Diferença absoluta (Maior - Menor): {diff:.2f} {diff_unit or 'p.p.'}")
            elif m_type in ("flow", "stock"):
                lines.append(f"- Diferença absoluta (Maior - Menor): {self._format_value(diff, unit)}")
                if val_min > 0:
                    ratio_pct = ((val_max / val_min) - 1.0) * 100.0
                    lines.append(f"- Diferença relativa (Maior vs. Menor): +{ratio_pct:.1f}%")
                # Só soma fluxo; nunca soma estoque ao longo do tempo
                if m_type == "flow":
                    lines.append(f"- Soma do fluxo no recorte: {self._format_value(float(valid.sum()), unit)}")
                elif m_type == "stock" and not has_time_series:
                    lines.append(f"- Estoque consolidado no recorte: {self._format_value(float(valid.sum()), unit)}")

            if len(df) <= 8:
                ordered = df.sort_values(by=col, ascending=False)
                ranking = ", ".join(
                    f"{_row_label(i)}: {self._format_value(float(ordered.loc[i, col]), unit)}"
                    for i in ordered.index
                    if pd.notna(ordered.loc[i, col])
                )
                lines.append(f"- Ranking completo: {ranking}")

            blocks.append("\n".join(lines))

        return "\n".join(blocks)

    def _generate_insight(self, question: str, df: pd.DataFrame) -> str:
        if df.empty:
            return "A consulta não retornou registros para os filtros informados."

        facts = self._build_metric_facts(df)

        system_prompt = (
            "Você é um analista sênior de mercado de crédito do Banco Central do Brasil.\n"
            "Escreva um parágrafo curto e objetivo (2 a 3 frases em português) respondendo à pergunta.\n"
            "REGRAS RÍGIDAS:\n"
            "- Use EXCLUSIVAMENTE os números, unidades e rankings fornecidos em FATOS VERIFICADOS.\n"
            "- Nunca troque valores de uma métrica com outra.\n"
            "- Para taxas (rate), expresse diferenças absolutas em pontos percentuais (p.p.).\n"
            "- Nunca some indicadores de estoque (stock) ao longo de meses distintos.\n"
            "- Não mencione SQL, tabelas, colunas técnicas ou o processo de consulta."
        )

        user_prompt = (
            f"Pergunta do usuário: {question}\n\n"
            f"FATOS VERIFICADOS (calculados direto do banco):\n{facts}"
        )

        self._log(f"[Etapa 2] Gerando análise executiva em texto livre com {self.model}...")
        t0 = time.perf_counter()

        insight = self._call_ollama(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            schema=None,
            temperature=0.2,
            num_predict=260,
        )
        self._log(f"   -> Análise gerada em {time.perf_counter() - t0:.1f}s")
        return insight

    def ask(self, question: str) -> AgentResult:
        sql_response, df, attempts = self._generate_sql_and_execute(question)
        chart_spec = infer_chart_spec(df, needs_chart=sql_response.needs_chart, question=question)
        insight = self._generate_insight(question, df)
        return AgentResult(
            question=question,
            sql_response=sql_response,
            df=df,
            chart_spec=chart_spec,
            insight=insight,
            attempts=attempts,
        )