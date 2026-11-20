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
    """Pacote completo devolvido pelo agente para o terminal ou Streamlit."""
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
        self.compact_schema = self._build_compact_schema(semantic_model_path)
        self.max_retries = max_retries
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _build_compact_schema(self, semantic_model_path: Path) -> str:
        """Lê o semantic_layer.yaml e gera um contexto compacto focado em SQL."""
        data = yaml.safe_load(semantic_model_path.read_text(encoding="utf-8"))
        lines = [
            "SCHEMA SQLITE (periodo: 2018-01-01 a 2026-09-01; 2026 parcial ate mes 9):",
            "ALIASES PADRAO: fato_indicadores_credito f | dim_modalidades m | dim_tempo t",
        ]

        for table_name, table_info in data["tables"].items():
            cols = []
            for col_name, col_info in table_info["columns"].items():
                allowed = col_info.get("allowed_values")
                if allowed:
                    cols.append(f"{col_name}({col_info['type']}, valores={allowed})")
                else:
                    cols.append(f"{col_name}({col_info['type']})")
            lines.append(f"- {table_name}: " + ", ".join(cols))

        if data.get("relationships"):
            joins = [f"{rel['from']} = {rel['to']}" for rel in data["relationships"]]
            lines.append("JOINS: " + "; ".join(joins))

        lines.append("METRICAS OFICIAIS:")
        for m_name, m_info in data["metrics"].items():
            lines.append(f"- {m_name}: {m_info['sql']}")

        lines.append("REGRAS SQL:")
        for rule in data["business_rules"]:
            if rule.lower().startswith("ao explicar"):
                continue
            lines.append(f"- {rule}")

        return "\n".join(lines)

    def _call_ollama(
        self,
        messages: list[dict],
        temperature: float,
        schema: dict | None = None,
        num_predict: int = 512,
    ) -> str:
        """Centraliza chamadas ao Ollama (suporta JSON Schema na Etapa 1 e texto livre na Etapa 2)."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": 2048,
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
            "Especialista em SQLite e mercado de credito do Banco Central (BCB). "
            "Escreva a query SELECT ou WITH mais simples e direta possivel para responder a pergunta.\n"
            "- Use SEMPRE os aliases f (fato_indicadores_credito), m (dim_modalidades) e t (dim_tempo).\n"
            "- So faca JOIN ou referencie uma tabela (m ou t) se realmente precisar de colunas dela.\n"
            "- Evite CTEs (WITH) e GROUP BY desnecessarios quando uma subquery simples ou SELECT direto resolver.\n"
            "- Retorne apenas as colunas necessarias para a pergunta; nao invente calculos extras.\n\n"
            f"{self.compact_schema}"
        )

    def _generate_sql_and_execute(self, question: str) -> tuple[SQLResponse, pd.DataFrame, int]:
        """Etapa 1: Gera SQL estruturado, executa no banco e tenta autocorrigir erros."""
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

            try:
                sql_response = SQLResponse.model_validate_json(raw_json)
                df = execute_read_only_query(sql_response.sql_query)
                self._log(f"   -> Query executada com sucesso ({len(df)} linhas retornadas)")
                return sql_response, df, attempt

            except UnsafeQueryError:
                raise

            except (ValueError, QueryExecutionError) as exc:
                last_error = exc
                self._log(f"   -> Falha na tentativa {attempt}: {exc}")
                messages.append({"role": "assistant", "content": raw_json})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Erro na query:\n{exc}\n"
                            "Corrija usando exatamente os aliases f, m, t e as colunas do schema."
                        ),
                    }
                )

        raise RuntimeError(
            f"Não foi possível gerar uma query válida após {self.max_retries + 1} tentativas. "
            f"Último erro: {last_error}"
        )

    def _build_metric_facts(self, df: pd.DataFrame) -> str:
        """Agrupa os fatos numéricos por métrica para eliminar confusão entre colunas."""
        metric_cols = [
            c for c in df.select_dtypes(include="number").columns
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
            items = [f"{friendly_label(c)}: {row[c]}" for c in df.columns]
            blocks.append("Resultado único: " + " ; ".join(items))
            return "\n".join(blocks)

        for col in metric_cols:
            valid = df[col].dropna()
            if valid.empty:
                continue

            idx_max, idx_min = valid.idxmax(), valid.idxmin()
            val_max, val_min = df.loc[idx_max, col], df.loc[idx_min, col]
            diff = val_max - val_min
            is_rate = any(k in col.lower() for k in ("taxa", "inadimp", "juros", "selic", "ipca"))
            unit = "%" if is_rate else ""
            diff_unit = " p.p." if is_rate else ""

            lines = [
                f"\nMétrica: {friendly_label(col)}",
                f"- MAIOR valor: {_row_label(idx_max)} = {val_max:.2f}{unit}",
                f"- MENOR valor: {_row_label(idx_min)} = {val_min:.2f}{unit}",
                f"- Amplitude (Maior - Menor): {diff:.2f}{diff_unit}",
            ]

            # Se a tabela for pequena, lista o ranking completo daquela métrica isolada
            if len(df) <= 8:
                ordered = df.sort_values(by=col, ascending=False)
                ranking = ", ".join(
                    f"{_row_label(i)}: {ordered.loc[i, col]:.2f}{unit}"
                    for i in ordered.index
                )
                lines.append(f"- Ranking completo: {ranking}")

            blocks.append("\n".join(lines))

        return "\n".join(blocks)

    def _generate_insight(self, question: str, df: pd.DataFrame) -> str:
        """Etapa 2: Redige análise executiva em texto livre ancorada nos fatos calculados."""
        if df.empty:
            return "A consulta não retornou registros para os filtros informados."

        facts = self._build_metric_facts(df)

        system_prompt = (
            "Você é um analista sênior de mercado de crédito do Banco Central do Brasil.\n"
            "Escreva um parágrafo curto e objetivo (2 a 3 frases em português) respondendo à pergunta.\n"
            "REGRAS RÍGIDAS:\n"
            "- Use EXCLUSIVAMENTE os números e rankings fornecidos em FATOS VERIFICADOS.\n"
            "- Nunca troque valores de uma métrica com outra.\n"
            "- Use pontos percentuais (p.p.) para diferenças absolutas de taxas.\n"
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
        """Ponto de entrada único: Pergunta -> SQL -> DataFrame -> ChartSpec (Python) + Insight (LLM)."""
        sql_response, df, attempts = self._generate_sql_and_execute(question)
        chart_spec = infer_chart_spec(df, needs_chart=sql_response.needs_chart)
        insight = self._generate_insight(question, df)
        return AgentResult(
            question=question,
            sql_response=sql_response,
            df=df,
            chart_spec=chart_spec,
            insight=insight,
            attempts=attempts,
        )