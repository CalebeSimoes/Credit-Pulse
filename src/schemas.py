from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class SQLResponse(BaseModel):
    """Contrato enxuto para a resposta do agente Text-to-SQL."""

    sql_query: str = Field(
        description=(
            "Apenas a query SQLite valida comecando com SELECT ou WITH. "
            "Sem comentarios, markdown ou texto extra."
        )
    )
    needs_chart: bool = Field(
        default=True,
        description="True se fizer sentido plotar grafico; False para um unico numero."
    )

    @field_validator("sql_query")
    @classmethod
    def validate_select_only(cls, v: str) -> str:
        """Guardrail inicial: garante que a query comece com SELECT ou WITH."""
        cleaned = v.strip().upper()
        if not (cleaned.startswith("SELECT") or cleaned.startswith("WITH")):
            raise ValueError("Apenas queries de leitura (SELECT / WITH) são permitidas.")
        return v.strip()


class ChartSpec(BaseModel):
    """Contrato determinístico de visualização Plotly (gerado em Python)."""

    chart_type: Literal["line", "bar", "scatter", "none"] = "none"
    x_col: Optional[str] = None
    y_col: Optional[str] = None
    color_col: Optional[str] = None
    title: Optional[str] = None