import unicodedata
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml

from src.schemas import ChartSpec

BASE_DIR = Path(__file__).resolve().parent.parent
SEMANTIC_MODEL_PATH = BASE_DIR / "config" / "semantic_layer.yaml"

CONTINUOUS_TIME_COLS = ("data_ref", "ano")
DISCRETE_TIME_COLS = {"mes", "trimestre", "semestre"}
TIME_COLS = set(CONTINUOUS_TIME_COLS) | DISCRETE_TIME_COLS


def _load_labels_from_semantic_layer(path: Path = SEMANTIC_MODEL_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    labels: dict[str, str] = {}
    for table_info in data.get("tables", {}).values():
        for col_name, col_info in table_info.get("columns", {}).items():
            if col_info.get("label"):
                labels[col_name] = col_info["label"]
    for metric_name, metric_info in data.get("metrics", {}).items():
        if metric_info.get("label"):
            labels[metric_name] = metric_info["label"]
    return labels


COLUMN_LABELS = _load_labels_from_semantic_layer()


def _normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def friendly_label(col_name: str) -> str:
    """Converte nomes técnicos de colunas em rótulos definidos na camada semântica."""
    if col_name in COLUMN_LABELS:
        return COLUMN_LABELS[col_name]
    return col_name.replace("_", " ").strip().title()


def _pick_primary_metric(numeric_cols: list[str], question: str | None) -> str:
    """Escolhe o eixo Y priorizando a métrica que aparece primeiro na pergunta."""
    if not question or len(numeric_cols) == 1:
        return numeric_cols[0]

    q_norm = _normalize_text(question)
    keyword_map = {
        "inadimplencia": ("inadimp",),
        "juros": ("juro", "taxa_juros"),
        "concessoes": ("concess", "volume"),
        "saldo": ("saldo", "carteira", "estoque"),
        "selic": ("selic",),
        "ipca": ("ipca", "inflac"),
    }

    best_col = numeric_cols[0]
    best_pos = len(q_norm) + 1

    for col in numeric_cols:
        col_norm = _normalize_text(col)
        label_norm = _normalize_text(friendly_label(col))
        tokens = set(col_norm.split("_")) | set(label_norm.split())
        for concept, stems in keyword_map.items():
            if any(s in col_norm for s in stems):
                tokens.add(concept)
                tokens.update(stems)

        for token in tokens:
            if len(token) < 4:
                continue
            pos = q_norm.find(token)
            if 0 <= pos < best_pos:
                best_pos = pos
                best_col = col

    return best_col


def infer_chart_spec(
    df: pd.DataFrame,
    needs_chart: bool = True,
    question: str | None = None,
) -> ChartSpec:
    """Escolhe o gráfico Plotly ideal usando granularidade temporal explícita."""
    if df.empty or not needs_chart or len(df) < 2:
        return ChartSpec(chart_type="none")

    numeric_cols = [
        c
        for c in df.select_dtypes(include="number").columns
        if c not in TIME_COLS and not c.startswith("id_")
    ]
    if not numeric_cols:
        return ChartSpec(chart_type="none")

    y_col = _pick_primary_metric(numeric_cols, question)

    cat_cols = [
        c
        for c in df.columns
        if c not in CONTINUOUS_TIME_COLS
        and c not in numeric_cols
        and not c.startswith("id_")
    ]

    # Caso 1: Eixo temporal contínuo explícito (data_ref ou ano) -> Linha
    for t_col in CONTINUOUS_TIME_COLS:
        if t_col in df.columns and df[t_col].nunique() >= 2:
            # Evita ligar pontos de mesma data/ano sem série separada por cor
            non_time_cats = [c for c in cat_cols if c not in DISCRETE_TIME_COLS]
            color_col = next((c for c in non_time_cats if df[c].nunique() >= 2), None)
            title = f"Evolução de {friendly_label(y_col)}"
            if color_col:
                title += f" por {friendly_label(color_col)}"
            return ChartSpec(
                chart_type="line",
                x_col=t_col,
                y_col=y_col,
                color_col=color_col,
                title=title,
            )

    # Caso 2: Categorias ou partes discretas do tempo (ex.: mes/trimestre isolado) -> Barras
    if cat_cols:
        x_col = cat_cols[0]
        color_col = next((c for c in cat_cols[1:] if df[c].nunique() >= 2), None)
        return ChartSpec(
            chart_type="bar",
            x_col=x_col,
            y_col=y_col,
            color_col=color_col,
            title=f"{friendly_label(y_col)} por {friendly_label(x_col)}",
        )

    # Caso 3: Apenas métricas numéricas -> Dispersão
    if len(numeric_cols) >= 2:
        x_metric = next((c for c in numeric_cols if c != y_col), numeric_cols[0])
        return ChartSpec(
            chart_type="scatter",
            x_col=x_metric,
            y_col=y_col,
            title=f"{friendly_label(y_col)} vs. {friendly_label(x_metric)}",
        )

    return ChartSpec(chart_type="none")


def render_chart(df: pd.DataFrame, spec: ChartSpec) -> go.Figure | None:
    """Renderiza um gráfico Plotly padronizado a partir do DataFrame e do ChartSpec."""
    if df.empty or spec.chart_type == "none" or not spec.x_col or not spec.y_col:
        return None

    if spec.x_col not in df.columns or spec.y_col not in df.columns:
        return None

    plot_df = df.copy()
    color_col = spec.color_col if spec.color_col in plot_df.columns else None

    if spec.chart_type == "line":
        plot_df = plot_df.sort_values(by=spec.x_col)
    elif spec.chart_type == "bar" and color_col is None:
        if spec.x_col in DISCRETE_TIME_COLS:
            plot_df = plot_df.sort_values(by=spec.x_col)
        else:
            plot_df = plot_df.sort_values(by=spec.y_col, ascending=False)

    labels = {col: friendly_label(col) for col in plot_df.columns}
    title = spec.title or f"{friendly_label(spec.y_col)} por {friendly_label(spec.x_col)}"

    common_kwargs = {
        "data_frame": plot_df,
        "x": spec.x_col,
        "y": spec.y_col,
        "color": color_col,
        "title": title,
        "labels": labels,
        "template": "plotly_dark",
    }

    if spec.chart_type == "line":
        fig = px.line(**common_kwargs, markers=True)
    elif spec.chart_type == "bar":
        fig = px.bar(**common_kwargs, text_auto=".2f")
        fig.update_traces(textposition="outside")
    elif spec.chart_type == "scatter":
        fig = px.scatter(**common_kwargs)
    else:
        return None

    fig.update_layout(
        margin=dict(l=20, r=20, t=55, b=20),
        legend_title_text=friendly_label(color_col) if color_col else "",
        hovermode="x unified" if spec.chart_type == "line" else "closest",
    )
    return fig