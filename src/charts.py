import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.schemas import ChartSpec

TIME_COLS = {"data_ref", "ano", "mes", "trimestre", "semestre"}

COLUMN_LABELS = {
    "data_ref": "Data de Referência",
    "ano": "Ano",
    "mes": "Mês",
    "trimestre": "Trimestre",
    "semestre": "Semestre",
    "nome_modalidade": "Modalidade de Crédito",
    "segmento": "Segmento",
    "tipo_recurso": "Tipo de Recurso",
    "taxa_juros_anual": "Taxa de Juros (% a.a.)",
    "taxa_inadimplencia_90d": "Inadimplência >90d (%)",
    "volume_concessoes_milhoes": "Volume de Concessões (R$ milhões)",
    "taxa_selic_meta": "Selic Meta (% a.a.)",
    "ipca_mensal": "IPCA Mensal (%)",
}


def friendly_label(col_name: str) -> str:
    """Converte nomes técnicos de colunas em rótulos legíveis."""
    if col_name in COLUMN_LABELS:
        return COLUMN_LABELS[col_name]
    return col_name.replace("_", " ").strip().title()


def infer_chart_spec(df: pd.DataFrame, needs_chart: bool = True) -> ChartSpec:
    """Escolhe o gráfico Plotly ideal de forma determinística em milissegundos."""
    if df.empty or not needs_chart or len(df) < 2:
        return ChartSpec(chart_type="none")

    time_cols = [c for c in df.columns if c in TIME_COLS]
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in TIME_COLS and not c.startswith("id_")
    ]
    cat_cols = [
        c for c in df.columns
        if c not in time_cols and c not in numeric_cols and not c.startswith("id_")
    ]

    if not numeric_cols:
        return ChartSpec(chart_type="none")

    y_col = numeric_cols[0]

    # Caso 1: Série temporal -> Gráfico de linhas
    for t_col in ("data_ref", "ano", "trimestre", "semestre", "mes"):
        if t_col in time_cols and df[t_col].nunique() >= 2:
            color_col = next((c for c in cat_cols if df[c].nunique() >= 2), None)
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

    # Caso 2: Comparação entre categorias (modalidades, segmentos) -> Barras
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
        return ChartSpec(
            chart_type="scatter",
            x_col=numeric_cols[0],
            y_col=numeric_cols[1],
            title=f"{friendly_label(numeric_cols[1])} vs. {friendly_label(numeric_cols[0])}",
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