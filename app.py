import pandas as pd
import streamlit as st

from src.agent import CreditPulseAgent
from src.charts import TIME_COLS, friendly_label, render_chart
from src.database import execute_read_only_query

st.set_page_config(
    page_title="CreditPulse | BI Conversacional BCB",
    page_icon="",
    layout="wide",
)


@st.cache_resource
def get_agent(model_name: str) -> CreditPulseAgent:
    return CreditPulseAgent(model=model_name, verbose=False)


@st.cache_data(ttl=300)
def load_macro_kpis() -> pd.DataFrame:
    """Carrega os dois meses mais recentes unindo crédito (ponderado) e macro sem duplicação."""
    query = """
    WITH credito_mensal AS (
        SELECT
            f.data_ref,
            SUM(f.volume_concessoes_milhoes) AS volume_total,
            SUM(f.saldo_carteira_milhoes) AS saldo_total,
            SUM(f.taxa_juros_anual * f.volume_concessoes_milhoes)
                / NULLIF(SUM(f.volume_concessoes_milhoes), 0) AS juros_ponderado,
            SUM(f.taxa_inadimplencia_90d * f.saldo_carteira_milhoes)
                / NULLIF(SUM(f.saldo_carteira_milhoes), 0) AS inadimplencia_ponderada
        FROM fato_indicadores_credito f
        GROUP BY f.data_ref
    )
    SELECT
        c.data_ref,
        fm.taxa_selic_meta AS selic,
        fm.ipca_mensal AS ipca,
        c.volume_total,
        c.saldo_total,
        c.juros_ponderado,
        c.inadimplencia_ponderada
    FROM credito_mensal c
    JOIN fato_macro_mensal fm ON c.data_ref = fm.data_ref
    ORDER BY c.data_ref DESC
    LIMIT 2
    """
    return execute_read_only_query(query)


def render_kpi_header() -> None:
    st.title("CreditPulse — Inteligência Conversacional de Crédito (BCB)")
    st.caption(
        "API SGS/BCB · Constellation Schema (SQLite Read-Only + AST Sandbox) · Ollama Local"
    )

    kpis = load_macro_kpis()
    if len(kpis) < 2:
        return

    atual, anterior = kpis.iloc[0], kpis.iloc[1]
    ref_mes = str(atual["data_ref"])[:7]

    st.markdown(f"**Panorama Consolidado ({ref_mes} vs. mês anterior)**")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Selic Meta",
        f"{atual['selic']:.2f}% a.a.",
        f"{atual['selic'] - anterior['selic']:+.2f} p.p.",
    )
    c2.metric(
        "IPCA Mensal",
        f"{atual['ipca']:.2f}%",
        f"{atual['ipca'] - anterior['ipca']:+.2f} p.p.",
    )
    c3.metric(
        "Juros Pond. (Concessões)",
        f"{atual['juros_ponderado']:.2f}% a.a.",
        f"{atual['juros_ponderado'] - anterior['juros_ponderado']:+.2f} p.p.",
        delta_color="inverse",
    )
    c4.metric(
        "Inadimplência Pond. (Saldo)",
        f"{atual['inadimplencia_ponderada']:.2f}%",
        f"{atual['inadimplencia_ponderada'] - anterior['inadimplencia_ponderada']:+.2f} p.p.",
        delta_color="inverse",
    )
    var_conc = ((atual["volume_total"] / anterior["volume_total"]) - 1) * 100
    c5.metric(
        "Concessões (Fluxo)",
        f"R$ {atual['volume_total'] / 1000:.1f} bi",
        f"{var_conc:+.1f}%",
    )
    var_saldo = ((atual["saldo_total"] / anterior["saldo_total"]) - 1) * 100
    c6.metric(
        "Carteira (Estoque)",
        f"R$ {atual['saldo_total'] / 1_000_000:.2f} tri",
        f"{var_saldo:+.1f}%",
    )
    st.divider()


def render_assistant_message(msg: dict, idx: int) -> None:
    st.markdown(msg["insight"])

    df: pd.DataFrame = msg["df"]
    spec = msg["chart_spec"]

    if not df.empty and spec.chart_type != "none":
        # Se a query trouxe múltiplas métricas numéricas, permite alternar o eixo Y na hora
        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in TIME_COLS and not c.startswith("id_")
        ]
        active_spec = spec.model_copy()

        if len(numeric_cols) > 1 and spec.chart_type in ("bar", "line"):
            default_idx = numeric_cols.index(spec.y_col) if spec.y_col in numeric_cols else 0
            selected_y = st.selectbox(
                "Métrica exibida no gráfico:",
                options=numeric_cols,
                index=default_idx,
                format_func=friendly_label,
                key=f"metric_select_{idx}",
            )
            active_spec.y_col = selected_y
            active_spec.title = (
                f"{friendly_label(selected_y)} por {friendly_label(active_spec.x_col)}"
            )

        fig = render_chart(df, active_spec)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")

    with st.expander(f"🔍 Auditar SQL e Dados ({len(df)} linhas · {msg['attempts']} tentativa(s))"):
        st.code(msg["sql"], language="sql")
        st.dataframe(df, use_container_width=True, hide_index=True)
        if not df.empty:
            st.download_button(
                label="⬇️ Baixar CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"creditpulse_consulta_{idx}.csv",
                mime="text/csv",
                key=f"dl_{idx}",
            )


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurações")
    model_name = st.selectbox(
        "Modelo Ollama (Local)",
        options=["qwen2.5:3b", "qwen2.5-coder:3b", "qwen3:4b"],
        index=0,
    )

    st.divider()
    st.subheader("💡 Perguntas Sugeridas")
    sugestoes = [
        "Compare a taxa de juros anual e a inadimplência das 4 modalidades no último mês disponível",
        "Mostre a evolução mensal da taxa Selic em 2025",
        "Qual foi o volume total de concessões por ano entre 2020 e 2025?",
        "Compare a evolução da inadimplência do Crédito Pessoal Não Consignado PF e do Consignado Total PF desde 2023",
        "Qual modalidade teve o maior volume acumulado de concessões em 2025?",
    ]
    clicked_prompt = None
    for i, s in enumerate(sugestoes):
        if st.button(s, key=f"sug_{i}", use_container_width=True):
            clicked_prompt = s

    st.divider()
    if st.button("🗑️ Limpar histórico do chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Corpo Principal ---
render_kpi_header()

if "messages" not in st.session_state:
    st.session_state.messages = []

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_assistant_message(message, idx)

user_input = st.chat_input("Pergunte sobre juros, inadimplência, concessões, Selic ou IPCA...")
prompt = clicked_prompt or user_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        agent = get_agent(model_name)
        try:
            with st.status("Consultando base do BCB e gerando análise...", expanded=True) as status:
                st.write("🧠 Traduzindo pergunta para SQLite (read-only)...")
                result = agent.ask(prompt)
                status.update(
                    label=f"Concluído em {result.attempts} tentativa(s) · {len(result.df)} registro(s)",
                    state="complete",
                    expanded=False,
                )

            assistant_msg = {
                "role": "assistant",
                "insight": result.insight,
                "df": result.df,
                "chart_spec": result.chart_spec,
                "sql": result.sql_response.sql_query,
                "attempts": result.attempts,
            }
            st.session_state.messages.append(assistant_msg)
            render_assistant_message(assistant_msg, len(st.session_state.messages) - 1)

        except Exception as exc:
            st.error(f"Não foi possível processar a pergunta: {exc}")