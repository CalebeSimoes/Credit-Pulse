# CreditPulse

Aplicação local de BI conversacional para consulta e análise de indicadores de crédito e macroeconomia do Brasil, utilizando dados públicos da API SGS do Banco Central do Brasil (BCB).

O sistema traduz perguntas em linguagem natural para SQLite (somente leitura) com apoio de uma camada semântica em YAML, executa a consulta localmente, renderiza gráficos em Plotly de forma determinística e gera um resumo textual via Ollama rodando em CPU.

## Como funciona

O fluxo é dividido em três partes para reduzir erros comuns de modelos pequenos (3B) em tarefas numéricas e de visualização:

1. **Text-to-SQL (`src/agent.py`, `src/schemas.py`, `src/database.py`)**: o arquivo `config/semantic_layer.yaml` é compactado em tempo de execução para caber confortavelmente no contexto da CPU. O modelo `qwen2.5:3b` gera um JSON validado por Pydantic (`SQLResponse`) contendo apenas queries `SELECT` ou `WITH`. Se o SQLite retornar erro de sintaxe ou coluna, o agente reenvia o erro ao modelo em um loop de autocorreção (até 2 tentativas extras).
2. **Gráficos e cálculos determinísticos (`src/charts.py`, `src/agent.py`)**: o Python analisa o DataFrame retornado e decide o gráfico (`line`, `bar`, `scatter` ou `none`) sem depender da LLM. Antes da etapa de texto, o próprio código calcula mínimo, máximo, amplitude e ranking isolado por métrica.
3. **Resumo textual (`src/agent.py`)**: o modelo recebe apenas os fatos já calculados em Python (sem tabela bruta e sem trava de JSON Schema) e redige uma resposta objetiva em português.

Na interface (`app.py`), cada resposta exibe o texto, o gráfico interativo (com seletor de métrica quando a query traz mais de um indicador numérico) e um painel expansível com o SQL executado, a tabela completa e exportação em CSV.

## Decisões técnicas

- **Modelo único em CPU (`qwen2.5:3b`)**: durante os testes em CPU local, o `qwen3:4b` apresentou latência alta e vazamento de raciocínio interno dentro do JSON Schema; já o `qwen2.5-coder:3b` gerou SQL consistente, mas trocou valores e métricas na redação analítica. Manter dois modelos locais causaria disputa de memória RAM e recarregamento constante no Ollama. O `qwen2.5:3b` equilibrou geração de SQL, tempo de resposta e texto em português.
- **Gráfico fora da LLM**: gerar `ChartSpec` via LLM adicionava latência e falhas de validação de colunas. A inferência em Python é instantânea e previsível.
- **Fatos pré-calculados**: enviar CSV bruto para um modelo 3B causou inversões entre maior e menor valor. Calcular os extremos por métrica no Pandas antes da chamada de texto eliminou esse tipo de erro.

## Base de dados e modelagem

Os dados são extraídos da API SGS do Banco Central (`config/bcb_series.yaml`) a partir de `2018-01-01` (420 registros mensais na fato, até `2026-09-01`) e persistidos em um Star Schema no SQLite:

- `fato_indicadores_credito`: grão mensal por modalidade (`data_ref`, `id_modalidade`), armazenando `taxa_juros_anual`, `taxa_inadimplencia_90d`, `volume_concessoes_milhoes`, `taxa_selic_meta` e `ipca_mensal`.
- `dim_modalidades`: `id_modalidade`, `nome_modalidade`, `segmento` (`PF`/`PJ`) e `tipo_recurso` (`Livre`).
- `dim_tempo`: `data_ref`, `ano`, `mes`, `trimestre` e `semestre`.

### Séries SGS monitoradas

| Modalidade / Indicador | Segmento | Juros (% a.a.) | Inadimplência >90d (%) | Concessões (R$ mi) |
| :--- | :---: | :---: | :---: | :---: |
| Crédito Pessoal Não Consignado PF | PF | 20742 | 21112 | 20662 |
| Crédito Pessoal Consignado Total PF | PF | 20747 | 21117 | 20667 |
| Aquisição de Veículos PF | PF | 20749 | 21119 | 20669 |
| Capital de Giro Total PJ | PJ | 20723 | 21093 | 20643 |
| Taxa Selic Meta (% a.a.) | Macro | 432 | - | - |
| IPCA Mensal (%) | Macro | 433 | - | - |

## Estrutura do repositório

```text
.
├── app.py                      # Interface Streamlit (KPIs do último mês, chat e auditoria SQL)
├── config/
│   ├── bcb_series.yaml         # Códigos das séries na API SGS do BCB
│   └── semantic_layer.yaml     # Schema, métricas oficiais e regras de negócio para o agente
├── src/
│   ├── ingest_bcb.py           # Extração da API do BCB e carga no SQLite
│   ├── database.py             # Execução de consultas com validação read-only
│   ├── schemas.py              # Contratos Pydantic (SQLResponse e ChartSpec)
│   ├── charts.py               # Inferência de gráfico e renderização Plotly
│   └── agent.py                # Orquestrador Text-to-SQL + geração de resumo
├── requirements.txt
└── .gitignore