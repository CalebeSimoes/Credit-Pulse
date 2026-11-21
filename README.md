# CreditPulse

Aplicação local de BI conversacional para consulta e análise de indicadores de crédito e macroeconomia do Brasil, utilizando dados públicos da API SGS do Banco Central do Brasil (BCB).

O sistema traduz perguntas em linguagem natural para SQLite com apoio de uma camada semântica em YAML, executa a consulta em uma sandbox somente leitura, renderiza gráficos em Plotly de forma determinística e gera um resumo textual via Ollama rodando em CPU.

## Como funciona

O fluxo é dividido em três etapas para reduzir erros comuns de modelos pequenos (3B) em tarefas numéricas, de agregação e de visualização:

1. **Text-to-SQL e Sandbox SQL (`src/agent.py`, `src/schemas.py`, `src/database.py`)**: o arquivo `config/semantic_layer.yaml` é compactado em tempo de execução junto a exemplos curtos (few-shot) para manter baixa latência em CPU. O modelo `qwen2.5:3b` retorna um JSON validado por Pydantic (`SQLResponse`). Antes de executar, a consulta passa por uma sandbox com validação AST via `sqlglot`, lista explícita de tabelas permitidas, conexão SQLite em modo somente leitura (`?mode=ro`), `PRAGMA query_only = ON`, timeout por `set_progress_handler` (3s) e limite de linhas (`fetchmany`, padrão 1.000). Em caso de erro de sintaxe ou coluna, o agente reenvia o diagnóstico ao modelo em um loop de autocorreção (até 2 tentativas extras).
2. **Gráficos e fatos tipados (`src/charts.py`, `src/agent.py`)**: o Python analisa o DataFrame e escolhe o gráfico (`line`, `bar`, `scatter` ou `none`) usando metadados do YAML, priorizando a métrica mencionada na pergunta e tratando partes discretas de data (`mes`, `trimestre`, `semestre`) como barras para não conectar anos distintos em linhas. Antes da etapa de texto, o código calcula mínimo, máximo, diferenças e ranking respeitando o tipo de cada métrica (`rate`, `flow` ou `stock`), sem somar estoques ao longo de meses diferentes.
3. **Resumo textual (`src/agent.py`)**: o modelo recebe apenas os fatos já calculados e formatados em Python (sem tabela bruta e sem trava de JSON Schema) e redige uma resposta objetiva em português.

Na interface (`app.py`), o topo exibe 6 KPIs consolidados do último mês disponível (Selic Meta, IPCA Mensal, Juros Ponderados por Concessões, Inadimplência Ponderada pelo Saldo, Concessões e Estoque da Carteira). Cada resposta do chat mostra o resumo, o gráfico interativo (com seletor de métrica quando há múltiplos indicadores numéricos) e um painel expansível com o SQL executado, a tabela completa e exportação em CSV.

## Decisões técnicas

- **Modelo único em CPU (`qwen2.5:3b`)**: nos testes locais, o `qwen3:4b` apresentou latência alta e vazamento de raciocínio interno dentro do JSON Schema; o `qwen2.5-coder:3b` gerou SQL consistente, mas trocou valores e métricas na redação. Manter dois modelos locais causaria disputa de RAM e recarregamento no Ollama. O `qwen2.5:3b` equilibrou geração de SQL, tempo de resposta e português.
- **Gráfico e aritmética fora da LLM**: gerar `ChartSpec` via LLM adicionava latência e falhas de validação, e enviar CSV bruto ao modelo 3B causava inversões entre extremos e confusão de unidades. Inferir o gráfico e calcular os extremos em Python usando os metadados da camada semântica resolveu ambos os problemas.
- **Separação entre fluxo e estoque**: a taxa média ponderada de juros usa o volume de concessões (fluxo mensal de novas operações) como denominador; já a inadimplência acima de 90 dias incide sobre a carteira ativa e passa a ser ponderada pelo saldo da carteira (estoque).
- **Separação do grão macroeconômico**: Selic e IPCA foram movidos para uma tabela fato própria (`fato_macro_mensal`), eliminando a repetição por modalidade e o uso artificial de `AVG()` para desduplicação.
- **Ingestão idempotente via staging**: em vez de recriar o banco com `DROP TABLE`, o pipeline usa retries HTTP com backoff exponencial, checagens de integridade em memória, tabelas temporárias de staging e `UPSERT` atômico em transação única. Na execução padrão, rebusca uma janela móvel dos últimos 3 meses para absorver revisões retroativas do BCB.

## Base de dados e modelagem

Os dados são extraídos da API SGS do Banco Central (`config/bcb_series.yaml`) desde `2018-01-01` (420 registros na fato de crédito e 107 meses na fato macroeconômica) e armazenados em um Constellation Schema no SQLite (`data/creditpulse_bcb.db`):

- `fato_indicadores_credito`: grão mensal por modalidade com chave primária composta `(data_ref, id_modalidade)`, armazenando `taxa_juros_anual`, `taxa_inadimplencia_90d`, `volume_concessoes_milhoes`, `saldo_carteira_milhoes` e `updated_at`.
- `fato_macro_mensal`: grão mensal único (`data_ref` como chave primária), armazenando `taxa_selic_meta`, `ipca_mensal` e `updated_at`.
- `dim_modalidades`: `id_modalidade`, `nome_modalidade`, `segmento` (`PF`/`PJ`) e `tipo_recurso` (`Livre`).
- `dim_tempo`: `data_ref`, `ano`, `mes`, `trimestre` e `semestre`, cobrindo a união das datas de crédito e macroeconomia.

### Séries SGS monitoradas

| Modalidade / Indicador | Segmento | Juros (% a.a.) | Inadimplência >90d (%) | Concessões (R$ mi) | Saldo da Carteira (R$ mi) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Crédito Pessoal Não Consignado PF | PF | 20742 | 21112 | 20666 | 20574 |
| Crédito Pessoal Consignado Total PF | PF | 20747 | 21117 | 20671 | 20579 |
| Aquisição de Veículos PF | PF | 20749 | 21119 | 20673 | 20581 |
| Capital de Giro Total PJ | PJ | 20723 | 21093 | 20642 | 20550 |
| Taxa Selic Meta (% a.a.) | Macro | 432 | - | - | - |
| IPCA Mensal (%) | Macro | 433 | - | - | - |

## Estrutura do repositório

```text
.
├── app.py                      # Interface Streamlit (6 KPIs consolidados, chat e auditoria SQL)
├── config/
│   ├── bcb_series.yaml         # Códigos das séries na API SGS do BCB
│   └── semantic_layer.yaml     # Schema, metadados de métricas (tipo/unidade) e regras de negócio
├── src/
│   ├── ingest_bcb.py           # Pipeline incremental BCB -> SQLite (retries, validação, staging e UPSERT)
│   ├── database.py             # Sandbox SQL read-only (AST sqlglot, allowlist, mode=ro, timeout e limite)
│   ├── schemas.py              # Contratos Pydantic (SQLResponse e ChartSpec)
│   ├── charts.py               # Inferência determinística de gráficos e renderização Plotly
│   └── agent.py                # Orquestrador Text-to-SQL + fatos tipados + resumo textual
├── requirements.txt
└── .gitignore