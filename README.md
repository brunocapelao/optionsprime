# Options Prime — Especificação do Sistema

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Propósito**: rotina diária que lê o **regime** do BTC (4H, *bar-close*) e monta **estruturas de opções de 7–10 dias** com risco limitado, usando bandas preditivas calibradas **fora-da-amostra (OOS)**. O "motor" de retorno combina **direção** + relação **IV vs RV** nesses 7–10 dias.

---

## 🚀 Quick Start

```bash
# Clone o repositório
git clone git@github.com:brunocapelao/optionsprime.git
cd optionsprime

# Configure o ambiente
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt

# Configure Git hooks (importante!)
pip install nbstripout pre-commit
nbstripout --install
pre-commit install

# Execute os notebooks na ordem
jupyter lab
```

📖 **[Leia o guia completo de contribuição](CONTRIBUTING.md)** para melhores práticas de Git e ML.

---ions Prime — Especificação do Sistema

> **Propósito**: rotina diária que lê o **regime** do BTC (4H, *bar-close*) e monta **estruturas de opções de 7–10 dias** com risco limitado, usando bandas preditivas calibradas **fora-da-amostra (OOS)**. O “motor” de retorno combina **direção** + relação **IV vs RV** nesses 7–10 dias.

---

## 1) Objetivos, Escopo e Princípios

**Objetivos**

* Gerar **um** trade candidato por dia útil, para vencimento **D+7 a D+10**.
* Em **TREND ON**: propor **call debit spread** (bias direcional).
* Em **RANGE ON** (futuro): propor **iron condor** (venda de prêmio), **apenas** quando houver cobertura OOS estável.
* Estimar **POP** (prob. de sucesso) e **EV** (valor esperado) sob a distribuição preditiva **P** (bandas).
* Garantir **governança estatística**: anti-leakage, OOS, *embargo*, reprodutibilidade.

**Escopo**

* Ativo: **BTCUSD (spot ou perp)**.
* Timeframe base: **4H**; horizonte alvo: **7–10 dias** (≈ **T=42 a 60** barras; MVP usa **T=42** e **T=84**).
* Sinais em **“bar close only”**; execução t+1 barra no subjacente para medição.

**Princípios**

* **Regime-aware** (ADX com histerese): operar apenas quando o regime estável habilita o *playbook*.
* **Simplicidade robusta**: indicadores *built-in*, regras objetivas, risco explícito (ATR).
* **Validação honesta**: calibração no **Calib**, medição no **Test** com **embargo**.
* **Risco limitado**: **nunca naked**; spreads com stops/gestão por ATR.

**Não-objetivos**

* Market making, scalping, arbitragem, execução automática em corretoras, *naked options*.

---

## 2) Visão Geral da Arquitetura

```
[00_baixar_dados]  ---->  [01_data_features]  ---->  [02_quantiles_bands]
       |                         |                          |
       |                         |                          +--> preds_T=42/84.parquet (+meta)
       |                         v
       |                   features_4H.parquet
       |
       +--> (chains de opções D+7..D+10)  <-- feed externo (CSV/JSON)

[02C_onpolicy_recalib_OOS] ---> preds_T=42/84_onpolicy_oos.parquet (+meta, qc_oos.json)
                                   |
                                   v
                            [03_options_mapper]
                                   |
                                   v
                             trade_card.json
                                   |
                                   v
                             [04_reporting]
```

**Estado atual (MVP)**

* **Aprovado**: TREND · **T=42 (7d)** — cobertura OOS ok.
* **Pendente**: RANGE (7d) e T=84 (14d) — aguardam ajustes de *pooling/buckets* (ver §9).

---

## Estrutura de Diretórios

O projeto é organizado na seguinte estrutura:

```
project/
├── config/
│   └── base.yaml           # Arquivo de configuração central para caminhos e parâmetros.
├── data/
│   ├── processed/          # Dados processados e resultados intermediários.
│   │   ├── features/       # Features de engenharia de dados.
│   │   └── preds/          # Previsões geradas pelos modelos.
│   ├── raw/
│   │   └── BTCUSD_CCCAGG_1h.csv # Dados brutos de preço (BTC/USD, 1 hora).
│   └── reports/            # (Opcional) Relatórios de análise.
├── notebooks/              # Notebooks Jupyter com o fluxo de trabalho da análise.
└── requirements.txt        # Dependências Python do projeto.
```

---

## 3) Módulos & Contratos (I/O)

### 3.1 00_baixar_dados (ingest)

* **Entrada**: API CryptoCompare/CCData (CCCAGG 1h).
* **Saída**: `BTCUSD_CCCAGG_1h.csv` incremental (campos: `open_time(ms),open,high,low,close,volume,cost`).
* **Regras**: *append-safe* (flush por página), *retry*, *resume* do último `open_time`.
* **Erros**: timeouts → *backoff*; falha parcial mantém dados já gravados.

### 3.2 01_data_features

* **Entrada**: `BTCUSD_CCCAGG_1h.csv`
* **Processo**:

  * *Resample* para 4H; validação de lacunas/gaps; UTC; *dedupe*.
  * Indicadores: **ADX14** (para histerese), **ATR14**, **Donchian20**, **EMAs20/50/200**, sinais (`BREAKOUT_*, CROSS_*`).
  * `ATR_PCT` (ATR relativo ao preço) + *winsor* leve.
* **Saída**: `features_4H.parquet` (index datetime, colunas numéricas + `REGIME` base).

### 3.3 02_quantiles_bands

* **Entrada**: `features_4H.parquet`
* **Processo**:

  * Alvos `y_T = log(S_{t+T}) - log(S_{t+1})` (**t+1** anti-leakage) para T∈{42,84}.
  * **Split temporal** (Train/Calib/Test). **Dropar últimas `T+1` por regime**.
  * Modelos de **quantis** (q ∈ {0.1,0.2,0.5,0.8,0.9}) por **regime** com **Mondrian por volatilidade** (`ATR_PCT`):

    * Quando N< limiar: quantis **empíricos** (baseline).
    * Quando N suficiente: **GBR quantile** (tuning com **Purged CV**).
* **Saída**: `preds_4H_T{42,84}.parquet` (+ `.meta.json` com *splits*, *env*, *hash*, *grid*).

### 3.4 02C_onpolicy_recalib_OOS (v3)

* **Entrada**: `preds_4H_T*.parquet`, `features_4H.parquet`
* **Processo (apenas no Calib on-policy)**:

  * *Gating* on-policy via **histerese ADX** (se `ADX14` disponível; senão `REGIME` salvo).
  * Buckets de vol por **tercis** (ou 4 quantis se N grande), **aprendidos no Calib**.
  * Ajustes por **bucket** e **regime**:

    1. **δ50** (*shift* da mediana)
    2. **γL/γU** (abertura assimétrica) para [10,90] & [20,80]
    3. **m** (*mid-band*) para abrir o miolo sem degradar [10,90]
  * **Embargo**: remove últimas `T` do Calib e primeiras `T` do Test.
  * **Aplicar parâmetros congelados no Test** → métricas **OOS**.
* **Saída**:

  * `preds_4H_T*_onpolicy_oos.parquet`
  * `preds_4H_T*_onpolicy_oos.meta.json` (parâmetros por regime×bucket; *splits*, *edges*)
  * `qc_onpolicy_oos.json` (coberturas OOS por regime).

### 3.5 03_options_mapper

* **Entrada**:

  * `preds_4H_T42_onpolicy_oos.parquet` (q’s)
  * `features_4H.parquet` (ADX para regime corrente)
  * **Chain** D+7..D+10 (CSV/JSON): `expiry, type, strike, bid, ask, mid, iv, delta, oi`
* **Processo (hoje, TREND 7d)**:

  * Validar **TREND ON** (histerese).
  * **K1** ≈ strike entre ATM e Δ~0.35–0.55; **K2** ≈ **`S0·exp(Q80)`** (ajuste ao strike disponível).
  * **POP_P** (prob de lucro) por **P**: uso de interpolação entre quantis.
  * **EV_P** (valor esperado) por **P**: MC simples (10k) com interpolação + *bootstrap* de resíduos.
  * **Regras de qualidade**:

    * POP_P ≥ **60%**;
    * Débito ≤ **40%** da largura do spread;
    * Spread *mid* coerente (bid/ask < ~5–6% do *mid* no ATM).
  * **Stop subjacente**: `max(1.3–1.8×ATR, último HL)`; **parcial** em +1R ou **50–70%** do valor máximo.
* **Saída**: `trade_card.json` (ver §6).

### 3.6 04_reporting

* **Entrada**: `trade_card.json`, `qc_onpolicy_oos.json`, gráficos (opcional).
* **Saída**: `daily_report.html` (+ PNGs) com:

  * Resumo de regime, bandas, sinal, POP/EV, preço/greeks, e *checklist* de qualidade.

---

## 4) Gating por Regime (ADX com histerese)

**Estados**: `1=TREND`, `-1=RANGE`, `0=NEUTRO`.

**Histerese (padrão)**

* **Trend ON** se **ADX ≥ 25**; **Trend OFF** se **ADX ≤ 23**.
* **Range ON** se **ADX < 20**; **Range OFF** se **ADX ≥ 22**.
* **Neutro** no restante; **não operar**.

**Filtro de estabilidade** (opcional, para 02 original): exigir **k** barras consecutivas em estado ON antes de emitir (MVP: `k=2`; antes: `k=3`).

**On-policy**: apenas onde **bandas válidas** & **regime ativo**.

---

## 5) Modelagem de Bandas & Calibração

**Quantis por regime + vol (Mondrian)**

* Quantis (0.1, 0.2, 0.5, 0.8, 0.9) com features: ADX/DI±, ATR/ATR_PCT, largura Donchian, desvios vs EMA20/VWAP/DC_mid, *flags* de *breakout/cross*;
* Buckets de vol: **tercis** em `ATR_PCT` no **Calib do regime** (winsor `[0.1%,99.9%]`).

**Recalibração OOS (v3)**

* Aprender **δ50, γL/γU, m** **somente no Calib on-policy**;
* **Embargo** `T` entre Calib e Test;
* Aplicar no **Test** e medir coberturas OOS.

**Metas de cobertura (on-policy, OOS)**

* Intervalo **[q10,q90]**: **80% ± 3pp**
* Intervalo **[q20,q80]**: **60% ± 3pp**
* **Monotonicidade por linha** sempre aplicada.

**Consistência entre horizontes**

* Mediana da **largura(T=84)** ≥ mediana **largura(T=42)**; quando necessário, **reescala distâncias relativas a q50** (por regime×bucket).

---

## 6) Mapeamento para Opções (TREND 7d)

**Entradas mínimas da chain**

* `expiry` (data), `strike` (float), `type` (‘C’/‘P’), `bid`, `ask`, `mid`, `iv`, `delta`, `oi`.

**Regras (call debit spread)**

1. Calcular **S0** (close 4H mais recente).
2. **K1**: strike entre ATM e Δ~0.35–0.55 (escolher com melhor *microstructure*: spread menor, OI > 0).
3. **K2**: aproximar **`S0·exp(Q80)`** ao strike disponível (não ultrapassar distância máxima).
4. **POP_P**: `P(S_T ≥ K2)` via interpolação dos quantis/bandas (ou MC).
5. **Preço teórico P** (EV_P): MC de `S_T` sob P → payoff do spread → média.
6. **Qualidade**: POP_P ≥ 60%; Débito ≤ 40% da largura; spreads aceitáveis (bid/ask).
7. **Gestão**:

   * **Stop** no subjacente (`max(1.3–1.8×ATR, último HL)`);
   * **Parcial**: encerrar a 50–70% do valor máximo do spread;
   * **Saída**: mudança de regime (histerese) → fechar.

**Saída (trade_card.json)**

```json
{
  "system": "options-prime",
  "as_of": "YYYY-MM-DDTHH:MMZ",
  "underlying": {"symbol":"BTCUSD","S0": 116500.0, "TF":"4H"},
  "regime": "TREND",
  "horizon_T": 42,
  "bands": {"q10": -0.035, "q20": -0.012, "q50": 0.004, "q80": 0.028, "q90": 0.045},
  "structure": "CallDebitSpread",
  "legs": [
    {"type":"C","strike":116000,"side":"LONG","mid":0.0220,"iv":0.316,"delta":0.52},
    {"type":"C","strike":120000,"side":"SHORT","mid":0.0086,"iv":0.305,"delta":0.22}
  ],
  "debit": 0.0134,
  "max_payoff": 0.0343,
  "pop_P": 0.64,
  "ev_P": 0.0061,
  "risk": {
    "stop_underlying": 114400.0,
    "R_atr": 1.5,
    "partial_rule": "50-70% do valor máximo",
    "regime_exit": "ADX <= 23 (histerese)"
  },
  "notes": "Gate TREND ativo; T=7d; K2≈S0·exp(Q80); bid/ask OK"
}
```

> **RANGE 7d (iron condor)**: ativar **somente** quando coberturas OOS atenderem metas. Regras previstas: shorts em **Q20/Q80**, asas em **Q10/Q90**, crédito ≥ 30–40% da largura, POP_P ≥ 60–65%.

---

## 7) Métricas, *Quality Gates* e Relatórios

**QC OOS (on-policy, por regime)**

* `cover_10_90_OOS` (meta 80±3pp)
* `cover_20_80_OOS` (meta 60±3pp)
* `median_width` por T & regime
* `emission_rate` (linhas não-neutras com bandas válidas)

**Relatório diário (`04_reporting`)**

* Regime atual (ADX, histerese) e gráficos simples (preço, bandas).
* Card do trade (estrutura, strikes, POP_P, EV_P, R, greeks).
* *Checklist* de qualidade e observações.

**Critérios de aprovação para produção (MVP)**

* **TREND T=42** OOS: metas batidas (✔) → **ON**.
* **RANGE** e **T=84**: **OFF** até coberturas OOS atenderem metas por ≥ 2 meses.

---

## 8) Anti-overfitting & Reprodutibilidade

* Ajustes (**δ50, γL/U, m**) **exclusivamente** no **Calib**, métricas **apenas no Test**, com **embargo=T**.
* **Pooling** por regime quando `n_bucket` baixo; **buckets dinâmicos** (3→2→1) conforme N.
* **Caps** e *shrink* em γ/m quando N é baixo.
* *Seeds*, versões, *data hash* e *commit hash* registrados no `.meta`.
* **Walk-forward** (mensal/semanal) recomendado na evolução pós-MVP.

---

## 9) Roadmap de Melhorias (curto prazo)

1. **02C v3.1**

   * *Pooling* por regime quando `n_bucket<100`;
   * Buckets dinâmicos por N;
   * *Mid-band* obrigatório em T=84 quando [20,80] < meta mas [10,90] ok;
   * Fallback T=42→T=84 por **fator de largura** (distâncias relativas a q50).

2. **Ativar RANGE 7d** assim que coberturas OOS estabilizarem (≥2 meses).

3. **Comparação IV vs σ_P** automática (edge score): usar IV ATM 7–10d e σ_P central para priorizar estruturas (comprar quando IV≪σ_P; vender quando IV≫σ_P).

4. **Diag/calendar** (opcional) para suavizar vega em TREND.

---

## 10) Operação Diária (runbook)

1. **00/01** (após *bar close* 4H): atualizar `features_4H.parquet`.
2. **02C v3 OOS**: carregar `preds_T42_onpolicy_oos.parquet`.
3. Checar **regime** (histerese ADX).
4. Se **TREND ON** → rodar **03_options_mapper** (T=7d).
5. Verificar **POP_P/EV_P**, *checklist*, custos; gerar `trade_card.json`.
6. **04_reporting**: emitir `daily_report.html`; logar em `signals.csv`.
7. Gestão/encerramento conforme regras (stop/partial/regime flip).

---

## 11) Convenções, Config & Logs

**Arquivos**

* `data/processed/features/features_4H.parquet`
* `data/processed/preds/preds_4H_T{42,84}.parquet`
* `data/processed/preds/preds_4H_T{42,84}_onpolicy_oos.parquet`
* `.meta.json` correspondentes; `qc_onpolicy_oos.json`
* `chains/YYYY-MM-DD_chain_7_10d.csv`
* `signals.csv`, `winners.csv` (opcional), `daily_report.html`

**Config (YAML/JSON)**

* Limiar ADX e histerese; k estável; metas de cobertura; T=42..60 (7–10d); caps γ/m; seeds; pastas.

**Logs**

* Passo, timestamp UTC, N por regime/bucket, parâmetros aprendidos, métricas OOS, *emission rate*, erros.

---

## 12) Segurança & Risco

* **Somente** estruturas de **risco limitado** (spreads/condors).
* Tamanho por **R** (distância de stop no subjacente); **máx 1R por trade**.
* **Nunca naked**; atenção a feriados, *events* e liquidez.
* **Não é recomendação de investimento**; uso educacional/experimental.

---

## 13) Glossário (rápido)

* **IV**: volatilidade implícita (mercado).
* **RV**: volatilidade realizada (ex-post).
* **POP**: probabilidade de lucro sob **P** (nossa distribuição).
* **EV_P**: valor esperado sob **P**.
* **On-policy**: janelas elegíveis (bandas válidas + regime ativo).
* **Embargo**: *gap* temporal entre Calib e Test para evitar contaminação.

---

### Estado de Conformidade (hoje)

* **TREND · 7d (T=42)** → **ON** (mapper ativo).
* **RANGE · 7d** e **14d (T=84)** → **OFF** (aguarda v3.1).

Se quiser, transformo este documento em um **README.md** e um **config.yaml** inicial para você versionar no repositório do projeto.
