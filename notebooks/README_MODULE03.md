# BTC Momentum Analysis — Module 02 Only

## Objetivo

Produzir **sinais de momentum** a partir das **bandas quantílicas** e **RV̂** entregues pelo **Módulo 02**, úteis para:
- Leitura tática do mercado
- Pré-checagem ao Módulo 03 (estratégias de opções)
- Análise de regime de volatilidade

## Escopo de Dados

**Entradas:**
- `preds_T=*.parquet` com T ∈ {42, 48, 54, 60}
- Campos obrigatórios: `ts0`, `T`, `S0`, `q05`, `q25`, `q50`, `q75`, `q95`
- Campos opcionais: `rvhat_ann`, `har_rv_ann_T`, `p05`-`p95`

**Saídas:**
- `momentum_timeseries.parquet`: série temporal completa com scores
- `momentum_snapshot.csv`: snapshot do último estado por horizonte
- `momentum_report.html`: relatório visual com gráficos
- `charts/`: visualizações PNG

## Estrutura do Notebook

### 1. Setup & Configuration
- Imports e configuração de ambiente
- Parâmetros via dicionário Python (lookback, horizontes, pesos)

### 2. Data Discovery & Loading
- Descoberta automática de arquivos `preds_T=*.parquet`
- Concatenação e validação inicial

### 3. Data Validation & Preprocessing
- Monotonicidade dos quantis (q05 ≤ q25 ≤ q50 ≤ q75 ≤ q95)
- Verificação de recência (últimos N dias)
- Conversão para UTC e filtragem de horizontes válidos

### 4. Feature Engineering
- Cálculo de bandas absolutas (p = S0 × q)
- Métricas derivadas:
  - `width_5_95`: largura relativa das bandas
  - `width_pct`: percentil de volatilidade
  - `tilt_mid`: assimetria em torno da mediana
  - `tilt_ratio`: força direcional
  - `slope_q50`: inclinação da mediana
  - `rv_delta`: pressão de volatilidade

### 5. Scoring System
- **Direcional (D ∈ [-1, 1])**: combina tilt_ratio e slope
- **Volatilidade (V ∈ [0, 1])**: combina width_pct e rv_delta
- **Confiança (C ∈ [0, 1])**: consistência entre horizontes + estabilidade
- **MomentumIndex**: score composto D × (0.5 + 0.5×C)

### 6. Visualizations
- Fan charts (bandas de confiança)
- Série temporal de RV̂ vs HAR-RV
- Regime de volatilidade (width_pct)
- Scores de momentum (D, V, C, Index)
- Heatmaps por horizonte

### 7. Pre-checks for Module 03
Heurísticas para estratégias de opções:
- **Condor/Butterfly**: vol alta + direção neutra
- **Calendar/Diagonal**: vol baixa + direção forte
- **Vertical debit**: direção forte

### 8. Export & Quality Control
- Exportação de todos os artefatos
- QC final com critérios GO/NO-GO

## Configuração

```python
CONFIG = {
    'lookback_days': 180,
    'horizons': [42, 48, 54, 60],
    'target_T_default': 48,
    'tilt_strength_hi': 1.2,
    'vol_hi_pct': 0.80,
    'vol_lo_pct': 0.20,
    'recency_max_days': 7,
    'score_weights': {
        'directional': {'tilt': 0.6, 'slope': 0.4},
        'volatility': {'width_pct': 0.7, 'rv_delta': 0.3},
        'confidence': {'consistency': 0.5, 'stability': 0.5}
    }
}
```

## Como Usar

1. **Pré-requisitos**: ter executado o Módulo 02 (treinamento de modelos)
2. **Verificar dados**: arquivos `data/processed/preds/preds_T=*.parquet`
3. **Executar notebook**: células sequencialmente
4. **Verificar outputs**: `data/processed/momentum/`

## Campos das Saídas

### momentum_timeseries.parquet
- **Chaves**: ts0 (UTC), T
- **Métricas**: S0, q05-q95, p05-p95, width_*, tilt_*, slope_*, rv_*
- **Scores**: D, V, C, MomentumIndex
- **Metadados**: consistency_T, stability_T

### momentum_snapshot.csv
- Última linha por horizonte T
- Classificação de regime: {ALTA|NEUTRA|BAIXA}
- Todos os scores atualizados

### momentum_report.html
- Metadados de execução
- Snapshots por horizonte
- Gráficos interativos
- Tabela de pré-checagens para Módulo 03

## Critérios de Qualidade

✅ **Inputs OK**: ≥1 arquivo válido, 100% linhas com monotonicidade  
✅ **Horizontes**: apenas T ∈ {42,48,54,60}  
✅ **Métricas**: todas colunas essenciais presentes  
✅ **Relatório**: HTML + ≥4 imagens geradas  
✅ **Sanidade**: D∈[-1,1], V,C∈[0,1], Index∈[-1,1]

## Status de Desenvolvimento

- [x] Estrutura inicial e configuração
- [x] Data discovery e loading
- [x] Validações básicas
- [ ] Feature engineering (métricas derivadas)
- [ ] Sistema de scoring
- [ ] Visualizações
- [ ] Pré-checagens para Módulo 03
- [ ] Export e relatório HTML
- [ ] QC final

## Observações

- **Timezone**: sempre UTC
- **Determinismo**: seed fixo (42)
- **Robustez**: winsorização para normalização de scores
- **Sem leakage**: apenas séries de previsões, sem realized futuro

---

**Branch**: `feature/btc-momentum-02-only`  
**Data criação**: October 2, 2025  
**Autor**: brunocapelao
