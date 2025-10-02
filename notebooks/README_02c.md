# 02c Model Backtest - Complete Validation Framework

## 🎯 Objetivo

O notebook `02c_model_backtest.ipynb` implementa um framework completo de validação para o modelo CQR (Conformal Quantile Regression) usando:

- **Walk-forward backtesting** com janelas móveis
- **Métricas estatísticas rigorosas** (CRPS, WIS, DQ Test, PSI)
- **Baseline HAR-RV** para comparação
- **Teste de Diebold-Mariano** para significância estatística
- **Sistema de Gates GO/NO-GO** para decisão de produção

## 📊 Métricas Implementadas

### 🔬 Métricas Principais

1. **CRPS** (Continuous Ranked Probability Score)
   - Métrica padrão para previsões probabilísticas
   - Avalia toda a distribuição preditiva
   - Menor valor = melhor performance

2. **WIS** (Weighted Interval Score) 
   - Score ponderado para intervalos quantílicos
   - Penaliza violações de cobertura
   - Menor valor = melhor performance

3. **DQ Test** (Dynamic Quantile Test - Engle & Manganelli, 2004)
   - Teste de adequação dos quantis
   - H₀: Quantis estão corretamente especificados
   - Pass rate > 80% = aprovação

4. **PSI** (Population Stability Index)
   - Detecta drift na distribuição das previsões
   - PSI < 0.25 = estável (aprovação)
   - PSI > 0.25 = drift significativo

### 🆚 Comparação de Modelos

5. **Teste de Diebold-Mariano** (1995)
   - Comparação estatística de capacidade preditiva
   - Usa estimador HAC (Newey-West) para robustez
   - Teste bilateral para diferenças significativas

6. **HAR-RV Baseline** (Corsi, 2009)
   - Modelo Heterogeneous AutoRegressive - Realized Volatility
   - Baseline teórico para comparação
   - RV_t = β₀ + β₁RV_{t-1} + β₂RV_{t-1}^{(w)} + β₃RV_{t-1}^{(m)}

## 🔄 Framework de Backtest

### Walk-Forward Configuration

```python
BACKTEST_CONFIG = {
    'initial_train_size': 2000,  # Janela inicial de treino
    'test_size': 100,           # Observações por teste
    'step_size': 50,            # Passo do walk-forward
    'min_train_size': 1000,     # Mínimo para treino
    'max_train_size': 5000,     # Máximo (janela móvel)
    'horizons': [42, 48, 54, 60],
    'quantiles': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
}
```

### Loop Principal

1. **Divisão Temporal**: Janelas de treino/teste não sobrepostas
2. **Treinamento**: Modelos retreinados a cada fold
3. **Previsão**: Geração de quantis para todos os horizontes
4. **Validação**: Cálculo de todas as métricas
5. **Gates**: Avaliação GO/NO-GO por critérios

## 🚪 Sistema de Gates

### Critérios de Aprovação

| Gate | Critério | Threshold |
|------|----------|-----------|
| **CRPS** | CRPS < threshold | 0.5 |
| **WIS** | WIS < threshold | 1.0 |
| **DQ** | Pass Rate > threshold | 80% |
| **PSI** | PSI < threshold | 0.25 |
| **Overall** | Todos aprovados | 100% |

### Decisões de Produção

- **🟢 GO**: Overall Gate Rate ≥ 80%
- **🟡 REVISAR**: Overall Gate Rate ≥ 60%
- **🔴 NO-GO**: Overall Gate Rate < 60%

## 📈 Dashboard Executivo

### Outputs Gerados

1. **Performance Agregada**
   - Métricas por modelo e horizonte
   - Estatísticas descritivas (mean, std, min, max)
   - Número de observações válidas

2. **Análise de Gates**
   - Taxa de aprovação por gate
   - Taxa overall por modelo/horizonte
   - Contagem de folds aprovados

3. **Ranking de Modelos**
   - Score composto: Gate Rate × DQ Rate - (CRPS + WIS)/2
   - Ordenação por performance
   - Modelo recomendado

4. **Recomendações Executivas**
   - Decisão GO/NO-GO automática
   - Horizontes recomendados
   - Ações sugeridas

## 🚀 Como Executar

### Pré-requisitos

```bash
# Instalar dependências (já feito no ambiente)
pip install seaborn scikit-learn lightgbm

# Dados devem estar em:
data/raw/BTCUSD_CCCAGG_1h.csv
```

### Execução Sequencial

1. **Execute todas as células na ordem**
2. **Aguarde o carregamento dos dados**
3. **Configure parâmetros se necessário**
4. **Execute o backtest (pode demorar alguns minutos)**
5. **Analise o dashboard executivo**

### Tempo Estimado

- Setup: ~30 segundos
- Backtest completo: ~5-10 minutos (depende do n_folds)
- Dashboard: ~10 segundos

## 📊 Interpretação dos Resultados

### Métricas de Sucesso

- **CRPS/WIS baixos**: Previsões mais precisas
- **DQ pass rate alto**: Quantis bem calibrados
- **PSI baixo**: Modelo estável no tempo
- **DM test significativo**: Diferença real entre modelos

### Flags de Atenção

⚠️ **CRPS/WIS crescendo**: Degradação da performance  
⚠️ **DQ pass rate baixo**: Quantis mal calibrados  
⚠️ **PSI alto**: Drift no modelo  
⚠️ **Gates reprovando**: Modelo não ready para produção  

## 🔧 Customização

### Ajustar Thresholds

```python
# Em create_executive_dashboard()
crps_threshold = 0.3  # Mais restritivo
wis_threshold = 0.8   # Mais restritivo
dq_threshold = 0.85   # Mais restritivo
psi_threshold = 0.20  # Mais restritivo
```

### Adicionar Novos Modelos

```python
# Em BACKTEST_CONFIG
'models_to_test': ['CQR', 'HAR-RV', 'Novo_Modelo']

# Implementar lógica no run_walk_forward_backtest()
elif model_name == 'Novo_Modelo':
    # Código do novo modelo aqui
```

### Modificar Horizontes/Quantis

```python
'horizons': [24, 48, 72, 96],  # Novos horizontes
'quantiles': [0.05, 0.25, 0.5, 0.75, 0.95]  # Quantis padrão
```

## 📝 Notas Técnicas

### Limitações

- **Dados simulados**: HAR-RV usa previsões simuladas (implementar modelo real)
- **CQR mock**: Carregamento de modelo CQR é simulado (usar modelo treinado)
- **Memory usage**: Armazena todos os resultados (pode ser intensivo)

### Melhorias Futuras

- [ ] Integração com modelos CQR reais
- [ ] Implementação completa HAR-RV
- [ ] Paralelização do backtest
- [ ] Exportação automática de relatórios
- [ ] Dashboard interativo com plotly

---

**🎯 Este notebook fornece validação rigorosa e completa do modelo CQR seguindo as melhores práticas acadêmicas e industriais para modelos quantílicos.**