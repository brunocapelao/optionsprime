# Contributing Guide

## 🔧 Setup de Desenvolvimento

### 1. Clone e Configure o Ambiente

```bash
# Clone o repositório
git clone git@github.com:brunocapelao/optionsprime.git
cd optionsprime

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Instale ferramentas de desenvolvimento
pip install nbstripout pre-commit black flake8 isort pytest

# Configure os hooks do Git
nbstripout --install
pre-commit install
```

### 2. Estrutura do Projeto

```
project/
├── notebooks/          # Jupyter notebooks (SEM outputs commitados)
│   ├── 00_*.ipynb     # Data collection
│   ├── 01_*.ipynb     # Feature engineering
│   ├── 02_*.ipynb     # Model training & validation
│   └── 0X_*.ipynb     # Analysis & experiments
├── src/quant_bands/   # Código Python produção
│   ├── train.py       # Pipeline de treinamento
│   ├── predict.py     # Pipeline de predição
│   └── utils.py       # Funções auxiliares
├── data/              # Dados (versionados com DVC se > 100MB)
│   ├── raw/          # Dados brutos (somente leitura)
│   └── processed/    # Features, modelos, resultados
├── config/           # Arquivos de configuração
└── tests/            # Testes unitários
```

## 🌿 Workflow de Branches

### Estratégia de Branches

```
main                    # Código estável e testado
├── develop            # Desenvolvimento ativo
    ├── feature/nome   # Nova funcionalidade
    ├── exp/nome       # Experimento ML (pode ser descartado)
    ├── data/nome      # Atualização de dados
    └── fix/nome       # Correção de bug
```

### Convenção de Commits

Use **commits semânticos** para facilitar o histórico:

```bash
# Features e Funcionalidades
feat: Add LSTM model for price prediction
feat(features): Add momentum indicators

# Experimentos de ML
exp: Test XGBoost with different hyperparameters
exp(model): Compare ensemble methods

# Dados
data: Update BTC data until Oct 2025
data(raw): Add ETH historical prices

# Correções
fix: Correct feature scaling in pipeline
fix(predict): Handle missing values in inference

# Performance
perf: Optimize feature computation with Numba
perf(train): Reduce memory usage in CV loop

# Refatoração
refactor: Modularize training pipeline
refactor(utils): Simplify data loading logic

# Documentação
docs: Add usage examples to README
docs(notebooks): Document experiment methodology

# Testes
test: Add unit tests for conformal prediction
test(cv): Validate cross-validation splits

# Chores
chore: Update dependencies
chore(config): Add new training configuration
```

## 📓 Trabalhando com Notebooks

### ⚠️ Regras Importantes

1. **NUNCA commite outputs de notebooks** ✋
   - O `nbstripout` remove automaticamente
   - Se esqueceu, rode: `nbstripout notebooks/*.ipynb`

2. **Execute células em ordem sequencial**
   - Restart kernel + Run All antes de commitar
   - Garante reprodutibilidade

3. **Documente suas análises**
   - Use células Markdown para explicar o que está fazendo
   - Inclua conclusões e próximos passos

4. **Notebooks são para exploração, não produção**
   - Código pronto → mova para `src/`
   - Mantenha notebooks concisos e focados

### Exemplo de Workflow

```bash
# Crie uma branch para seu experimento
git checkout -b exp/test-lightgbm-v2

# Trabalhe no notebook
jupyter lab notebooks/02_experiments.ipynb

# Antes de commitar, verifique se outputs foram removidos
nbstripout notebooks/02_experiments.ipynb

# Commit
git add notebooks/02_experiments.ipynb
git commit -m "exp: test LightGBM with custom objective function"

# Push e abra PR
git push origin exp/test-lightgbm-v2
```

## 🧪 Reprodutibilidade

### Seeds Aleatórias

Sempre configure seeds no início dos notebooks:

```python
import random
import numpy as np
import lightgbm as lgb

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
lgb.set_random_state(SEED)
```

### Rastreamento de Experimentos

Para experimentos importantes, documente:

```python
experiment_config = {
    "model": "LightGBM",
    "version": "v2.3",
    "features": ["momentum", "volatility", "volume"],
    "params": {
        "learning_rate": 0.01,
        "max_depth": 5,
        "n_estimators": 1000
    },
    "data": {
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
        "timeframe": "4H"
    },
    "cv_strategy": "TimeSeriesSplit(n_splits=5)"
}

# Salve os resultados
import json
with open("data/processed/models/experiment_v2.3.json", "w") as f:
    json.dump(experiment_config, f, indent=2)
```

## 🧹 Código Limpo

### Formatação Automática

Os hooks do pre-commit executam automaticamente:
- **black**: Formatação de código Python
- **isort**: Organização de imports
- **flake8**: Linting

Para rodar manualmente:

```bash
# Formatar código
black src/ tests/

# Organizar imports
isort src/ tests/

# Verificar qualidade
flake8 src/ tests/
```

### Testes

```bash
# Rodar todos os testes
pytest tests/

# Rodar com cobertura
pytest --cov=src tests/

# Rodar teste específico
pytest tests/test_train.py::test_cv_split
```

## 📦 Dependências

### Adicionar Nova Dependência

```bash
# Instale a dependência
pip install new-package

# Atualize requirements.txt
pip freeze > requirements.txt

# Ou manualmente (preferível):
echo "new-package==1.2.3" >> requirements.txt

# Commit
git add requirements.txt
git commit -m "chore: add new-package dependency"
```

### Boas Práticas

- ✅ Fixe versões exatas: `pandas==2.0.3`
- ✅ Documente o motivo da dependência
- ✅ Mantenha requirements.txt organizado por categoria

## 🚀 Pull Requests

### Checklist antes de abrir PR

- [ ] Código formatado (black, isort)
- [ ] Notebooks sem outputs (nbstripout)
- [ ] Testes passando (pytest)
- [ ] README atualizado (se necessário)
- [ ] Commits seguem convenção semântica
- [ ] Branch atualizada com main/develop

### Template de PR

```markdown
## Descrição
[Descreva o que foi implementado/alterado]

## Motivação
[Por que essa mudança é necessária?]

## Resultados
[Para experimentos ML: métricas, gráficos, conclusões]

## Checklist
- [ ] Código testado
- [ ] Notebooks sem outputs
- [ ] Documentação atualizada
```

## 📊 Dados e Modelos Grandes

### Quando usar DVC

Se arquivos > 10MB ou > 100 arquivos:

```bash
# Instalar DVC
pip install dvc

# Inicializar
dvc init

# Rastrear dados
dvc add data/processed/features/features_4H.parquet

# Commit o .dvc file (não os dados)
git add data/processed/features/features_4H.parquet.dvc .gitignore
git commit -m "data: track features with DVC"

# Configurar storage remoto
dvc remote add -d myremote s3://mybucket/dvc-storage
dvc push
```

### Boas Práticas

- ✅ Dados raw são **imutáveis** (nunca modifique)
- ✅ Versionamento de features e modelos com DVC
- ✅ Documente transformações aplicadas
- ✅ Use nomes descritivos: `features_4H_v2.parquet`

## 🆘 Problemas Comuns

### "Pre-commit hooks falharam"

```bash
# Ver o que falhou
git commit -m "message"  # Mostra os erros

# Corrigir automaticamente
black src/
isort src/

# Tentar novamente
git add .
git commit -m "message"
```

### "Notebook com outputs commitado"

```bash
# Limpar notebook
nbstripout notebooks/problematic.ipynb

# Re-commitar
git add notebooks/problematic.ipynb
git commit --amend --no-edit
```

### "Arquivo muito grande no Git"

```bash
# Remover do último commit
git rm --cached large_file.parquet
git commit --amend -m "Remove large file"

# Adicionar ao DVC
dvc add large_file.parquet
git add large_file.parquet.dvc .gitignore
git commit -m "data: track large_file with DVC"
```

## 📚 Recursos

- [Git Flow](https://nvie.com/posts/a-successful-git-branching-model/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [nbstripout](https://github.com/kynan/nbstripout)
- [DVC Documentation](https://dvc.org/doc)
- [Pre-commit](https://pre-commit.com/)

## 🤝 Code Review

### Para Revisores

- ✅ Código segue as convenções do projeto?
- ✅ Notebooks sem outputs?
- ✅ Experimentos ML documentados?
- ✅ Resultados reproduzíveis?
- ✅ Testes adequados?

### Para Autores

- Seja receptivo a feedback
- Explique decisões técnicas
- Mantenha PRs pequenos e focados
- Responda comentários rapidamente

---

**Dúvidas?** Abra uma issue ou contate o time! 🚀
