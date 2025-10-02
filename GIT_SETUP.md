# Git Repository Setup - Quantitative Trading System

## Estrutura do Repositório

Este repositório contém um sistema completo de trading quantitativo usando CQR (Conformal Quantile Regression) com LightGBM.

### Branches Strategy

- `main`: Branch principal com código de produção
- `development`: Branch de desenvolvimento para novas features
- `hotfix/*`: Branches para correções urgentes

### Workflow Recomendado

1. **Para novas features:**
   ```bash
   git checkout -b feature/nome-da-feature
   # desenvolvimento...
   git add .
   git commit -m "feat: descrição da feature"
   git push origin feature/nome-da-feature
   # criar pull request para development
   ```

2. **Para correções:**
   ```bash
   git checkout -b fix/nome-da-correcao
   # correção...
   git add .
   git commit -m "fix: descrição da correção"
   git push origin fix/nome-da-correcao
   ```

3. **Para hotfixes:**
   ```bash
   git checkout -b hotfix/nome-do-hotfix
   # correção urgente...
   git add .
   git commit -m "hotfix: descrição do hotfix"
   git push origin hotfix/nome-do-hotfix
   ```

### Convenções de Commit

Seguimos o padrão [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` nova funcionalidade
- `fix:` correção de bug
- `docs:` documentação
- `style:` formatação (sem mudança no código)
- `refactor:` refatoração
- `test:` testes
- `chore:` tarefas de manutenção

### Arquivos Importantes

#### Sempre Commitados
- `src/`: Código fonte principal
- `config/`: Configurações do sistema
- `notebooks/`: Notebooks organizados e limpos
- `requirements.txt`: Dependências
- `README.md`: Documentação principal
- `setup.py`: Configuração do pacote

#### Ignorados pelo Git (.gitignore)
- `data/processed/`: Dados processados (muito grandes)
- `data/raw/`: Dados brutos
- `__pycache__/`: Cache do Python
- `*.joblib`: Modelos treinados (muito grandes)
- `*.parquet`: Arquivos de dados
- `.venv/`: Ambiente virtual

### Tags de Versão

Para marcar releases importantes:

```bash
# Criar tag anotada
git tag -a v1.0.0 -m "Versão 1.0.0: Sistema CQR LightGBM produção"
git push origin v1.0.0

# Listar tags
git tag -l
```

### Comandos Úteis

```bash
# Status do repositório
git status

# Ver diferenças
git diff

# Histórico compacto
git log --oneline --graph

# Ver arquivos ignorados
git ls-files --others --ignored --exclude-standard

# Limpar arquivos não rastreados
git clean -fd
```

### Backup e Sincronização

Para configurar um repositório remoto:

```bash
# Adicionar remote
git remote add origin https://github.com/user/quant-trading-cqr.git

# Push inicial
git push -u origin main

# Push de tags
git push origin --tags
```

### Estrutura de Dados

⚠️ **Importante**: Os arquivos de dados não são commitados devido ao tamanho:

- `data/processed/preds/`: ~67MB de modelos treinados
- `data/processed/features/`: Features processadas
- `data/raw/`: Dados históricos BTCUSD

Para reproduzir o ambiente completo:
1. Clone o repositório
2. Instale dependências: `pip install -r requirements.txt`
3. Execute notebooks de preparação de dados
4. Treine modelos: `python run_train.py --config config/base.yaml`

---

*Repositório configurado em: $(date)*
*Commit inicial: ef0bafc*