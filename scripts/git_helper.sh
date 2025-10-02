#!/bin/bash

# Git Helper Script for Quantitative Trading System
# Facilita operações comuns do Git

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para exibir ajuda
show_help() {
    echo -e "${BLUE}Git Helper Script - Quantitative Trading System${NC}"
    echo ""
    echo "Uso: $0 [comando] [argumentos]"
    echo ""
    echo "Comandos disponíveis:"
    echo "  status     - Mostra status detalhado do repositório"
    echo "  clean      - Limpa arquivos temporários e não rastreados"
    echo "  backup     - Cria backup do estado atual"
    echo "  feature    - Cria nova branch de feature"
    echo "  fix        - Cria nova branch de correção"
    echo "  sync       - Sincroniza com repositório remoto"
    echo "  log        - Mostra histórico visual"
    echo "  size       - Mostra tamanho do repositório"
    echo "  help       - Mostra esta ajuda"
    echo ""
    echo "Exemplos:"
    echo "  $0 feature nova-estrategia"
    echo "  $0 fix corrigir-bug-dados"
    echo "  $0 clean"
}

# Status detalhado
git_status() {
    echo -e "${BLUE}=== Status do Repositório ===${NC}"
    git status --short --branch
    echo ""
    
    echo -e "${BLUE}=== Últimos 5 Commits ===${NC}"
    git log --oneline -5
    echo ""
    
    echo -e "${BLUE}=== Branches ===${NC}"
    git branch -a
    echo ""
}

# Limpeza
git_clean() {
    echo -e "${YELLOW}Limpando arquivos temporários...${NC}"
    
    # Limpar cache Python
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    
    # Limpar arquivos temporários
    find . -name ".DS_Store" -delete 2>/dev/null || true
    find . -name "*.tmp" -delete 2>/dev/null || true
    find . -name "*~" -delete 2>/dev/null || true
    
    # Git clean
    git clean -fdx --exclude=".venv" --exclude="data/processed" --exclude="data/raw"
    
    echo -e "${GREEN}✅ Limpeza concluída${NC}"
}

# Criar feature branch
create_feature() {
    if [ -z "$1" ]; then
        echo -e "${RED}❌ Nome da feature é obrigatório${NC}"
        echo "Uso: $0 feature nome-da-feature"
        exit 1
    fi
    
    branch_name="feature/$1"
    echo -e "${BLUE}Criando branch: $branch_name${NC}"
    
    git checkout -b "$branch_name"
    echo -e "${GREEN}✅ Branch $branch_name criada e ativada${NC}"
}

# Criar fix branch
create_fix() {
    if [ -z "$1" ]; then
        echo -e "${RED}❌ Nome da correção é obrigatório${NC}"
        echo "Uso: $0 fix nome-da-correcao"
        exit 1
    fi
    
    branch_name="fix/$1"
    echo -e "${BLUE}Criando branch: $branch_name${NC}"
    
    git checkout -b "$branch_name"
    echo -e "${GREEN}✅ Branch $branch_name criada e ativada${NC}"
}

# Backup
create_backup() {
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_name="backup_$timestamp"
    
    echo -e "${BLUE}Criando backup: $backup_name${NC}"
    git tag -a "$backup_name" -m "Backup automático - $timestamp"
    
    echo -e "${GREEN}✅ Backup criado: $backup_name${NC}"
    echo "Para restaurar: git checkout $backup_name"
}

# Sincronizar
sync_repo() {
    echo -e "${BLUE}Sincronizando com repositório remoto...${NC}"
    
    if git remote | grep -q origin; then
        git fetch origin
        git pull origin $(git branch --show-current) --rebase
        echo -e "${GREEN}✅ Sincronização concluída${NC}"
    else
        echo -e "${YELLOW}⚠️  Nenhum repositório remoto configurado${NC}"
        echo "Configure com: git remote add origin <URL>"
    fi
}

# Log visual
show_log() {
    echo -e "${BLUE}=== Histórico do Repositório ===${NC}"
    git log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit -10
}

# Tamanho do repositório
show_size() {
    echo -e "${BLUE}=== Tamanho do Repositório ===${NC}"
    
    echo "Tamanho total:"
    du -sh .git 2>/dev/null || echo "Erro ao calcular tamanho"
    
    echo ""
    echo "Arquivos grandes (>1MB):"
    find . -type f -size +1M -not -path "./.git/*" -not -path "./data/*" -exec ls -lh {} \; 2>/dev/null | head -10 || true
}

# Main
case "${1:-help}" in
    "status")
        git_status
        ;;
    "clean")
        git_clean
        ;;
    "backup")
        create_backup
        ;;
    "feature")
        create_feature "$2"
        ;;
    "fix")
        create_fix "$2"
        ;;
    "sync")
        sync_repo
        ;;
    "log")
        show_log
        ;;
    "size")
        show_size
        ;;
    "help"|*)
        show_help
        ;;
esac