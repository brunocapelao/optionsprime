#!/bin/bash
# 🧹 Limpeza automática do repositório

echo "🚀 Iniciando limpeza do repositório..."

# Remover cache do Python
echo "🗑️  Removendo cache Python..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Remover arquivos temporários do sistema
echo "🗑️  Removendo arquivos temporários..."
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "*.tmp" -delete 2>/dev/null || true
find . -name "*.log" -delete 2>/dev/null || true

# Remover arquivos de debug/teste temporários
echo "🗑️  Removendo arquivos de debug..."
rm -f debug_*.py test_quick*.py final_*.py temp_*.py 2>/dev/null || true

# Remover checkpoints do Jupyter
echo "🗑️  Removendo checkpoints Jupyter..."
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null || true

echo "✅ Limpeza concluída!"
echo "📊 Tamanho atual do projeto:"
du -sh .
echo ""
echo "🎯 Estrutura limpa:"
tree -I '__pycache__|*.pyc|.DS_Store|.ipynb_checkpoints' -L 2