#!/bin/bash
# Git Best Practices Setup Script for ML Projects

echo "🚀 Configurando melhores práticas de Git para ML..."
echo ""

# 1. Instalar ferramentas
echo "📦 Instalando ferramentas..."
pip install -q nbstripout pre-commit black isort flake8

# 2. Configurar nbstripout
echo "🧹 Configurando nbstripout..."
nbstripout --install

# 3. Configurar pre-commit
echo "🪝 Instalando pre-commit hooks..."
pre-commit install

# 4. Limpar notebooks existentes (opcional)
read -p "🤔 Deseja limpar outputs dos notebooks existentes? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧽 Limpando notebooks..."
    for notebook in notebooks/*.ipynb; do
        if [ -f "$notebook" ]; then
            nbstripout "$notebook"
            echo "  ✓ $(basename "$notebook")"
        fi
    done
    echo "✅ Notebooks limpos!"
fi

echo ""
echo "✅ Setup completo!"
echo ""
echo "📋 Próximos passos:"
echo "  1. Revise as mudanças: git status"
echo "  2. Adicione os arquivos: git add ."
echo "  3. Commit: git commit -m 'chore: setup Git best practices for ML'"
echo ""
echo "📚 Leia CONTRIBUTING.md para mais informações!"
