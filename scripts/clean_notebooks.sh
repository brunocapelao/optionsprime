#!/bin/bash
# Script para limpar outputs de todos os notebooks

echo "🧹 Limpando outputs dos notebooks..."

# Contador
count=0

# Limpar todos os notebooks
for notebook in notebooks/*.ipynb; do
    if [ -f "$notebook" ]; then
        nbstripout "$notebook"
        count=$((count + 1))
        echo "  ✓ $(basename "$notebook")"
    fi
done

echo ""
echo "✅ Total de $count notebooks limpos!"
echo ""
echo "Dica: Para verificar mudanças antes de commitar:"
echo "  git diff --stat"
