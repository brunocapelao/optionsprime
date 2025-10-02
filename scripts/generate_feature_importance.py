#!/usr/bin/env python3
"""
Script para gerar arquivos de feature importance a partir dos modelos treinados.
Resolve a pendência de feature_importance_T*.csv files.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json

def extract_feature_importance(models_path, output_path, T_horizon):
    """
    Extrai feature importance dos modelos LightGBM e salva em CSV.
    
    Args:
        models_path: Path para o arquivo models_T*.joblib
        output_path: Path onde salvar o feature_importance_T*.csv
        T_horizon: Horizonte (42, 48, 54, 60)
    """
    print(f"📊 Extraindo feature importance para T={T_horizon}")
    
    try:
        # Carregar modelos
        models = joblib.load(models_path)
        print(f"   ✅ Modelos carregados: {len(models)} quantis")
        
        # Lista para armazenar importance de todos os quantis
        all_importances = []
        feature_names = None
        
        # Iterar pelos quantis
        for quantile, model in models.items():
            if hasattr(model, 'feature_importance'):
                # Para LightGBM Booster, usar feature_importance()
                importance = model.feature_importance()
                
                # Pegar feature names se ainda não temos
                if feature_names is None and hasattr(model, 'feature_name'):
                    feature_names = model.feature_name()
                elif feature_names is None:
                    # Gerar nomes genéricos se não disponível
                    feature_names = [f'feature_{i}' for i in range(len(importance))]
                
                # Criar DataFrame para este quantil
                df_quantile = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance,
                    'quantile': quantile,
                    'T_horizon': T_horizon
                })
                
                all_importances.append(df_quantile)
                print(f"   📈 Quantil {quantile}: {len(importance)} features")
        
        if all_importances:
            # Concatenar todos os quantis
            df_final = pd.concat(all_importances, ignore_index=True)
            
            # Calcular importance média por feature
            df_summary = df_final.groupby('feature')['importance'].agg([
                'mean', 'std', 'min', 'max'
            ]).reset_index()
            df_summary['T_horizon'] = T_horizon
            df_summary = df_summary.sort_values('mean', ascending=False)
            
            # Salvar CSV detalhado
            detailed_path = output_path.parent / f'feature_importance_T{T_horizon}_detailed.csv'
            df_final.to_csv(detailed_path, index=False)
            
            # Salvar CSV resumo (formato esperado pelo notebook)
            df_summary.to_csv(output_path, index=False)
            
            print(f"   💾 Salvo: {output_path}")
            print(f"   💾 Salvo: {detailed_path}")
            print(f"   📊 Top 5 features:")
            for idx, row in df_summary.head().iterrows():
                print(f"      {idx+1}. {row['feature']}: {row['mean']:.4f}")
            
            return True
        else:
            print(f"   ❌ Nenhum modelo com feature_importances_ encontrado")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro ao extrair feature importance: {e}")
        return False

def main():
    """Main function to generate all feature importance files."""
    print("🔧 GERANDO FEATURE IMPORTANCE FILES")
    print("=" * 50)
    
    # Diretórios
    preds_dir = Path("data/processed/preds")
    
    if not preds_dir.exists():
        print(f"❌ Diretório não encontrado: {preds_dir}")
        return
    
    # Ler horizontes do training_summary.json
    training_summary_path = preds_dir / "training_summary.json"
    if training_summary_path.exists():
        with open(training_summary_path, 'r') as f:
            training_summary = json.load(f)
            horizons = training_summary.get('trained_horizons', [42, 48, 54, 60])
    else:
        # Fallback: detectar pelos arquivos disponíveis
        model_files = list(preds_dir.glob("models_T*.joblib"))
        horizons = []
        for f in model_files:
            try:
                T = int(f.stem.split('_T')[1])
                horizons.append(T)
            except:
                continue
        horizons = sorted(horizons)
    
    print(f"📋 Horizontes detectados: {horizons}")
    
    success_count = 0
    total_count = len(horizons)
    
    # Processar cada horizonte
    for T in horizons:
        models_path = preds_dir / f"models_T{T}.joblib"
        output_path = preds_dir / f"feature_importance_T{T}.csv"
        
        if models_path.exists():
            success = extract_feature_importance(models_path, output_path, T)
            if success:
                success_count += 1
        else:
            print(f"❌ Arquivo não encontrado: {models_path}")
    
    print(f"\n🎯 RESUMO:")
    print(f"   ✅ Sucessos: {success_count}/{total_count}")
    print(f"   📁 Arquivos gerados em: {preds_dir}")
    
    if success_count == total_count:
        print("✅ TODOS OS FEATURE IMPORTANCE FILES GERADOS COM SUCESSO!")
        return True
    else:
        print("⚠️  Alguns arquivos falharam na geração")
        return False

if __name__ == "__main__":
    main()