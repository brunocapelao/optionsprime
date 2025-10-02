#!/usr/bin/env python3
"""
📊 ANÁLISE TEXTUAL DOS RESULTADOS DO BACKTEST HISTÓRICO
Cria relatório detalhado sem dependências gráficas
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_backtest_results():
    """
    Análise completa dos resultados do backtest histórico
    """
    print("="*80)
    print("📊 ANÁLISE DETALHADA DO BACKTEST HISTÓRICO")
    print("Framework 02c - Validação de Modelos Quantílicos")
    print("="*80)
    
    # Carregar resultados
    results_file = 'data/processed/preds/historical_backtest_results.json'
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"✅ Resultados carregados de: {results_file}")
    except Exception as e:
        print(f"❌ Erro ao carregar resultados: {e}")
        return False
    
    config = results['config']
    fold_results = results['fold_results']
    gates_summary = results['gates_summary']
    
    print(f"\n🎯 CONFIGURAÇÃO DO BACKTEST:")
    print(f"   • Data de execução: {results['timestamp']}")
    print(f"   • Tempo de execução: {results['execution_time']:.2f}s")
    print(f"   • Número de folds: {len(fold_results)}")
    print(f"   • Modelos testados: {', '.join(config['models'])}")
    print(f"   • Horizontes: {config['horizons']}")
    print(f"   • Quantis: {config['quantiles']}")
    
    # Análise de performance por modelo
    print(f"\n📈 ANÁLISE DE PERFORMANCE POR MODELO:")
    print("=" * 50)
    
    model_stats = {}
    
    for model_name in config['models']:
        print(f"\n🤖 {model_name}:")
        
        # Coletar todas as métricas
        all_mae = []
        all_rmse = []
        all_coverage = []
        all_crps = []
        
        for fold in fold_results:
            for horizon in config['horizons']:
                horizon_str = str(horizon)  # Convert to string for JSON key
                metrics = fold['models'][model_name]['metrics'][horizon_str]
                all_mae.append(metrics['MAE'])
                all_rmse.append(metrics['RMSE'])
                all_coverage.append(metrics['Coverage_90'])
                all_crps.append(metrics['CRPS'])
        
        # Calcular estatísticas
        mae_stats = {
            'mean': np.mean(all_mae),
            'std': np.std(all_mae),
            'min': np.min(all_mae),
            'max': np.max(all_mae)
        }
        
        rmse_stats = {
            'mean': np.mean(all_rmse),
            'std': np.std(all_rmse),
            'min': np.min(all_rmse),
            'max': np.max(all_rmse)
        }
        
        coverage_stats = {
            'mean': np.mean(all_coverage),
            'std': np.std(all_coverage),
            'min': np.min(all_coverage),
            'max': np.max(all_coverage)
        }
        
        crps_stats = {
            'mean': np.mean(all_crps),
            'std': np.std(all_crps),
            'min': np.min(all_crps),
            'max': np.max(all_crps)
        }
        
        model_stats[model_name] = {
            'MAE': mae_stats,
            'RMSE': rmse_stats,
            'Coverage': coverage_stats,
            'CRPS': crps_stats
        }
        
        print(f"   📊 MAE:")
        print(f"      • Média: {mae_stats['mean']:.4f} ± {mae_stats['std']:.4f}")
        print(f"      • Range: [{mae_stats['min']:.4f}, {mae_stats['max']:.4f}]")
        
        print(f"   📊 RMSE:")
        print(f"      • Média: {rmse_stats['mean']:.4f} ± {rmse_stats['std']:.4f}")
        print(f"      • Range: [{rmse_stats['min']:.4f}, {rmse_stats['max']:.4f}]")
        
        print(f"   📊 Coverage 90%:")
        print(f"      • Média: {coverage_stats['mean']:.3f} ± {coverage_stats['std']:.3f}")
        print(f"      • Range: [{coverage_stats['min']:.3f}, {coverage_stats['max']:.3f}]")
        print(f"      • Desvio do target (90%): {abs(coverage_stats['mean'] - 0.90):.3f}")
        
        print(f"   📊 CRPS:")
        print(f"      • Média: {crps_stats['mean']:.4f} ± {crps_stats['std']:.4f}")
        print(f"      • Range: [{crps_stats['min']:.4f}, {crps_stats['max']:.4f}]")
        
        # Gates summary
        gates_info = gates_summary[model_name]
        print(f"   🚪 GATES:")
        print(f"      • Aprovados: {gates_info['total_passed']}/{gates_info['total_gates']}")
        print(f"      • Taxa: {gates_info['approval_rate']:.1%}")
        print(f"      • Decisão: {gates_info['final_decision']}")
    
    # Comparação entre modelos
    if len(config['models']) >= 2:
        print(f"\n🏆 COMPARAÇÃO ENTRE MODELOS:")
        print("=" * 40)
        
        model1, model2 = config['models'][0], config['models'][1]
        
        # MAE comparison
        mae1 = model_stats[model1]['MAE']['mean']
        mae2 = model_stats[model2]['MAE']['mean']
        mae_improvement = ((mae2 - mae1) / mae2 * 100) if mae1 < mae2 else ((mae1 - mae2) / mae1 * 100)
        mae_winner = model1 if mae1 < mae2 else model2
        
        print(f"📊 MAE Comparison:")
        print(f"   • {model1}: {mae1:.4f}")
        print(f"   • {model2}: {mae2:.4f}")
        print(f"   • Vencedor: {mae_winner} ({mae_improvement:.1f}% melhor)")
        
        # RMSE comparison
        rmse1 = model_stats[model1]['RMSE']['mean']
        rmse2 = model_stats[model2]['RMSE']['mean']
        rmse_improvement = ((rmse2 - rmse1) / rmse2 * 100) if rmse1 < rmse2 else ((rmse1 - rmse2) / rmse1 * 100)
        rmse_winner = model1 if rmse1 < rmse2 else model2
        
        print(f"📊 RMSE Comparison:")
        print(f"   • {model1}: {rmse1:.4f}")
        print(f"   • {model2}: {rmse2:.4f}")
        print(f"   • Vencedor: {rmse_winner} ({rmse_improvement:.1f}% melhor)")
        
        # Coverage comparison
        cov1 = model_stats[model1]['Coverage']['mean']
        cov2 = model_stats[model2]['Coverage']['mean']
        cov1_error = abs(cov1 - 0.90)
        cov2_error = abs(cov2 - 0.90)
        cov_winner = model1 if cov1_error < cov2_error else model2
        
        print(f"📊 Coverage Comparison:")
        print(f"   • {model1}: {cov1:.3f} (erro: {cov1_error:.3f})")
        print(f"   • {model2}: {cov2:.3f} (erro: {cov2_error:.3f})")
        print(f"   • Melhor calibrado: {cov_winner}")
        
        # Gates comparison
        gates1 = gates_summary[model1]['approval_rate']
        gates2 = gates_summary[model2]['approval_rate']
        gates_winner = model1 if gates1 > gates2 else model2
        
        print(f"🚪 Gates Comparison:")
        print(f"   • {model1}: {gates1:.1%}")
        print(f"   • {model2}: {gates2:.1%}")
        print(f"   • Melhor aprovação: {gates_winner}")
    
    # Análise por horizonte
    print(f"\n📈 ANÁLISE POR HORIZONTE:")
    print("=" * 35)
    
    for horizon in config['horizons']:
        print(f"\n⏰ Horizonte {horizon}h:")
        
        for model_name in config['models']:
            # Coletar métricas deste horizonte
            horizon_mae = []
            horizon_coverage = []
            
            for fold in fold_results:
                horizon_str = str(horizon)  # Convert to string for JSON key
                metrics = fold['models'][model_name]['metrics'][horizon_str]
                horizon_mae.append(metrics['MAE'])
                horizon_coverage.append(metrics['Coverage_90'])
            
            mae_avg = np.mean(horizon_mae)
            cov_avg = np.mean(horizon_coverage)
            
            print(f"   {model_name}: MAE={mae_avg:.4f}, Coverage={cov_avg:.3f}")
    
    # Análise por fold
    print(f"\n🔄 ANÁLISE POR FOLD:")
    print("=" * 25)
    
    for fold in fold_results:
        print(f"\nFold {fold['fold_id']}:")
        print(f"   📚 Treino: obs {fold['train_period'][0]} → {fold['train_period'][1]}")
        print(f"   🧪 Teste: obs {fold['test_period'][0]} → {fold['test_period'][1]}")
        
        for model_name in config['models']:
            gates_info = fold['gates'][model_name]
            status_icon = "✅" if gates_info['decision'] == "GO" else "🟡" if gates_info['decision'] == "CONDITIONAL" else "❌"
            
            print(f"   {status_icon} {model_name}: {gates_info['gates_passed']}/{gates_info['gates_total']} ({gates_info['approval_rate']:.1%}) → {gates_info['decision']}")
    
    # Resumo dos Gates
    print(f"\n🚪 RESUMO FINAL DOS GATES:")
    print("=" * 30)
    
    for model_name, gates_info in gates_summary.items():
        approval_rate = gates_info['approval_rate']
        decision = gates_info['final_decision']
        
        if decision == 'GO':
            status_icon = "✅"
            status_color = "GREEN"
        elif decision == 'CONDITIONAL':
            status_icon = "🟡"
            status_color = "YELLOW"
        else:
            status_icon = "❌"
            status_color = "RED"
        
        print(f"{status_icon} {model_name}:")
        print(f"   • Gates aprovados: {gates_info['total_passed']}/{gates_info['total_gates']}")
        print(f"   • Taxa de aprovação: {approval_rate:.1%}")
        print(f"   • Status: {status_color} - {decision}")
        
        # Interpretação do status
        if decision == 'GO':
            print(f"   • Interpretação: ✅ Aprovado para produção")
        elif decision == 'CONDITIONAL':
            print(f"   • Interpretação: 🟡 Aprovação condicional - monitorar de perto")
        else:
            print(f"   • Interpretação: ❌ Necessita melhorias antes da produção")
    
    # Recomendação final
    print(f"\n🎯 RECOMENDAÇÃO FINAL:")
    print("=" * 25)
    
    best_model = max(gates_summary.keys(), 
                    key=lambda x: gates_summary[x]['approval_rate'])
    best_rate = gates_summary[best_model]['approval_rate']
    best_decision = gates_summary[best_model]['final_decision']
    
    print(f"🏆 Modelo recomendado: {best_model}")
    print(f"📊 Taxa de aprovação: {best_rate:.1%}")
    print(f"🚀 Status final: {best_decision}")
    
    if best_decision == 'GO':
        print(f"\n✅ APROVADO PARA PRODUÇÃO")
        print(f"📋 Próximos passos recomendados:")
        print(f"   1. 📊 Implementar sistema de monitoramento em tempo real")
        print(f"   2. 🚨 Configurar alertas de degradação de performance")
        print(f"   3. 📈 Executar backtest em período mais longo (6+ meses)")
        print(f"   4. 📋 Preparar documentação técnica para deploy")
        print(f"   5. 🔄 Estabelecer ciclo de retreinamento periódico")
        print(f"   6. 🎯 Definir KPIs de monitoramento em produção")
        
    elif best_decision == 'CONDITIONAL':
        print(f"\n🟡 APROVAÇÃO CONDICIONAL")
        print(f"📋 Ações recomendadas antes do deploy:")
        print(f"   1. 🔍 Revisar thresholds dos gates que falharam")
        print(f"   2. 📊 Aumentar frequência de monitoramento")
        print(f"   3. 🎯 Implementar alertas mais sensíveis")
        print(f"   4. 📈 Validar performance em dados mais recentes")
        print(f"   5. 🔧 Considerar ajustes finos nos hiperparâmetros")
        print(f"   6. 📋 Plano de contingência em caso de degradação")
        
    else:
        print(f"\n❌ NECESSITA MELHORIAS SIGNIFICATIVAS")
        print(f"📋 Ações obrigatórias antes de considerar produção:")
        print(f"   1. 🔄 Retreinar modelo com dados mais recentes/extensos")
        print(f"   2. 🧪 Revisar e melhorar engenharia de features")
        print(f"   3. ⚙️  Otimizar hiperparâmetros com busca mais ampla")
        print(f"   4. 🎯 Validar qualidade e consistência dos dados")
        print(f"   5. 📊 Considerar arquiteturas de modelo alternativas")
        print(f"   6. 🔍 Analisar casos de falha específicos")
    
    # Insights adicionais
    print(f"\n💡 INSIGHTS ADICIONAIS:")
    print("=" * 25)
    
    # Análise de consistência
    consistency_scores = {}
    for model_name in config['models']:
        fold_rates = [fold['gates'][model_name]['approval_rate'] for fold in fold_results]
        consistency_scores[model_name] = {
            'mean': np.mean(fold_rates),
            'std': np.std(fold_rates),
            'range': max(fold_rates) - min(fold_rates)
        }
    
    print(f"📊 Consistência entre folds:")
    for model_name, scores in consistency_scores.items():
        print(f"   {model_name}:")
        print(f"      • Desvio padrão: {scores['std']:.3f}")
        print(f"      • Range: {scores['range']:.3f}")
        if scores['std'] < 0.1:
            print(f"      • Avaliação: ✅ Muito consistente")
        elif scores['std'] < 0.2:
            print(f"      • Avaliação: 🟡 Moderadamente consistente")
        else:
            print(f"      • Avaliação: ❌ Inconsistente - investigar")
    
    # Performance vs Horizonte
    print(f"\n📈 Tendências por horizonte:")
    for model_name in config['models']:
        mae_by_horizon = {}
        for horizon in config['horizons']:
            mae_values = []
            for fold in fold_results:
                horizon_str = str(horizon)  # Convert to string for JSON key
                mae_values.append(fold['models'][model_name]['metrics'][horizon_str]['MAE'])
            mae_by_horizon[horizon] = np.mean(mae_values)
        
        print(f"   {model_name}:")
        trend = "crescente" if mae_by_horizon[60] > mae_by_horizon[42] else "decrescente"
        print(f"      • Tendência MAE: {trend}")
        print(f"      • H42: {mae_by_horizon[42]:.4f} → H60: {mae_by_horizon[60]:.4f}")
    
    # Salvar resumo em arquivo
    print(f"\n💾 Salvando resumo executivo...")
    
    summary_report = []
    summary_report.append("RESUMO EXECUTIVO - BACKTEST HISTÓRICO")
    summary_report.append("=" * 50)
    summary_report.append(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_report.append(f"Framework: 02c")
    summary_report.append("")
    summary_report.append("MODELO RECOMENDADO:")
    summary_report.append(f"• Nome: {best_model}")
    summary_report.append(f"• Taxa de aprovação: {best_rate:.1%}")
    summary_report.append(f"• Decisão: {best_decision}")
    summary_report.append("")
    summary_report.append("MÉTRICAS PRINCIPAIS:")
    for model_name in config['models']:
        mae_avg = model_stats[model_name]['MAE']['mean']
        cov_avg = model_stats[model_name]['Coverage']['mean']
        summary_report.append(f"• {model_name}: MAE={mae_avg:.4f}, Coverage={cov_avg:.3f}")
    summary_report.append("")
    summary_report.append("STATUS PARA PRODUÇÃO:")
    if best_decision == 'GO':
        summary_report.append("✅ APROVADO - Pronto para deploy")
    elif best_decision == 'CONDITIONAL':
        summary_report.append("🟡 CONDICIONAL - Deploy com monitoramento intensivo")
    else:
        summary_report.append("❌ REPROVADO - Necessita melhorias")
    
    try:
        with open('data/processed/preds/executive_summary.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_report))
        print("✅ Resumo executivo salvo em: data/processed/preds/executive_summary.txt")
    except Exception as e:
        print(f"⚠️  Erro ao salvar resumo: {e}")
    
    print(f"\n🎯 ANÁLISE COMPLETA FINALIZADA!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    print("📊 Iniciando análise detalhada dos resultados...")
    success = analyze_backtest_results()
    
    if success:
        print("✅ Análise concluída com sucesso!")
    else:
        print("❌ Erro na análise dos resultados")