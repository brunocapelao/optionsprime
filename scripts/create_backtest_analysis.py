#!/usr/bin/env python3
"""
📊 ANÁLISE VISUAL DOS RESULTADOS DO BACKTEST HISTÓRICO
Cria gráficos e dashboard dos resultados do framework 02c
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def create_backtest_dashboard():
    """
    Cria dashboard visual completo dos resultados do backtest
    """
    print("="*60)
    print("📊 CRIANDO DASHBOARD VISUAL DO BACKTEST")
    print("="*60)
    
    # Carregar resultados
    results_file = 'data/processed/preds/historical_backtest_results.json'
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"✅ Resultados carregados de: {results_file}")
    except Exception as e:
        print(f"❌ Erro ao carregar resultados: {e}")
        return False
    
    # Extrair dados para visualização
    config = results['config']
    fold_results = results['fold_results']
    gates_summary = results['gates_summary']
    
    # Configurar estilo dos gráficos
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('🎯 Backtest Histórico - Dashboard Executivo\nFramework 02c', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Performance MAE por Fold e Modelo
    ax1 = plt.subplot(3, 3, 1)
    
    fold_numbers = []
    mae_data = {model: [] for model in config['models']}
    
    for fold in fold_results:
        fold_numbers.append(fold['fold_id'])
        for model_name in config['models']:
            # Calcular MAE médio do fold
            mae_values = [fold['models'][model_name]['metrics'][h]['MAE'] 
                         for h in config['horizons']]
            mae_data[model_name].append(np.mean(mae_values))
    
    for model_name, mae_vals in mae_data.items():
        ax1.plot(fold_numbers, mae_vals, marker='o', linewidth=3, 
                markersize=8, label=model_name)
    
    ax1.set_title('📈 MAE por Fold', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('MAE')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Coverage por Horizonte
    ax2 = plt.subplot(3, 3, 2)
    
    horizons = config['horizons']
    coverage_data = {model: [] for model in config['models']}
    
    for model_name in config['models']:
        for horizon in horizons:
            coverage_values = [fold['models'][model_name]['metrics'][horizon]['Coverage_90'] 
                             for fold in fold_results]
            coverage_data[model_name].append(np.mean(coverage_values))
    
    x_pos = np.arange(len(horizons))
    width = 0.35
    
    for i, (model_name, cov_vals) in enumerate(coverage_data.items()):
        ax2.bar(x_pos + i*width, cov_vals, width, label=model_name, alpha=0.8)
    
    ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
    ax2.set_title('📊 Coverage 90% por Horizonte', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Horizonte (horas)')
    ax2.set_ylabel('Coverage')
    ax2.set_xticks(x_pos + width/2)
    ax2.set_xticklabels(horizons)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.7, 1.0)
    
    # 3. Gates Approval Rate
    ax3 = plt.subplot(3, 3, 3)
    
    models = list(gates_summary.keys())
    approval_rates = [gates_summary[model]['approval_rate'] for model in models]
    
    # Cores baseadas na aprovação
    colors = ['green' if rate >= 0.75 else 'orange' if rate >= 0.60 else 'red' 
              for rate in approval_rates]
    
    bars = ax3.bar(models, approval_rates, color=colors, alpha=0.7)
    ax3.axhline(y=0.75, color='green', linestyle='--', alpha=0.7, label='GO (75%)')
    ax3.axhline(y=0.60, color='orange', linestyle='--', alpha=0.7, label='CONDITIONAL (60%)')
    
    # Adicionar valores nas barras
    for bar, rate in zip(bars, approval_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_title('🚪 Taxa de Aprovação nos Gates', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Taxa de Aprovação')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. RMSE Comparison
    ax4 = plt.subplot(3, 3, 4)
    
    rmse_data = {model: [] for model in config['models']}
    
    for model_name in config['models']:
        rmse_values = []
        for fold in fold_results:
            for horizon in config['horizons']:
                rmse_values.append(fold['models'][model_name]['metrics'][horizon]['RMSE'])
        rmse_data[model_name] = rmse_values
    
    # Box plot para RMSE
    data_for_boxplot = [rmse_data[model] for model in config['models']]
    box_plot = ax4.boxplot(data_for_boxplot, labels=config['models'], patch_artist=True)
    
    # Colorir as caixas
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_title('📊 Distribuição do RMSE', fontsize=14, fontweight='bold')
    ax4.set_ylabel('RMSE')
    ax4.grid(True, alpha=0.3)
    
    # 5. Performance por Horizonte (Heatmap)
    ax5 = plt.subplot(3, 3, 5)
    
    # Criar matriz de performance (MAE)
    perf_matrix = []
    for model_name in config['models']:
        model_row = []
        for horizon in config['horizons']:
            mae_values = [fold['models'][model_name]['metrics'][horizon]['MAE'] 
                         for fold in fold_results]
            model_row.append(np.mean(mae_values))
        perf_matrix.append(model_row)
    
    perf_df = pd.DataFrame(perf_matrix, 
                          index=config['models'], 
                          columns=[f'H{h}' for h in config['horizons']])
    
    sns.heatmap(perf_df, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                ax=ax5, cbar_kws={'label': 'MAE'})
    ax5.set_title('🎯 MAE por Modelo e Horizonte', fontsize=14, fontweight='bold')
    
    # 6. Gates Passed por Fold
    ax6 = plt.subplot(3, 3, 6)
    
    fold_gates = {model: [] for model in config['models']}
    
    for fold in fold_results:
        for model_name in config['models']:
            gates_info = fold['gates'][model_name]
            approval_rate = gates_info['approval_rate']
            fold_gates[model_name].append(approval_rate)
    
    for model_name, rates in fold_gates.items():
        ax6.plot(fold_numbers, rates, marker='s', linewidth=3, 
                markersize=8, label=model_name)
    
    ax6.axhline(y=0.75, color='green', linestyle='--', alpha=0.7, label='GO Threshold')
    ax6.set_title('🚪 Gates por Fold', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Fold')
    ax6.set_ylabel('Taxa de Aprovação')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim(0, 1)
    
    # 7. Coverage Distribution
    ax7 = plt.subplot(3, 3, 7)
    
    coverage_data_full = {model: [] for model in config['models']}
    
    for model_name in config['models']:
        for fold in fold_results:
            for horizon in config['horizons']:
                coverage_data_full[model_name].append(
                    fold['models'][model_name]['metrics'][horizon]['Coverage_90'])
    
    # Histograma de coverage
    for model_name, cov_vals in coverage_data_full.items():
        ax7.hist(cov_vals, alpha=0.6, label=model_name, bins=10, density=True)
    
    ax7.axvline(x=0.9, color='red', linestyle='--', alpha=0.7, label='Target (90%)')
    ax7.set_title('📊 Distribuição do Coverage', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Coverage 90%')
    ax7.set_ylabel('Densidade')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Performance Summary (Radar Chart)
    ax8 = plt.subplot(3, 3, 8, projection='polar')
    
    # Métricas para radar chart
    metrics = ['MAE', 'RMSE', 'Coverage', 'Gates']
    
    for model_name in config['models']:
        # Calcular valores médios (normalizar para 0-1)
        mae_avg = np.mean([np.mean([fold['models'][model_name]['metrics'][h]['MAE'] 
                                  for h in config['horizons']]) for fold in fold_results])
        rmse_avg = np.mean([np.mean([fold['models'][model_name]['metrics'][h]['RMSE'] 
                                   for h in config['horizons']]) for fold in fold_results])
        cov_avg = np.mean([np.mean([fold['models'][model_name]['metrics'][h]['Coverage_90'] 
                                  for h in config['horizons']]) for fold in fold_results])
        gates_avg = gates_summary[model_name]['approval_rate']
        
        # Normalizar (inverter MAE e RMSE - menor é melhor)
        mae_norm = max(0, 1 - mae_avg / 0.05)  # Normalizar para 0-1
        rmse_norm = max(0, 1 - rmse_avg / 0.08)
        cov_norm = cov_avg  # Já está em 0-1
        gates_norm = gates_avg  # Já está em 0-1
        
        values = [mae_norm, rmse_norm, cov_norm, gates_norm]
        values += values[:1]  # Completar o círculo
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        ax8.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax8.fill(angles, values, alpha=0.25)
    
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(metrics)
    ax8.set_ylim(0, 1)
    ax8.set_title('🎯 Performance Radar', fontsize=14, fontweight='bold')
    ax8.legend()
    ax8.grid(True)
    
    # 9. Executive Summary (Text)
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Preparar texto do resumo
    best_model = max(gates_summary.keys(), 
                    key=lambda x: gates_summary[x]['approval_rate'])
    best_rate = gates_summary[best_model]['approval_rate']
    best_decision = gates_summary[best_model]['final_decision']
    
    # Calcular improvement
    if len(config['models']) >= 2:
        model1, model2 = config['models'][0], config['models'][1]
        rate1 = gates_summary[model1]['approval_rate']
        rate2 = gates_summary[model2]['approval_rate']
        
        if rate1 > rate2:
            improvement = ((rate1 - rate2) / rate2) * 100 if rate2 > 0 else 0
            winner = model1
        else:
            improvement = ((rate2 - rate1) / rate1) * 100 if rate1 > 0 else 0
            winner = model2
    
    summary_text = f"""
🎯 RESUMO EXECUTIVO

📊 Backtest Histórico Executado:
   • {len(fold_results)} folds analisados
   • {len(config['models'])} modelos comparados
   • {len(config['horizons'])} horizontes testados
   • Tempo total: {results['execution_time']:.2f}s

🏆 Modelo Recomendado:
   • {best_model}
   • Taxa de aprovação: {best_rate:.1%}
   • Status: {best_decision}

📈 Performance:
   • Melhoria: {improvement:.1f}%
   • Melhor modelo: {winner}

🚀 Status: {'✅ APROVADO' if best_decision == 'GO' else '🟡 CONDICIONAL' if best_decision == 'CONDITIONAL' else '❌ REPROVADO'}
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar gráfico
    output_file = 'data/processed/preds/backtest_dashboard.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"💾 Dashboard salvo em: {output_file}")
    
    # Mostrar gráfico
    plt.show()
    
    return True

def create_performance_report():
    """
    Cria relatório detalhado de performance
    """
    print("\n📋 CRIANDO RELATÓRIO DE PERFORMANCE...")
    
    # Carregar resultados
    results_file = 'data/processed/preds/historical_backtest_results.json'
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"❌ Erro ao carregar resultados: {e}")
        return False
    
    config = results['config']
    fold_results = results['fold_results']
    gates_summary = results['gates_summary']
    
    # Criar relatório
    report = []
    report.append("="*80)
    report.append("📊 RELATÓRIO DETALHADO DO BACKTEST HISTÓRICO")
    report.append("Framework 02c - Validação de Modelos Quantílicos")
    report.append("="*80)
    report.append("")
    
    # Configuração
    report.append("🎯 CONFIGURAÇÃO DO BACKTEST:")
    report.append(f"   • Data de execução: {results['timestamp']}")
    report.append(f"   • Tempo de execução: {results['execution_time']:.2f}s")
    report.append(f"   • Número de folds: {len(fold_results)}")
    report.append(f"   • Modelos testados: {', '.join(config['models'])}")
    report.append(f"   • Horizontes: {config['horizons']}")
    report.append(f"   • Quantis: {config['quantiles']}")
    report.append("")
    
    # Performance por modelo
    report.append("📈 PERFORMANCE POR MODELO:")
    report.append("")
    
    for model_name in config['models']:
        report.append(f"🤖 {model_name}:")
        
        # Coletar métricas
        all_mae = []
        all_rmse = []
        all_coverage = []
        
        for fold in fold_results:
            for horizon in config['horizons']:
                metrics = fold['models'][model_name]['metrics'][horizon]
                all_mae.append(metrics['MAE'])
                all_rmse.append(metrics['RMSE'])
                all_coverage.append(metrics['Coverage_90'])
        
        # Estatísticas
        report.append(f"   📊 MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
        report.append(f"   📊 RMSE: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")
        report.append(f"   📊 Coverage: {np.mean(all_coverage):.3f} ± {np.std(all_coverage):.3f}")
        
        # Gates
        gates_info = gates_summary[model_name]
        report.append(f"   🚪 Gates: {gates_info['total_passed']}/{gates_info['total_gates']} ({gates_info['approval_rate']:.1%})")
        report.append(f"   🎯 Decisão: {gates_info['final_decision']}")
        report.append("")
    
    # Análise por fold
    report.append("🔄 ANÁLISE POR FOLD:")
    report.append("")
    
    for fold in fold_results:
        report.append(f"Fold {fold['fold_id']}:")
        report.append(f"   📚 Período treino: {fold['train_period'][0]} → {fold['train_period'][1]}")
        report.append(f"   🧪 Período teste: {fold['test_period'][0]} → {fold['test_period'][1]}")
        
        for model_name in config['models']:
            gates_info = fold['gates'][model_name]
            report.append(f"   {model_name}: {gates_info['gates_passed']}/{gates_info['gates_total']} ({gates_info['approval_rate']:.1%}) → {gates_info['decision']}")
        report.append("")
    
    # Recomendações
    best_model = max(gates_summary.keys(), 
                    key=lambda x: gates_summary[x]['approval_rate'])
    best_rate = gates_summary[best_model]['approval_rate']
    best_decision = gates_summary[best_model]['final_decision']
    
    report.append("🎯 RECOMENDAÇÕES:")
    report.append(f"   🏆 Melhor modelo: {best_model}")
    report.append(f"   📊 Taxa de aprovação: {best_rate:.1%}")
    report.append(f"   🚀 Status: {best_decision}")
    report.append("")
    
    if best_decision == 'GO':
        report.append("   ✅ APROVADO PARA PRODUÇÃO")
        report.append("   📋 Próximos passos:")
        report.append("      • Implementar monitoramento em tempo real")
        report.append("      • Configurar alertas de performance")
        report.append("      • Executar backtest em período mais longo")
        report.append("      • Preparar documentação para deploy")
    elif best_decision == 'CONDITIONAL':
        report.append("   🟡 APROVAÇÃO CONDICIONAL")
        report.append("   📋 Ações recomendadas:")
        report.append("      • Revisar thresholds dos gates críticos")
        report.append("      • Aumentar frequência de monitoramento")
        report.append("      • Considerar ajustes finos no modelo")
        report.append("      • Validar em dados mais recentes")
    else:
        report.append("   ❌ NECESSITA MELHORIAS")
        report.append("   📋 Ações obrigatórias:")
        report.append("      • Retreinar modelo com dados adicionais")
        report.append("      • Revisar engenharia de features")
        report.append("      • Ajustar hiperparâmetros")
        report.append("      • Validar qualidade dos dados")
    
    report.append("")
    report.append("="*80)
    
    # Salvar relatório
    report_text = "\n".join(report)
    
    output_file = 'data/processed/preds/backtest_report.txt'
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"📋 Relatório salvo em: {output_file}")
    except Exception as e:
        print(f"❌ Erro ao salvar relatório: {e}")
    
    # Imprimir resumo
    print("\n" + report_text)
    
    return True

if __name__ == "__main__":
    print("📊 Criando análise visual dos resultados do backtest...")
    
    # Criar dashboard
    dashboard_success = create_backtest_dashboard()
    
    if dashboard_success:
        print("✅ Dashboard criado com sucesso!")
        
        # Criar relatório
        report_success = create_performance_report()
        
        if report_success:
            print("✅ Relatório de performance criado!")
            print("\n🎯 ANÁLISE COMPLETA FINALIZADA!")
        else:
            print("⚠️  Erro na criação do relatório")
    else:
        print("❌ Erro na criação do dashboard")