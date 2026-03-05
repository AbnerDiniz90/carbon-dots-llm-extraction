import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_csv_files(directory='.', pattern='*.csv'):
    """
    Carrega todos os arquivos CSV do diretório especificado.
    Detecta automaticamente CSVs com linhas de cabeçalho extras (nested).
    Retorna um DataFrame consolidado com todos os dados.
    """
    csv_files = glob.glob(os.path.join(directory, pattern))
    
    if not csv_files:
        print(f"Nenhum arquivo CSV encontrado em '{directory}' com padrão '{pattern}'")
        return None
    
    print(f"Arquivos CSV encontrados: {len(csv_files)}")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    
    expected_columns = ['model', 'prompt_generated', 'Prompt_Length', 'Structure_Evaluation', 
                        'Scientific_Relevance', 'Constraints_Evaluation', 'Robustness_Evaluation']
    
    dfs = []
    
    for csv_file in csv_files:
        try:
            df = None
            skiprows_used = 0
            
            for skiprows in [0, 1, 2]:
                try:
                    temp_df = pd.read_csv(csv_file, skiprows=skiprows)
                    
                    temp_df = temp_df.loc[:, ~temp_df.columns.str.contains('^Unnamed')]
                    
                    temp_df.columns = temp_df.columns.str.strip()
                    
                    columns_lower = [col.lower() for col in temp_df.columns]
                    expected_lower = [col.lower() for col in expected_columns]
                    
                    matches = sum(1 for col in expected_lower if col in columns_lower)

                    if matches >= 4:
                        df = temp_df
                        skiprows_used = skiprows
                        break
                        
                except Exception as e:
                    continue
            
            if df is not None:
                df['source_file'] = os.path.basename(csv_file)
                dfs.append(df)
                
                format_info = "(formato padrão)" if skiprows_used == 0 else f"(pulou {skiprows_used} linha(s) de cabeçalho)"
                print(f"  Carregado: {os.path.basename(csv_file)} ({len(df)} registros) {format_info}")
            else:
                print(f"  Não foi possível identificar o formato de: {os.path.basename(csv_file)}")
                
        except Exception as e:
            print(f"  Erro ao carregar {csv_file}: {e}")
    
    if not dfs:
        print("Nenhum dado carregado dos arquivos CSV.")
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    combined_df.columns = combined_df.columns.str.strip()
    
    combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('^Unnamed')]
    
    return combined_df

def clean_and_prepare_data(df):
    """
    Limpa e prepara os dados para análise.
    """
    column_mapping = {
        'model': 'model',
        'Model': 'model',
        'prompt_generated': 'prompt_generated',
        'Prompt_Length': 'prompt_length',
        'prompt_length': 'prompt_length',
        'Structure_Evaluation': 'structure_evaluation',
        'Scientific_Relevance': 'scientific_relevance',
        'Constraints_Evaluation': 'constraints_evaluation',
        'Robustness_Evaluation': 'robustness_evaluation'
    }
    
    df = df.rename(columns=column_mapping)
    
    numeric_columns = ['prompt_length', 'scientific_relevance', 'constraints_evaluation', 'robustness_evaluation']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'model' in df.columns:
        df = df.dropna(subset=['model'])
        df['model'] = df['model'].astype(str).str.strip()
    
    return df

def calculate_model_statistics(df):
    """
    Calcula estatísticas (médias e máximos) por modelo.
    """
    averages = df.groupby('model').agg({
        'scientific_relevance': 'mean',
        'constraints_evaluation': 'mean',
        'robustness_evaluation': 'mean',
        'prompt_length': 'mean'
    }).reset_index()
    
    averages.columns = ['model', 'scientific_relevance_avg', 'constraints_evaluation_avg', 
                        'robustness_evaluation_avg', 'prompt_length_avg']
    
    max_values = df.groupby('model').agg({
        'scientific_relevance': 'max',
        'constraints_evaluation': 'max',
        'robustness_evaluation': 'max'
    }).reset_index()
    
    max_values.columns = ['model', 'scientific_relevance_max', 'constraints_evaluation_max', 
                          'robustness_evaluation_max']
    
    counts = df.groupby('model').size().reset_index(name='count')
    
    detailed_data = {}
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        detailed_data[model] = {
            'scientific_relevance': model_data['scientific_relevance'].dropna().tolist(),
            'constraints_evaluation': model_data['constraints_evaluation'].dropna().tolist(),
            'robustness_evaluation': model_data['robustness_evaluation'].dropna().tolist(),
            'prompt_length': model_data['prompt_length'].dropna().tolist()
        }
    
    return averages, max_values, counts, detailed_data

def calculate_structure_evaluation_stats(df):
    """
    Calcula estatísticas de STRUCTURE_EVALUATION por modelo.
    Retorna contagem de STRICT JSON vs outros valores.
    """
    if 'structure_evaluation' not in df.columns:
        print("Coluna 'structure_evaluation' não encontrada.")
        return None
    
    def is_strict_json(value):
        if pd.isna(value):
            return False
        value_upper = str(value).upper().strip()
        return 'STRICT JSON' in value_upper
    
    df['is_strict_json'] = df['structure_evaluation'].apply(is_strict_json)
    
    structure_stats = df.groupby('model').agg(
        total=('model', 'count'),
        strict_json_count=('is_strict_json', 'sum')
    ).reset_index()
    
    structure_stats['not_strict_json_count'] = structure_stats['total'] - structure_stats['strict_json_count']
    
    structure_stats['strict_json_pct'] = (structure_stats['strict_json_count'] / structure_stats['total']) * 100
    structure_stats['not_strict_json_pct'] = (structure_stats['not_strict_json_count'] / structure_stats['total']) * 100
    
    print("\n[DEBUG] Valores de Structure_Evaluation por modelo:")
    for model in df['model'].unique():
        values = df[df['model'] == model]['structure_evaluation'].unique()
        print(f"  {simplify_model_name(model)}: {list(values)}")
    
    return structure_stats

def find_and_save_best_prompts(df, output_filename='melhores_prompts.txt', max_score=3):
    """
    Encontra os prompts que obtiveram pontuação máxima em TODAS as categorias
    e salva em um arquivo .txt formatado.
    
    Args:
        df: DataFrame com os dados
        output_filename: Nome do arquivo de saída
        max_score: Pontuação máxima esperada (default: 3)
    
    Returns:
        DataFrame com os melhores prompts
    """
    print("\nBuscando prompts com pontuação máxima em todas as categorias...")

    best_prompts = df[
        (df['scientific_relevance'] == max_score) &
        (df['constraints_evaluation'] == max_score) &
        (df['robustness_evaluation'] == max_score)
    ].copy()
    
    num_best = len(best_prompts)
    total_prompts = len(df)
    
    print(f"   Encontrados: {num_best} de {total_prompts} prompts ({(num_best/total_prompts*100):.1f}%)")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RELATÓRIO: PROMPTS COM PONTUAÇÃO MÁXIMA EM TODAS AS CATEGORIAS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Critérios: Scientific_Relevance = {max_score}, Constraints_Evaluation = {max_score}, Robustness_Evaluation = {max_score}\n")
        f.write(f"Total de prompts analisados: {total_prompts}\n")
        f.write(f"Prompts com pontuação máxima: {num_best}\n")
        f.write("=" * 80 + "\n\n")
        
        if num_best == 0:
            f.write("Nenhum prompt atingiu a pontuação máxima em todas as categorias.\n\n")

            df['total_score'] = df['scientific_relevance'] + df['constraints_evaluation'] + df['robustness_evaluation']
            max_total = df['total_score'].max()
            near_best = df[df['total_score'] == max_total].copy()
            
            f.write(f"Prompts com maior pontuação total ({int(max_total)} pontos):\n")
            f.write("-" * 80 + "\n\n")
            
            for idx, row in near_best.iterrows():
                write_prompt_details(f, row, idx)
        else:
            models = best_prompts['model'].unique()
            
            f.write(f"Modelos com prompts de pontuação máxima: {len(models)}\n")
            for model in models:
                count = len(best_prompts[best_prompts['model'] == model])
                f.write(f"  • {simplify_model_name(model)}: {count} prompt(s)\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            for i, (idx, row) in enumerate(best_prompts.iterrows(), 1):
                f.write(f"PROMPT #{i}\n")
                write_prompt_details(f, row, idx)
    
    print(f"   Salvo em: {output_filename}")
    
    return best_prompts

def write_prompt_details(f, row, idx):
    """
    Escreve os detalhes de um prompt no arquivo.
    """
    f.write("-" * 80 + "\n")
    f.write(f"MODELO: {row.get('model', 'N/A')}\n")
    f.write(f"ARQUIVO FONTE: {row.get('source_file', 'N/A')}\n")
    f.write("-" * 80 + "\n\n")

    f.write("PONTUAÇÕES:\n")
    f.write(f"  • Scientific Relevance:    {int(row.get('scientific_relevance', 0))}\n")
    f.write(f"  • Constraints Evaluation:  {int(row.get('constraints_evaluation', 0))}\n")
    f.write(f"  • Robustness Evaluation:   {int(row.get('robustness_evaluation', 0))}\n")
    f.write(f"  • Structure Evaluation:    {row.get('structure_evaluation', 'N/A')}\n")
    f.write(f"  • Tamanho do Prompt:       {int(row.get('prompt_length', 0))} caracteres\n")
    f.write("\n")
    
    f.write("PROMPT GERADO:\n")
    f.write("-" * 40 + "\n")
    prompt_text = row.get('prompt_generated', 'N/A')
    if pd.notna(prompt_text):
        f.write(str(prompt_text))
    else:
        f.write("(Prompt não disponível)")
    f.write("\n" + "-" * 40 + "\n\n")
    
    f.write("JUSTIFICATIVAS:\n")
    
    reasoning_fields = [
        ('scientific_relevance_reasoning', 'Scientific Relevance'),
        ('constraints_evaluation_reasoning', 'Constraints Evaluation'),
        ('robustness_evaluation_reasoning', 'Robustness Evaluation'),
        ('structure_evaluation_reasoning', 'Structure Evaluation')
    ]
    
    for col_name, label in reasoning_fields:
        possible_cols = [c for c in row.index if 'reasoning' in c.lower() and label.lower().split()[0] in c.lower()]
        
        if possible_cols:
            value = row.get(possible_cols[0], 'N/A')
        else:
            alt_name = f"{label.replace(' ', '_')}_reasoning"
            value = row.get(alt_name, row.get(col_name, 'N/A'))
        
        if pd.notna(value) and str(value).strip():
            f.write(f"\n  [{label}]\n")
            f.write(f"  {value}\n")
    
    f.write("\n" + "=" * 80 + "\n\n")

def simplify_model_name(model_name):
    """
    Simplifica o nome do modelo para exibição nos gráficos.
    """
    name = str(model_name)
    name = name.replace(':free', '')
    name = name.replace(':latest', '')
    return name

def plot_metrics_average(averages):
    """
    Gráfico 1: Média das métricas por cada modelo.
    """
    modelos = averages['model'].tolist()
    modelos_short = [simplify_model_name(m) for m in modelos]
    
    scientific = averages['scientific_relevance_avg'].tolist()
    constraints = averages['constraints_evaluation_avg'].tolist()
    robustness = averages['robustness_evaluation_avg'].tolist()
    
    x = np.arange(len(modelos_short))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(18, 12), layout="constrained")
    
    bars1 = ax.bar(x - width, scientific, width, label='Scientific Relevance', color='#2ecc71')
    bars2 = ax.bar(x, constraints, width, label='Constraints Evaluation', color='#3498db')
    bars3 = ax.bar(x + width, robustness, width, label='Robustness Evaluation', color='#e74c3c')
    
    ax.set_xlabel('Modelos', fontsize=22, fontweight='bold')
    ax.set_ylabel('Média das Avaliações', fontsize=22, fontweight='bold')
    ax.set_title('Gráfico 1: Média das Métricas por Modelo', fontsize=22, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos_short, rotation=45, ha='right', fontsize=18)

    ax.set_xlim(-0.5, len(modelos_short) + 0.7)

    ax.legend(loc='upper right', fontsize=18)
    ax.set_ylim(0, 4)
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=18)
    
    plt.savefig('grafico1_media_metricas.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico 1 salvo como 'grafico1_media_metricas.png'")

def plot_prompt_length_average(averages):
    """
    Gráfico 2: Tamanho médio dos prompts para cada modelo.
    """
    modelos = averages['model'].tolist()
    modelos_short = [simplify_model_name(m) for m in modelos]
    
    lengths = averages['prompt_length_avg'].tolist()
    
    fig, ax = plt.subplots(figsize=(18, 12), layout="constrained")
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(modelos_short)))
    bars = ax.bar(modelos_short, lengths, color=colors, linewidth=0.5)
    
    ax.set_xlabel('Modelos', fontsize=22, fontweight='bold')
    ax.set_ylabel('Tamanho Médio do Prompt (caracteres)', fontsize=22, fontweight='bold')
    ax.set_title('Gráfico 2: Tamanho Médio dos Prompts por Modelo', fontsize=22, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=18)
    ax.grid(axis='y', alpha=0.3)
    
    max_length = max([l for l in lengths if not pd.isna(l)])
    ax.set_ylim(0, max_length * 1.30) 

    for bar, length in zip(bars, lengths):
        if not np.isnan(length):
            ax.annotate(f'{int(length):,}'.replace(',', '.'),
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=18)
    
    plt.savefig('grafico2_tamanho_prompts.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico 2 salvo como 'grafico2_tamanho_prompts.png'")

def plot_best_performance_bars(max_values):
    """
    Gráfico 3: Gráfico de barras mostrando o melhor desempenho de cada modelo.
    """
    modelos = max_values['model'].tolist()
    modelos_short = [simplify_model_name(m) for m in modelos]
    
    scientific = max_values['scientific_relevance_max'].tolist()
    constraints = max_values['constraints_evaluation_max'].tolist()
    robustness = max_values['robustness_evaluation_max'].tolist()
    
    x = np.arange(len(modelos_short))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(18, 12), layout="constrained")
    
    bars1 = ax.bar(x - width, scientific, width, label='Scientific Relevance', color='#2ecc71')
    bars2 = ax.bar(x, constraints, width, label='Constraints Evaluation', color='#3498db')
    bars3 = ax.bar(x + width, robustness, width, label='Robustness Evaluation', color='#e74c3c')
    
    ax.set_xlabel('Modelos', fontsize=22, fontweight='bold')
    ax.set_ylabel('Melhor Pontuação (Máximo)', fontsize=22, fontweight='bold')
    ax.set_title('Gráfico 3: Melhor Desempenho de cada Modelo', fontsize=22, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos_short, rotation=45, ha='right', fontsize=18)

    ax.set_xlim(-0.5, len(modelos_short) + 0.7)

    ax.legend(loc='upper right', fontsize=18)
    ax.set_ylim(0, 4)
    ax.set_yticks([0, 1, 2, 3])
    ax.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=18)
    
    plt.savefig('grafico3_melhor_desempenho.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico 3 salvo como 'grafico3_melhor_desempenho.png'")

def plot_structure_evaluation_comparison(structure_stats):
    """
    Gráfico 4: Comparação de STRICT JSON vs Outros por modelo (barras empilhadas).
    Mostra quantas respostas foram STRICT JSON e quantas foram diferentes.
    """
    if structure_stats is None:
        print("Não foi possível gerar o gráfico de Structure Evaluation.")
        return
    
    modelos = structure_stats['model'].tolist()
    modelos_short = [simplify_model_name(m) for m in modelos]
    
    strict_json = structure_stats['strict_json_count'].tolist()
    not_strict_json = structure_stats['not_strict_json_count'].tolist()
    total = structure_stats['total'].tolist()
    
    x = np.arange(len(modelos_short))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(18, 12), layout="constrained")
    
    bars1 = ax.bar(x, strict_json, width, label='STRICT JSON', color='#27ae60')
    bars2 = ax.bar(x, not_strict_json, width, bottom=strict_json, label='UNKNOWN ou VAGUE', color='#e74c3c')
    
    ax.set_xlabel('Modelos', fontsize=22, fontweight='bold')
    ax.set_ylabel('Quantidade de Respostas', fontsize=22, fontweight='bold')
    ax.set_title('Gráfico 4: Distribuição de Structure Evaluation por Modelo', 
                 fontsize=22, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos_short, rotation=45, ha='right', fontsize=18)

    ax.set_xlim(-0.5, len(modelos_short) + 0.7)

    ax.legend(loc='upper right', fontsize=18)
    ax.grid(axis='y', alpha=0.3)

    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        if height1 > 0:
            ax.annotate(f'{int(height1)}',
                       xy=(bar1.get_x() + bar1.get_width() / 2, height1 / 2),
                       ha='center', va='center', fontsize=18, color='white')
        
        height2 = bar2.get_height()
        if height2 > 0:
            ax.annotate(f'{int(height2)}',
                       xy=(bar2.get_x() + bar2.get_width() / 2, height1 + height2 / 2),
                       ha='center', va='center', fontsize=18, color='white')
        
        ax.annotate(f'Total: {int(total[i])}',
                   xy=(bar1.get_x() + bar1.get_width() / 2, height1 + height2),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=18)
    
    plt.savefig('grafico4_structure_evaluation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico 4 salvo como 'grafico4_structure_evaluation.png'")

def plot_structure_evaluation_percentage(structure_stats):
    """
    Gráfico 5: Porcentagem de respostas não-STRICT JSON por modelo.
    """
    if structure_stats is None:
        print("Não foi possível gerar o gráfico de porcentagem.")
        return
    
    modelos = structure_stats['model'].tolist()
    modelos_short = [simplify_model_name(m) for m in modelos]
    
    strict_json_pct = structure_stats['strict_json_pct'].tolist()
    not_strict_json_pct = structure_stats['not_strict_json_pct'].tolist()
    
    x = np.arange(len(modelos_short))
    width = 0.6
    
    fig, ax = plt.subplots(figsize=(18, 12), layout="constrained")
    
    bars1 = ax.bar(x, strict_json_pct, width, label='STRICT JSON', color='#27ae60')
    bars2 = ax.bar(x, not_strict_json_pct, width, bottom=strict_json_pct, 
                   label='UNKNOWN ou VAGUE', color='#e74c3c')
    
    ax.set_xlabel('Modelos', fontsize=22, fontweight='bold')
    ax.set_ylabel('Porcentagem (%)', fontsize=22, fontweight='bold')
    ax.set_title('Gráfico 5: Porcentagem de Structure Evaluation por Modelo', 
                 fontsize=22, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modelos_short, rotation=45, ha='right', fontsize=18)

    ax.set_xlim(-0.5, len(modelos_short) + 0.7)

    ax.legend(loc='upper right', fontsize=18)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        if height1 > 5: 
            ax.annotate(f'{height1:.1f}%',
                       xy=(bar1.get_x() + bar1.get_width() / 2, height1 / 2),
                       ha='center', va='center', fontsize=18, color='white')
        
        if height2 > 5:  
            ax.annotate(f'{height2:.1f}%',
                       xy=(bar2.get_x() + bar2.get_width() / 2, height1 + height2 / 2),
                       ha='center', va='center', fontsize=18, color='white')
    
    plt.savefig('grafico5_structure_evaluation_percentual.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico 5 salvo como 'grafico5_structure_evaluation_percentual.png'")

def print_detailed_report(averages, max_values, counts, detailed_data, structure_stats=None):
    """
    Imprime um relatório detalhado das estatísticas.
    """
    print("\n" + "="*70)
    print("                     RELATÓRIO DETALHADO")
    print("="*70)
    
    for i, row in averages.iterrows():
        model = row['model']
        model_short = simplify_model_name(model)
        count = counts[counts['model'] == model]['count'].values[0]
        max_row = max_values[max_values['model'] == model].iloc[0]
        
        print(f"\nModelo: {model_short}")
        print(f"   Nome completo: {model}")
        print(f"   Total de prompts analisados: {count}")
        print("-" * 50)
        
        print("   MÉDIAS:")
        print(f"     • Scientific Relevance:    {row['scientific_relevance_avg']:.2f}")
        print(f"     • Constraints Evaluation:  {row['constraints_evaluation_avg']:.2f}")
        print(f"     • Robustness Evaluation:   {row['robustness_evaluation_avg']:.2f}")
        print(f"     • Tamanho do Prompt:       {int(row['prompt_length_avg']):,} caracteres")
        
        print("   MELHOR DESEMPENHO (Máximo):")
        print(f"     • Scientific Relevance:    {int(max_row['scientific_relevance_max'])}")
        print(f"     • Constraints Evaluation:  {int(max_row['constraints_evaluation_max'])}")
        print(f"     • Robustness Evaluation:   {int(max_row['robustness_evaluation_max'])}")
        
        if structure_stats is not None:
            struct_row = structure_stats[structure_stats['model'] == model].iloc[0]
            print("   STRUCTURE EVALUATION:")
            print(f"     • STRICT JSON:       {int(struct_row['strict_json_count'])} ({struct_row['strict_json_pct']:.1f}%)")
            print(f"     • Outros:            {int(struct_row['not_strict_json_count'])} ({struct_row['not_strict_json_pct']:.1f}%)")
        
        if model in detailed_data:
            print("   VALORES INDIVIDUAIS:")
            print(f"     • Scientific:  {detailed_data[model]['scientific_relevance']}")
            print(f"     • Constraints: {detailed_data[model]['constraints_evaluation']}")
            print(f"     • Robustness:  {detailed_data[model]['robustness_evaluation']}")
    
    print("\n" + "="*70)

def main():
    """
    Função principal que executa todo o pipeline de análise.
    """

    csvPath = os.path.join(os.path.dirname(__file__), "prompts_evals")

    CSV_DIRECTORY = csvPath
    CSV_PATTERN = '*.csv'
    
    print("="*70)
    print("     ANÁLISE DE MÉTRICAS DE MODELOS - LEITURA DE CSV")
    print("="*70)
    print()
    
    print("Carregando arquivos CSV...")
    df = load_csv_files(CSV_DIRECTORY, CSV_PATTERN)
    
    if df is None or df.empty:
        print("\nNenhum dado encontrado. Verifique os arquivos CSV.")
        return
    
    print(f"\nTotal de registros carregados: {len(df)}")
    
    print("\nPreparando dados...")
    df = clean_and_prepare_data(df)
    print(f"   Registros após limpeza: {len(df)}")
    
    if 'model' in df.columns:
        modelos_unicos = df['model'].unique()
        print(f"\nModelos identificados ({len(modelos_unicos)}):")
        for m in modelos_unicos:
            count = len(df[df['model'] == m])
            print(f"   - {simplify_model_name(m)}: {count} registro(s)")
    
    print("\nCalculando estatísticas...")
    averages, max_values, counts, detailed_data = calculate_model_statistics(df)
    
    print("\nCalculando estatísticas de Structure Evaluation...")
    structure_stats = calculate_structure_evaluation_stats(df)

    best_prompts = find_and_save_best_prompts(df, output_filename='melhores_prompts.txt', max_score=3)
    
    print("\n" + "="*70)
    print("                    GERANDO GRÁFICOS")
    print("="*70)
    
    print("\n[1/5] Gerando gráfico de médias das métricas...")
    plot_metrics_average(averages)
    
    print("\n[2/5] Gerando gráfico de tamanho médio dos prompts...")
    plot_prompt_length_average(averages)
    
    print("\n[3/5] Gerando gráfico de melhor desempenho...")
    plot_best_performance_bars(max_values)
    
    print("\n[4/5] Gerando gráfico de Structure Evaluation (contagem)...")
    plot_structure_evaluation_comparison(structure_stats)
    
    print("\n[5/5] Gerando gráfico de Structure Evaluation (porcentagem)...")
    plot_structure_evaluation_percentage(structure_stats)
    
    print_detailed_report(averages, max_values, counts, detailed_data, structure_stats)
    
    print("   Arquivos salvos:")
    print("   - grafico1_media_metricas.png")
    print("   - grafico2_tamanho_prompts.png")
    print("   - grafico3_melhor_desempenho.png")
    print("   - grafico4_structure_evaluation.png")
    print("   - grafico5_structure_evaluation_percentual.png")

if __name__ == "__main__":
    main()