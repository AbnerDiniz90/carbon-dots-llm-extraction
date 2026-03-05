import json
import glob
import os

# Nome do arquivo final que será criado
arquivo_saida = 'dataset_final.jsonl'

goldenDatasetPath = os.path.join(os.path.dirname(__file__), "GoldenDataset")

# Encontra todos os arquivos .json na pasta atual
arquivos_json = glob.glob(f'{goldenDatasetPath}/*.json')

print(f"Encontrados {len(arquivos_json)} arquivos JSON. Iniciando conversão...")

with open(arquivo_saida, 'w', encoding='utf-8') as saida:
    for nome_arquivo in arquivos_json:
        # Pula o próprio arquivo de saída se ele já existir (para não dar erro)
        if nome_arquivo == arquivo_saida:
            continue
            
        try:
            with open(nome_arquivo, 'r', encoding='utf-8') as entrada:
                # 1. Lê o JSON original
                dados = json.load(entrada)
                
                # Opcional: Adiciona o nome do arquivo original nos dados (útil para rastreio)
                # Se não quiser isso, pode apagar a linha abaixo
                dados['nome_arquivo'] = os.path.basename(nome_arquivo)
                
                # 2. Transforma o objeto em uma string de linha única
                linha_json = json.dumps(dados, ensure_ascii=False)
                
                # 3. Escreve no arquivo final e pula uma linha
                saida.write(linha_json + '\n')
                
            print(f"Sucesso: {nome_arquivo}")
            
        except Exception as e:
            print(f"Erro ao ler {nome_arquivo}: {e}")

print(f"\nConcluído! Seu arquivo '{arquivo_saida}' está pronto.")