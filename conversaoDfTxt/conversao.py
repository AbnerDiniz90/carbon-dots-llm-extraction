import pandas as pd
import os

def csv_to_formatted_txt(input_csv_path, output_txt_path):
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{input_csv_path}' não foi encontrado.")
        return
    except Exception as e:
        print(f"Erro ao ler o CSV: {e}")
        return

    column_mapping = {
        'model': 'model',
        'prompt': 'prompt_generated',      
        'length': 'Prompt_Length',          
        'evaluation': 'Structure_Evaluation',
        'reasoning': 'Structure_Evaluation reasoning'
    }
    
    special_columns_values = list(column_mapping.values())

    separator = "=" * 30

    with open(output_txt_path, 'a', encoding='utf-8') as f:
        
        for index, row in df.iterrows():
            
            model_val = row.get(column_mapping['model'], "Modelo Desconhecido")
            f.write("MODELO: \n\n")
            f.write(f"{model_val}\n")
            f.write(f"{separator}\n\n")

            prompt_col = column_mapping['prompt']
            if prompt_col in df.columns:
                f.write("PROMPT_GERADO:\n\n")
                f.write(f"{row[prompt_col]}\n")
                f.write(f"{separator}\n\n")

            len_col = column_mapping['length']
            if len_col in df.columns:
                f.write("PROMPT_LENGHT:\n") 
                f.write(f"{row[len_col]}\n")
                f.write(f"{separator}\n\n")

            eval_col = column_mapping['evaluation']
            if eval_col in df.columns:
                f.write("STRUCTURE_EVALUATION:\n")
                f.write(f"{row[eval_col]}\n")
                f.write(f"{separator}\n\n")

            reason_col = column_mapping['reasoning']
            if reason_col in df.columns:
                f.write("STRUCTURE_EVALUATION REASONING:\n")
                f.write(f"{row[reason_col]}\n")
                f.write(f"{separator}\n\n")

            for col_name in df.columns:
                if col_name not in special_columns_values:
                    header_title = col_name.upper().replace(" ", "_")
                    
                    f.write(f"{header_title}:\n")
                    f.write(f"{row[col_name]}\n")
                    f.write(f"{separator}\n\n")

            if index < len(df) - 1:
                f.write("\n" + "#" * 50 + "\n\n")

    print(f"Processamento concluído! Arquivo salvo em: {output_txt_path}")

arquivo_entrada = '019bb2a0-c4c8-7bb7-b395-40c3321b5e7d.csv'
arquivo_saida = 'resultado_formatado.txt'

if __name__ == "__main__":
    csv_to_formatted_txt(arquivo_entrada, arquivo_saida)