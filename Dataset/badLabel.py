import json
import glob
import os
import copy
import random
import re

#region [Config]
pasta_entrada = os.path.join(os.path.dirname(__file__), "GoldenDataset")   
pasta_saida = 'dataset_juiz_gerados' 

if not os.path.exists(pasta_saida):
    os.makedirs(pasta_saida)

def carregar_arquivos(pasta):
    padrao = os.path.join(pasta, '*.json')
    return glob.glob(padrao)

def get_nested(data, path):
    """Navega no JSON usando uma lista de chaves"""
    ref = data
    for key in path:
        if isinstance(ref, dict) and key in ref:
            ref = ref[key]
        else:
            return None
    return ref
#endregion

#region [omissao]
def erro_omissao(dados):
    alvos_possiveis = [
        (['carbon_dots', 'general_info'], 'definition'),
        (['carbon_dots', 'general_info'], 'alternative_names'),
        (['carbon_dots', 'synthesis', 'method'], 'type'),
        (['carbon_dots', 'synthesis', 'method'], 'conditions'),
        (['carbon_dots', 'synthesis', 'precursors'], 'source'),
        (['carbon_dots', 'synthesis'], 'post_treatment'),
        (['carbon_dots', 'properties', 'optical', 'quantum_yield'], 'value'),
        (['carbon_dots', 'properties', 'optical', 'quantum_yield'], 'reference_material'),
        (['carbon_dots', 'properties', 'optical'], 'excitation_max'),
        (['carbon_dots', 'properties', 'optical'], 'emission_max'),
        (['carbon_dots', 'properties', 'optical'], 'fluorescence_lifetime'),
        (['carbon_dots', 'properties', 'structural', 'size'], 'avg'),
        (['carbon_dots', 'properties', 'structural', 'size'], 'distribution')
    ]
    
    random.shuffle(alvos_possiveis)
    
    for caminho, chave_final in alvos_possiveis:
        objeto_pai = get_nested(dados, caminho)
        
        if objeto_pai and chave_final in objeto_pai:
            valor_original = objeto_pai[chave_final]
            
            if valor_original is None or valor_original == "[MISSING]":
                continue
                
            objeto_pai[chave_final] = random.choice([None, "[MISSING]"])
            
            path_str = " -> ".join(caminho + [chave_final])
            comentario = f"Omission Error: The field '{path_str}' contained '{valor_original}' in the source text, but it was extracted as null/missing."
            return dados, comentario, "omissao_campo"
            
    return None, None, None
#endregion

#region [alucinacao]
def erro_alucinacao(dados):
    alvos_numericos = [
        (['carbon_dots', 'properties', 'optical', 'quantum_yield'], 'value'),
        (['carbon_dots', 'properties', 'optical'], 'excitation_max'),
        (['carbon_dots', 'properties', 'optical'], 'emission_max'),
        (['carbon_dots', 'properties', 'structural', 'size'], 'avg'),
        (['carbon_dots', 'properties', 'chemical', 'elemental_composition'], 'C'),
        (['carbon_dots', 'properties', 'chemical', 'elemental_composition'], 'N'),
        (['carbon_dots', 'properties', 'chemical', 'elemental_composition'], 'O')
    ]
    
    random.shuffle(alvos_numericos)
    
    for caminho, chave_final in alvos_numericos:
        objeto_pai = get_nested(dados, caminho)
        
        if objeto_pai and chave_final in objeto_pai:
            valor_str = str(objeto_pai[chave_final])
            
            numeros = re.findall(r'\d+\.?\d*', valor_str)
            
            if numeros:
                num_original = random.choice(numeros)
                try:
                    val_float = float(num_original)
    
                    fator = random.uniform(0.1, 10.0) 
                    novo_val = round(val_float * fator, 2)
                    
                    novo_texto = valor_str.replace(num_original, str(novo_val), 1)
                    objeto_pai[chave_final] = novo_texto

                    path_str = " -> ".join(caminho + [chave_final])
                    
                    comentario = f"Numerical Hallucination: The correct value for {path_str} was '{valor_str}', but the model hallucinated '{novo_texto}'."
                    return dados, comentario, "alucinacao_num"
                except:
                    continue

    return None, None, None
#endregion

def erro_remocao_secao(dados):
    alvos_estruturais = [
        # Seções Principais (Main Sections)
        ['carbon_dots', 'general_info'],
        ['carbon_dots', 'synthesis'],
        ['carbon_dots', 'properties'],
        ['carbon_dots', 'applications'],
        
        # Subseções (Sub-sections)
        ['carbon_dots', 'general_info', 'definition'],
        ['carbon_dots', 'general_info', 'alternative_names'],

        ['carbon_dots', 'synthesis', 'method'],
        ['carbon_dots', 'synthesis', 'precursors'],
        ['carbon_dots', 'synthesis', 'post_treatment'],

        ['carbon_dots', 'properties', 'optical'],
        ['carbon_dots', 'properties', 'structural'],
        ['carbon_dots', 'properties', 'chemical'],

        # Sub-subseções
        ['carbon_dots', 'synthesis', 'method', 'type'],
        ['carbon_dots', 'synthesis', 'method', 'conditions'],
        ['carbon_dots', 'synthesis', 'method', 'catalysts'],

        ['carbon_dots', 'synthesis', 'precursors', 'source'],
        ['carbon_dots', 'synthesis', 'precursors', 'dopants'],

        ['carbon_dots', 'properties', 'optical', 'quantum_yield'],
        ['carbon_dots', 'properties', 'optical', 'excitation_max'],
        ['carbon_dots', 'properties', 'optical', 'emission_max'],
        ['carbon_dots', 'properties', 'optical', 'fluorescence_lifetime'],

        ['carbon_dots', 'properties', 'structural', 'size'],
        ['carbon_dots', 'properties', 'structural', 'morphology'],
        ['carbon_dots', 'properties', 'structural', 'crystallinity'],
        
        ['carbon_dots', 'properties', 'chemical', 'surface_groups'],
        ['carbon_dots', 'properties', 'chemical', 'elemental_composition'],

        # Sub-sub-subseções
        ['carbon_dots', 'properties', 'optical', 'quantum_yield', 'value'],
        ['carbon_dots', 'properties', 'optical', 'quantum_yield', 'reference_material'],

        ['carbon_dots', 'properties', 'structural', 'size', 'avg'],
        ['carbon_dots', 'properties', 'structural', 'size', 'distribution']
    ]
    
    random.shuffle(alvos_estruturais)
    
    for caminho_alvo in alvos_estruturais:
        caminho_pai = caminho_alvo[:-1]
        chave_vitima = caminho_alvo[-1]
        
        objeto_pai = get_nested(dados, caminho_pai)
        
        if objeto_pai and isinstance(objeto_pai, dict) and chave_vitima in objeto_pai:
            del objeto_pai[chave_vitima]
            
            path_str = " -> ".join(caminho_alvo)
            comentario = f"Incomplete Structure: The entire section '{path_str}' was incorrectly omitted. The JSON is missing a mandatory structural block."
            
            return dados, comentario, "falta_estrutura"

    return None, None, None

#region [Main]
if __name__ == "__main__":

    PENALIDADES = {
    "falta_estrutura": 3.0,  
    "alucinacao_num": 2.0,   
    "omissao_campo": 1.0     
    }
    SCORE_INICIAL = 5.0

    erros = [erro_omissao, erro_alucinacao, erro_remocao_secao]

    print(f"Lendo arquivos de '{pasta_entrada}'...")
    arquivos = carregar_arquivos(pasta_entrada)

    total_gerados = 0

    for arquivo_caminho in arquivos:
        with open(arquivo_caminho, 'r', encoding='utf-8') as f:
            json_original = json.load(f)
        
        nome_base = os.path.splitext(os.path.basename(arquivo_caminho))[0]

        feedbacks_positivos = {

            "Ap_V" : "Applicability Verified: The extracted aplication domains align perfectly with those mentioned in the source text. The model accurately identified, listed and described all relevant application areas for the Carbon Dots used in the article.",

            "HPE" : "High Precision Extraction: Numerical values (such as Quantum Yield, Size, and Chemical) were extracted with correct units and significant figures. The model accurately reflects the quantitative data from the source text.",

            "Gnd_V" : "Grounded Verification: All extracted content is fully supported by the source text. The model correctly returned 'null' or '[MISSING]' for fields absent in the document, avoiding unsupported inferences or hallucinations.",

            "HSC" : "High Scientific Context: Scientific data relative to Carbon Dots was extracted with high precision. The model successfully preserved scientific context (e.g., general information and synthesis) and handled nomenclature and experimental details as requested in the 'Data Quality' section.",

            "SI" : "Structural Integrity: The output matches the required JSON schema perfectly. Complex nested arrays (such as precursors/dopants and application domains) were populated correctly without truncating list items or breaking the syntax.",
            
            "PM" : "Perfect Match: The extraction captures all relevant technical details defined in the 'Extraction Scope'. Synthesis methods, precursors, and optical properties align precisely with the source text, demonstrating full compliance with the expert guidelines."
        }
        
        itens_feedback = list(feedbacks_positivos.items())

        qtd_escolha = min(2, len(itens_feedback))
        escolhidos = random.sample(itens_feedback, k=qtd_escolha)

        tags_sorteadas = [item[0] for item in escolhidos]      
        comentarios_sorteados = [item[1] for item in escolhidos] 

        tags_para_nome = "_".join(tags_sorteadas)             
        comentario_final = " | ".join(comentarios_sorteados)
        
        registro_good = {
            "input_source": os.path.basename(arquivo_caminho),  
            "text_content": json_original, 
            "feedback": {
                "label": "Good",
                "score": 5.0,
                "expert_comment": comentario_final  
            }
        }

        nome_final_good = f"{nome_base}_Good_{tags_para_nome}.json"
        caminho_final_good = os.path.join("FeedbackGoldenDataset", nome_final_good)
        
        with open(caminho_final_good, 'w', encoding='utf-8') as f_out:
            json.dump(registro_good, f_out, indent=2, ensure_ascii=False)
            
        print(f"Gerado: {nome_final_good} (Score 5.0)")

        dados_acumulados = copy.deepcopy(json_original)
        mensagens_acumuladas = []
        tags_aplicadas = []
        score_atual = SCORE_INICIAL

        qtd_max_erros = min(3, len(erros))
        qtd_erros = random.randint(1, qtd_max_erros)

        estrategias_sorteadas = random.choices(erros, k=qtd_erros)
        
        for estrategia in estrategias_sorteadas:
            dados_temp, msg, tag = estrategia(dados_acumulados)
            
            if dados_temp is not None:
                dados_acumulados = dados_temp
                mensagens_acumuladas.append(msg)
                tags_aplicadas.append(tag)

                penalidade = PENALIDADES.get(tag, 0.0)
                score_atual -= penalidade
        
        score_final = max(0.0, score_atual)

        tag = ""
        if score_final < 2.0:
            tag = "Bad"
        elif score_final >= 2.0 and score_final < 5.0:
            tag = "Regular"
        else:
            tag = "Good"

        if len(tags_aplicadas) > 0:

            comentario_final = " | ".join(mensagens_acumuladas)

            tag_final = "_".join(tags_aplicadas)
            
            registro_bad = {
                "input_source": os.path.basename(arquivo_caminho),     
                "text_content": dados_acumulados, 
                "feedback": {
                    "label": tag,
                    "score": round(score_final, 1),
                    "expert_comment": comentario_final  
                }
            }

            nome_final = f"{nome_base}_{tag}_{tag_final}.json"
            if len(nome_final) > 150: 
                nome_final = f"{nome_base}_BAD_MultiplosErros.json"
                
            caminho_final = os.path.join(pasta_saida, nome_final)
            """
            with open(caminho_final, 'w', encoding='utf-8') as f_out:
                json.dump(registro_bad, f_out, indent=2, ensure_ascii=False)
            """
            total_gerados += 1
            print(f"Gerado: {nome_final} ({len(tags_aplicadas)} erros)")

    print(f"\nConcluído! Foram gerados {total_gerados} arquivos com erros acumulados.")
#endregion