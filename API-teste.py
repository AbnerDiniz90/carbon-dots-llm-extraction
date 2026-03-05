
import xml.etree.ElementTree as ET
import os
import requests
import json
import re
from datetime import datetime
import pandas as pd
import glob
from evidently import DataDefinition, Dataset, Report
from evidently.presets import TextEvals
from evidently.descriptors import LLMEval, TextLength
from evidently.llm.templates import MulticlassClassificationPromptTemplate, BinaryClassificationPromptTemplate
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# region [Classes]
class modelExtractor:
    """
    Classe que representa a configuração e o estado de um Modelo de Linguagem (LLM) extrator.
    Atua como um contêiner para armazenar os parâmetros de execução, facilitando a passagem 
    dessas configurações entre as diversas funções do pipeline de extração de artigos.

    Atributos:
    ----------
    modelo : str
        O identificador do modelo na API (ex: 'openrouter/z-ai/glm-5').
    apiKey : str
        A chave de autenticação para acesso à API do modelo.
    batch_size : int, opcional
        O limite de artigos a serem processados por lote. Se None, processará todos os disponíveis.
    pasta_xmls : str
        O caminho do diretório onde os artigos científicos em formato XML estão armazenados.
    """

    def __init__(self, modelo, apiKey, batch_size=None, pasta_xmls="carbon_dots"):
        self.modelo = modelo
        self.apiKey = apiKey
        self.batch_size = batch_size
        self.pasta_xmls = pasta_xmls

class evalDataset:
    """
    Classe contêiner que gerencia todos os caminhos de diretórios e as estruturas de dados 
    necessárias para o pipeline de avaliação (LLM-as-a-Judge). Centraliza o estado dos 
    datasets em suas diferentes etapas (bruto, pandas e Evidently).

    Atributos:
    ----------
    pasta_ex : str
        Caminho para a pasta contendo os arquivos JSON de exemplo (usados no Bloco A de calibração).
    pasta_ref : str
        Caminho para a pasta contendo os arquivos JSON de referência/gabarito (Ground Truth).
    jsonDataset : pd.DataFrame, opcional
        O DataFrame do pandas contendo a mescla dos exemplos com os candidatos a serem avaliados.
    evidentlyJsonDataset : evidently.Dataset, opcional
        O dataset convertido e tipado com as definições do Evidently AI, pronto para a auditoria.
    resultados_llm : list, opcional
        Lista contendo as extrações brutas geradas pela LLM na etapa de predição.
    testEval : bool
        Flag (bandeira) que define o comportamento do dataset: True para testes internos do prompt 
        do juiz (A/B testing) e False para o fluxo padrão de avaliação de extração em massa.
    """

    def __init__(self, pasta_ex, pasta_ref, jsonDataset = None, evidentlyJsonDataset = None, resultados_llm=None, testEval=False):
        self.pasta_ex = pasta_ex
        self.pasta_ref = pasta_ref
        self.jsonDataset = jsonDataset
        self.evidentlyJsonDataset = evidentlyJsonDataset
        self.resultados_llm = resultados_llm
        self.testEval = testEval
# endregion

# region [OpenAlex API]

def baixar_artigo_openAlex(topico, max_artigos=None, json_output=False):
    """
    Função para conectar à API OpenAlex e baixar artigos relacionados a um tópico específico.

    :param topico: Tópico de busca na API OpenAlex
    :param max_artigos: Número máximo de artigos a serem baixados (opcional)
    :param json_output: Se True, salva os resultados em um arquivo JSON
    """

    base_url = "https://api.openalex.org/works"

    parameters = {
        "filter": f"default.search:{topico},open_access.is_oa:true",
        "per-page": 200, 
        "cursor": "*",    
        "mailto": "abner.diniz@unifesp.br"
    }
    
    todos_artigos = []

    while max_artigos is None or len(todos_artigos) < max_artigos:
        try:
            response = requests.get(base_url, params=parameters)
            
            if response.status_code != 200:
                print(f"Erro: {response.status_code}")
                break
                
            data = response.json()
            results = data.get('results', [])
            meta = data.get('meta', {})
            
            if not results:
                print("Fim dos resultados.")
                break
                
            todos_artigos.extend(results)

            print(f"Baixados +{len(results)} artigos. Total acumulado: {len(todos_artigos)}")

            next_cursor = meta.get('next_cursor')
            
            if not next_cursor:
                break
                
            parameters['cursor'] = next_cursor

        except Exception as e:
            print(f"Ocorreu um erro: {e}")
            break

        if json_output:    
            json_arq = f"result_openalex_{topico}.json"

            with open(json_arq, "w", encoding="utf-8") as f:
                json.dump(todos_artigos, f, ensure_ascii=False, indent=4)

# endregion

# region [.txt extract]
def formatar_tabela_markdown(table_elem, tag_cleaner):
    """
    Converte um elemento de tabela estruturada em XML para o formato Markdown.
    Esta conversão é crucial porque preserva a relação espacial e semântica das linhas e colunas 
    em um formato de texto plano, permitindo que os Modelos de Linguagem (LLMs) consigam 
    interpretar nativamente os dados tabulares do artigo.

    :param table_elem: O elemento raiz da tabela extraído pela árvore XML (xml.etree.ElementTree).
    :param tag_cleaner: Função auxiliar usada para higienizar e extrair o texto interno das tags HTML/XML.
    :return: Uma string contendo a representação da tabela perfeitamente formatada em Markdown.
    """

    label = ""
    caption = ""
    
    for child in table_elem.iter():
        tag = tag_cleaner(child)
        if tag == 'label':
            label = "".join(child.itertext()).strip()
        elif tag == 'caption':
            caption = " ".join("".join(child.itertext()).split())
    
    titulo_tabela = f"### {label}: {caption} ###\n" if label or caption else "### TABLE ###\n"
    
    tgroup = None
    for child in table_elem:
        if tag_cleaner(child) == 'tgroup':
            tgroup = child
            break
            
    if not tgroup:
        return "" 

    header_rows = []
    for section in tgroup:
        if tag_cleaner(section) == 'thead':
            for row in section:
                if tag_cleaner(row) == 'row':
                    cols = []
                    for entry in row:
                        if tag_cleaner(entry) == 'entry':
                            cols.append(" ".join("".join(entry.itertext()).split()))
                    header_rows.append(f"| {' | '.join(cols)} |")
    
    body_rows = []
    for section in tgroup:
        if tag_cleaner(section) == 'tbody':
            for row in section:
                if tag_cleaner(row) == 'row':
                    cols = []
                    for entry in row:
                        if tag_cleaner(entry) == 'entry':
                            cols.append(" ".join("".join(entry.itertext()).split()))
                    body_rows.append(f"| {' | '.join(cols)} |")

    tabela_final = [titulo_tabela]
    
    if header_rows:
        tabela_final.extend(header_rows)
        num_cols = header_rows[0].count('|') // 2
        tabela_final.append(f"| {' | '.join(['---'] * max(1, num_cols))} |")
    
    tabela_final.extend(body_rows)
    
    return "\n".join(tabela_final) + "\n"

def extrair_metodologia_por_numero(caminho_artigo, save_txt=False):
    """
    Realiza o parsing estrutural (varredura) de um arquivo de artigo científico em formato XML.
    Filtra tags ruidosas (como afiliações, autores e referências bibliográficas) e converte tabelas
    para a sintaxe Markdown, extraindo apenas o conteúdo textual útil para a etapa de síntese.

    :param caminho_artigo: Caminho (string) do arquivo XML local a ser lido.
    :param save_txt: Booleano que indica se o texto extraído deve ser salvo em um arquivo .txt na pasta "TxtArtigos".
    :return: Uma string contendo o texto higienizado e consolidado do artigo científico.
    """
    if not os.path.exists(caminho_artigo):
        return None

    try:
        tree = ET.parse(caminho_artigo)
        root = tree.getroot()

        def tag_name(elem):
            if '}' in elem.tag:
                return elem.tag.split('}')[-1]
            return elem.tag

        texto_limpo = []
        parent_map = {c: p for p in root.iter() for c in p}
        
        tags_ignorar = [
            'affiliation', 'correspondence', 'date-received', 
            'date-accepted', 'copyright', 'bib-reference',
            'acknowledgment', 'grant-sponsor', 'maintitle', 
            'bibliography', 'contribution', 'reference', 'series', 'issue', 'host', 
            'authors'
        ]

        tags_conteudo = [
            'para', 'simple-para', 'note-para', 'list-item', 'text', 'given-name', 'surname', 'title', 'label'
        ]

        for elem in root.iter():
            tag = tag_name(elem)
            
            if tag == 'table':
                tabela_md = formatar_tabela_markdown(elem, tag_name)
                texto_limpo.append(f"\n{tabela_md}\n")
                continue 

            ancestral_bloqueado = False
            dentro_de_tabela = False 
            
            curr = elem
            while True:
                parent = parent_map.get(curr)
                if parent is None:
                    break
                
                p_tag = tag_name(parent)
                
                if p_tag == 'table':
                    dentro_de_tabela = True
                    break

                if p_tag in tags_ignorar:
                    ancestral_bloqueado = True
                    break
                
                curr = parent
            
            if ancestral_bloqueado or dentro_de_tabela:
                continue

            if tag in tags_ignorar:
                continue

            allow = False
            parent = parent_map.get(elem)
            if parent is not None and tag_name(parent) in ['figure', 'table']:
                allow = True

            if tag == 'title' or tag == 'section-title':

                titulo_formatado = "".join(elem.itertext()).strip()
                
                if titulo_formatado:
                    header = "TITLE" if tag == 'title' else titulo_formatado.upper()

                    if tag == 'section-title':
                        texto_limpo.append(f"\n\n### {header} ###\n")
                    else:
                        texto_limpo.append(f"\n\n### {header} ###\n\n{titulo_formatado}")
                continue
            
            if tag == 'abstract':
                texto_limpo.append("\n\n### ABSTRACT ###\n")
                continue

            if tag == 'author-group':
                texto_limpo.append("\n\n### AUTHOR GROUP ###\n")
                continue

            elif tag in tags_conteudo:
                conteudo_atual = " ".join(("".join(elem.itertext())).split())

                if tag == 'given-name':
                    nome = conteudo_atual
                    continue 

                elif tag == 'surname':
                    sobrenome = conteudo_atual
                    nome_completo = f"{nome} {sobrenome}"
                    
                    if nome and sobrenome:
                         conteudo_atual = nome_completo

                    nome = ""
                    sobrenome = ""

                if tag == 'label' :
                    if allow:
                        label_text = conteudo_atual.upper()
                    continue

                elif tag == 'simple-para' and len(label_text) > 1:
                    caption = " ".join("".join(elem.itertext()).split())
                    conteudo_atual = f"\n\n### {label_text}: {caption} ###"

                    label_text = ""
                    caption = ""

                if conteudo_atual:
                    texto_limpo.append(conteudo_atual)
        
        if save_txt:
            nameArticle = f"{os.path.basename(caminho_artigo).replace('.xml', '')}.txt"
            with open(os.path.join("TxtArtigos", nameArticle), "w", encoding="utf-8") as f:
                f.write("\n".join(texto_limpo))

        return "\n".join(texto_limpo)

    except Exception as e:
        return f"Erro: {e}"

# endregion

# region [Json extract]
def limpar_resposta_json(texto_bruto):
    """
    Higieniza a string bruta retornada pela API da LLM, garantindo que seja um JSON válido.
    Remove artefatos comuns gerados por modelos de linguagem, como blocos de código Markdown 

    :param texto_bruto: A string bruta retornada pela LLM, que pode conter blocos de código, texto adicional ou formatação Markdown.
    :return: Uma string contendo apenas o JSON limpo, pronto para ser decodificado por json.loads().
    """
    if not texto_bruto:
        return None

    texto_limpo = texto_bruto.replace("```json", "").replace("```", "")
    
    texto_limpo = texto_limpo.strip()

    idx_inicio = texto_limpo.find("{")
    idx_fim = texto_limpo.rfind("}")
    
    if idx_inicio != -1 and idx_fim != -1:
        texto_limpo = texto_limpo[idx_inicio : idx_fim + 1]

    return texto_limpo

def analisar_com_llm(texto_metodologia, model):
    """
    Envia o texto estruturado do artigo para a API do modelo de linguagem solicitando a extração.
    Utiliza o parâmetro 'response_format' ou similar para forçar a LLM a retornar a saída 
    estritamente amarrada ao JSON Schema hierárquico dos Carbon Dots.

    :param texto_metodologia: String contendo o texto limpo do artigo a ser analisado.
    :param modelo: Identificador do modelo na API (ex: 'openrouter/z-ai/glm-5').
    :return: Um dicionário (dict) em Python com os dados químicos e morfológicos extraídos.
    """

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {model.apiKey}",
        "Content-Type": "application/json"
    }

    modelo = model.modelo
    schema_metodologia = {
        "name": "extracao_metodologia",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "carbon_dots": {
                    "type": "object",
                    "properties": {
                        "general_info": {
                            "type": "object",
                            "properties": {
                                "definition": {"type": ["string", "null"], "description": "Brief definition of the material"},
                                "iupac_name": {"type": ["string", "null"]},
                                "alternative_names": {
                                    "type": ["array", "null"],
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["definition", "iupac_name", "alternative_names"],
                            "additionalProperties": False
                        },
                        "synthesis": {
                            "type": "object",
                            "properties": {
                                "method": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": ["string", "null"], "description": "e.g., Hydrothermal, Pyrolysis"},
                                        "conditions": {"type": ["string", "null"], "description": "Temp, time, pressure"},
                                        "catalysts": {"type": ["array", "null"], "items": {"type": "string"}}
                                    },
                                    "required": ["type", "conditions", "catalysts"],
                                    "additionalProperties": False
                                },
                                "precursors": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": ["array", "null"], "items": {"type": "string"}},
                                        "dopants": {"type": ["array", "null"], "items": {"type": "string"}}
                                    },
                                    "required": ["source", "dopants"],
                                    "additionalProperties": False
                                },
                                "post_treatment": {
                                    "type": ["array", "null"],
                                    "items": {"type": "string"},
                                    "description": "Purification methods like dialysis"
                                }
                            },
                            "required": ["method", "precursors", "post_treatment"],
                            "additionalProperties": False
                        },
                        "properties": {
                            "type": "object",
                            "properties": {
                                "optical": {
                                    "type": "object",
                                    "properties": {
                                        "quantum_yield": {
                                            "type": ["object", "null"],
                                            "properties": {
                                                "value": {"type": ["string", "number", "null"]},
                                                "reference_material": {"type": ["string", "null"]}
                                            },
                                            "required": ["value", "reference_material"],
                                            "additionalProperties": False
                                        },
                                        "excitation_max": {"type": ["string", "null"]},
                                        "emission_max": {"type": ["string", "null"]},
                                        "fluorescence_lifetime": {"type": ["string", "null"]}
                                    },
                                    "required": ["quantum_yield", "excitation_max", "emission_max", "fluorescence_lifetime"],
                                    "additionalProperties": False
                                },
                                "structural": {
                                    "type": "object",
                                    "properties": {
                                        "size": {
                                            "type": ["object", "null"],
                                            "properties": {
                                                "avg": {"type": ["string", "null"]},
                                                "distribution": {"type": ["string", "null"]}
                                            },
                                            "required": ["avg", "distribution"],
                                            "additionalProperties": False
                                        },
                                        "morphology": {"type": ["string", "null"]},
                                        "crystallinity": {"type": ["string", "null"]}
                                    },
                                    "required": ["size", "morphology", "crystallinity"],
                                    "additionalProperties": False
                                },
                                "chemical": {
                                    "type": "object",
                                    "properties": {
                                        "surface_groups": {
                                            "type": ["array", "null"],
                                            "items": {"type": "string"}
                                        },
                                        "elemental_composition": {
                                            "type": ["object", "null"],
                                            "properties": {
                                                "C": {"type": ["string", "null"]},
                                                "N": {"type": ["string", "null"]},
                                                "O": {"type": ["string", "null"]},
                                                "[Other_Elements]": {"type": ["string", "null"]}
                                            },
                                            "required": ["C", "N", "O"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "required": ["surface_groups", "elemental_composition"],
                                    "additionalProperties": False
                                }
                            },
                            "required": ["optical", "structural", "chemical"],
                            "additionalProperties": False
                        },
                        "applications": {
                            "type": ["array", "null"],
                            "items": {
                                "type": "object",
                                "properties": {
                                    "domain": {"type": ["string", "null"]},
                                    "performance": {"type": ["string", "null"]},
                                    "comparative_advantage": {"type": ["string", "null"]}
                                },
                                "required": ["domain", "performance", "comparative_advantage"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["general_info", "synthesis", "properties", "applications"],
                    "additionalProperties": False
                }
            },
            "required": ["carbon_dots"],
            "additionalProperties": False
        }
    }

    prompt_sistema = """
        You are a scientific data extraction specialist with expertise in nanomaterials chemistry.
        Extract all relevant technical details about Carbon Dots (CDs) from the provided text
        and structure them into a comprehensive JSON format. Follow ALL rules rigorously.


        ══════════════════════════════════════════════════════
        PRIME DIRECTIVE — ANTI-HALLUCINATION MANDATE
        ══════════════════════════════════════════════════════
        You MUST extract ONLY from the provided Article_Content text.
        You are PROHIBITED from using any knowledge from your training data to fill JSON fields.
        Every single extracted value must be traceable to a specific sentence in the text.


        BEFORE filling any JSON field, you MUST internally perform this check:
        "What exact sentence or phrase in Article_Content supports this value?"
        → If you cannot identify a specific supporting sentence: use null / [MISSING].
        → If you are unsure or extrapolating from context: use null / [MISSING].


        ══════════════════════════════════════════════════════
        STEP 0 — INVALID INPUT DETECTION (run FIRST)
        ══════════════════════════════════════════════════════
        Before any extraction, check: Is the Article_Content a valid scientific text?
        BAD SIGNALS: Python/system error messages, empty strings, lorem ipsum,
                    non-English gibberish, file-not-found messages.
        → If Article_Content is invalid/empty: IMMEDIATELY return the full JSON schema
            with ALL fields set to null. Do not attempt any extraction.


        ══════════════════════════════════════════════════════
        STEP 1 — MATERIAL VERIFICATION
        ══════════════════════════════════════════════════════
        ACCEPTED MATERIALS: Carbon Dots (CDs), Carbon Quantum Dots (CQDs),
        Graphene Quantum Dots (GQDs), Carbon Nanodots (CNDs) or direct synonyms.


        REJECTED MATERIALS (return all-null JSON if article is primarily about these):
        dialysed caramel, organic dyes, fluorescent polymers, metallic nanoparticles,
        semiconductor QDs (CdSe etc.), nanotubes, Polymer Dots (PDs),
        Copolymer Dots (e.g., PDA-PEI dots), PEI nanoparticles.


        If the primary material is REJECTED, set ALL JSON fields to null immediately.
        Do NOT force non-CD properties into the CD schema.


        ══════════════════════════════════════════════════════
        STEP 2 — PRE-EXTRACTION SCAN (mandatory)
        ══════════════════════════════════════════════════════
        Before writing any JSON, scan Article_Content and produce an internal checklist:
        [ ] Synthesis method type mentioned? (quote it)
        [ ] Synthesis conditions (T, time, atmosphere) stated? (quote values)
        [ ] Precursor names and amounts stated? (quote them)
        [ ] Dopants explicitly stated as dopants? (quote)
        [ ] Particle size explicitly stated? (quote value + unit)
        [ ] Morphology explicitly named (spherical, nanosheet...)? (quote)
        [ ] Crystallinity described? (quote)
        [ ] Quantum yield with reference material? (quote both)
        [ ] Excitation max explicitly stated as a peak/maximum? (quote)
        [ ] Emission max explicitly stated as a peak/maximum? (quote)
        [ ] Fluorescence lifetime value given? (quote)
        [ ] Elemental composition percentages given? (quote with source: XPS/EDS/EA)
        [ ] Surface functional groups listed? (quote)
        [ ] Post-treatment steps listed in order? (quote each)
        [ ] Application domain and performance metrics? (quote)
        Only fields confirmed in this scan should be populated. All others → null/[MISSING].


        ══════════════════════════════════════════════════════
        SCOPE & ISOLATION RULES
        ══════════════════════════════════════════════════════
        Scientific papers often combine CDs with other materials (TiO2, polymers, hydrogels)
        to build a sensor or device. STRICTLY ISOLATE the Carbon Dot properties.


        Commercial Purchase: If CDs were purchased (not synthesized), set
        synthesis.method.type = "Commercial purchase",
        synthesis.method.conditions = null,
        synthesis.precursors.source = null.
        Do NOT invent synthesis conditions for a bought material.


        Functionalized Probe vs. Hybrid Device:
        - Soluble probe (CD + small molecule/polymer, still dispersible):
            Extract properties of the FINAL functionalized probe.
        - Solid substrate device (CD on electrode/hydrogel/TiO2):
            Extract BARE CD properties only. Device fabrication steps → applications field.


        ══════════════════════════════════════════════════════
        FIELD-SPECIFIC RULES
        ══════════════════════════════════════════════════════
        synthesis.method.conditions: Be EXHAUSTIVE.
        Include temperature, time, atmosphere (N2, Ar), specific equipment
        (Teflon-lined autoclave, dialysis MWCO), heating rate if stated.
        Quote directly from text.


        synthesis.precursors.dopants: ONLY list as dopant if the article EXPLICITLY states
        doping OR a secondary precursor is added specifically to introduce heteroatoms.
        If the main precursor already contains N/S/B (e.g., 3-pyridylboronic acid),
        do NOT list those atoms as dopants.


        properties.structural.morphology: ONLY extract explicit geometric shapes
        (spherical, cubic, nanosheet, etc.) from TEM/HRTEM characterization text.
        "Monodisperse", "uniform", "equally distributed" → these describe distribution,
        NOT morphology. If only distribution adjectives are used → [MISSING].


        properties.optical.excitation_max & emission_max:
        ONLY extract if the text identifies the value as the peak/maximum/optimum.
        "measured at 320 nm for this assay" is NOT the excitation maximum → [MISSING].


        properties.chemical.elemental_composition:
        Add dynamic keys for ANY element quantified (XPS, EDS, EA).
        Example: {"C":"37.5%","O":"31.7%","N":"8.5%","B":"22.0%","Fe":"3.1%"}


        ══════════════════════════════════════════════════════
        JSON SCHEMA (use exact field order)
        ══════════════════════════════════════════════════════
        {
        "carbon_dots": {
            "general_info": { "definition": "string", "iupac_name": "string",
                            "alternative_names": "array" },
            "synthesis": {
            "method": { "type": "string", "conditions": "string", "catalysts": "array" },
            "precursors": { "source": "array", "dopants": "array" },
            "post_treatment": "array"
            },
            "properties": {
            "optical": {
                "quantum_yield": { "value": "percentage", "reference_material": "string" },
                "excitation_max": "string (nm)", "emission_max": "string (nm)",
                "fluorescence_lifetime": "string (ns)"
            },
            "structural": {
                "size": { "avg": "string (nm)", "distribution": "string" },
                "morphology": "string", "crystallinity": "string"
            },
            "chemical": {
                "surface_groups": "array",
                "elemental_composition": { "C": "string%", "N": "string%", "O": "string%", "[Other_Elements]": "string%"}
            }
            },
            "applications": [{ "domain": "string", "performance": "string",
                            "comparative_advantage": "string" }]
        }
        }


        ══════════════════════════════════════════════════════
        VALIDATION CHECKLIST (before final output)
        ══════════════════════════════════════════════════════
        [ ] Every value can be traced to a specific sentence in Article_Content
        [ ] No value was inferred from training knowledge (not from the text)
        [ ] Commercial purchase → conditions/precursors are null
        [ ] Morphology is an explicit geometric term, NOT a distribution adjective
        [ ] Excitation/emission are true peaks, not assay measurement wavelengths
        [ ] Dopants are explicitly labeled as such in the text
        [ ] Post-treatment steps are in the correct order from the text
        [ ] All extracted data refers to Carbon Dots, not precursors or reference materials
        [ ] JSON field order matches the schema above
        [ ] Missing fields → null (not omitted from JSON)

"""
    payload = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": texto_metodologia}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": schema_metodologia
        },
        "temperature": 0,
        "reasoning": {"enabled": True}
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            dados_api = response.json() 
            
            conteudo_bruto = dados_api['choices'][0]['message']['content']
    
            print(f"   [Debug] Recebido {len(conteudo_bruto)} caracteres.")
            
            conteudo_limpo = limpar_resposta_json(conteudo_bruto)
            
            if not conteudo_limpo:
                print("   -> Erro: A LLM retornou vazio ou inválido.")
                return None

            try:
                dados_json = json.loads(conteudo_limpo)
                return dados_json
                
            except json.JSONDecodeError as e:
                print(f"   -> Erro ao decodificar JSON: {e}")
                print(f"   -> Trecho: {conteudo_limpo[:100]}...") 
                return None
        else:
            print(f"Erro na API ({response.status_code}): {response.text}")
            return None
            
    except Exception as e:
        print(f"Erro de conexão ou timeout: {e}")
        return None

def processar_lote_artigos(model):
    """
    Orquestra o processamento em lote (batch processing) da extração de dados científicos.
    Itera de forma sistemática sobre os artigos do corpus de teste, invoca a função de extração 
    (analisar_com_llm) para cada documento utilizando o modelo especificado e compila todos os resultados.

    :param model: Identificador do modelo de linguagem a ser utilizado (ex: 'openrouter/z-ai/glm-5').
    :return: Uma lista de dicionários contendo os dados químicos e morfológicos extraídos de cada artigo.
    """
    
    resultados_finais = []

    modelo = model.modelo
    pasta_xmls = model.pasta_xmls
    batch_size = model.batch_size 

    arquivos = [f for f in os.listdir(pasta_xmls) if f.endswith('.xml')]

    arquivos = sorted(arquivos, key=lambda f: int(re.search(r'[0-9]+', f).group()))
    
    print(f"Encontrados {len(arquivos)} artigos. Iniciando pipeline...")

    dict_txt = {}

    if batch_size is None:
        batch_size = len(arquivos)

    for arquivo in arquivos[:batch_size]:
        caminho = os.path.join(pasta_xmls, arquivo)
        print(f"Processando: {arquivo} na RAM...")

        texto_metodologia = extrair_metodologia_por_numero(caminho, save_txt=True)

        if not texto_metodologia:
            print(f"  -> Vazio ou erro no XML.")
            continue

        dict_txt[arquivo] = texto_metodologia

        dados_extraidos = analisar_com_llm(texto_metodologia, model)
        
        if dados_extraidos:
            registro = {
                "arquivo_origem": arquivo,
                "dados_extraidos": dados_extraidos,
            }
            resultados_finais.append(registro)
            print(f"  -> Sucesso!")
        else:
            print(f"  -> Falha na LLM.")

    return resultados_finais, dict_txt
# endregion

# region [Json eval]

def ConverterJsonTexto(data, nivel=0):
    """
    Converte recursivamente um objeto JSON (dicionário ou lista) em texto plano estruturado.
    Remove caracteres problemáticos (como chaves '{}' e aspas) que podem quebrar a formatação 
    do f-string no Python ou causar erros de sintaxe (parsing) quando o texto é injetado 
    no prompt do LLM Juiz.

    :param data: O objeto (dict, list, str, int, etc.) a ser convertido.
    :param nivel: Nível de indentação atual (usado internamente para controle de recursão).
    :return: Uma string contendo os dados formatados em texto plano e com indentação hierárquica.
    """

    texto = ""
    espaco = "  " * nivel
    
    if isinstance(data, dict):
        for chave, valor in data.items():
            chave_limpa = str(chave).replace("{", "").replace("}", "").replace("_", " ").title()
            
            if isinstance(valor, (dict, list)):
                texto += f"{espaco}{chave_limpa}:\n{ConverterJsonTexto(valor, nivel + 1)}"
            else:
                valor_limpo = str(valor).replace("{", "(").replace("}", ")")
                texto += f"{espaco}{chave_limpa}: {valor_limpo}\n"
                
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                texto += f"{espaco}- Item:\n{ConverterJsonTexto(item, nivel + 1)}"
            else:
                valor_limpo = str(item).replace("{", "(").replace("}", ")")
                texto += f"{espaco}- {valor_limpo}\n"
    
    else:
        return str(data).replace("{", "(").replace("}", ")") + "\n"

    return texto

def JsonEvalFolderDataset(pasta_ex, pasta_ref, batch_size=None, testEval=False):
    """
    Lê arquivos JSON locais e constrói a base de dados em DataFrame contendo o Bloco A e/ou Bloco B.
    Se testEval for True, divide os arquivos lidos na metade: uma parte vira calibração (Bloco A) 
    e a outra vira candidato (Bloco B) para testar e otimizar o prompt do juiz internamente.

    :param pasta_ex: Caminho da pasta contendo os arquivos de extração originais (exemplos).
    :param pasta_ref: Caminho da pasta contendo os arquivos de referência (Ground Truth / Gabarito).
    :param batch_size: Número máximo de arquivos a serem processados. Se None, processa todos.
    :param testEval: Booleano que define se o dataset será dividido para teste interno do juiz (True) 
                     ou se será preparado para o pipeline principal de extração (False).
    :return: Um DataFrame (pandas) estruturado contendo os dados de calibração e/ou candidatos.
    """

    dados = []
    referencias = {}

    arquivos_ex = glob.glob(os.path.join(pasta_ex, '*.json'))
    arquivos_ref = glob.glob(os.path.join(pasta_ref, '*.json'))

    arquivos_ex.sort(key=lambda x: int(re.search(r'[0-9]+', os.path.basename(x)).group()))
    arquivos_ref.sort(key=lambda x: int(re.search(r'[0-9]+', os.path.basename(x)).group()))

    if batch_size is None:
        batch_size = len(arquivos_ex)
    
    for arquivo in arquivos_ref:
        try:
            nome_arquivo = os.path.basename(arquivo)
            with open(arquivo, 'r', encoding='utf-8') as f:
                conteudo = json.load(f)
    
            referencias[nome_arquivo] = ConverterJsonTexto(conteudo)

        except Exception as e:
            print(f"Erro ao ler referência {arquivo}: {e}")

    if testEval:
        meio = len(arquivos_ex[:batch_size]) // 2  
        arquivos_exemplos = arquivos_ex[:meio]
        arquivos_candidatos = arquivos_ex[meio:batch_size]
        
        print(f"[Modo Teste] Dividindo {batch_size} arquivos:")
        print(f"  - Bloco A (Exemplos): {len(arquivos_exemplos)} arquivos")
        print(f"  - Bloco B (Candidatos): {len(arquivos_candidatos)} arquivos")
        
    else:
        arquivos_exemplos = arquivos_ex[:batch_size]
        arquivos_candidatos = []

    bloco_a = []
    for arquivo in arquivos_exemplos:
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                conteudo = json.load(f)
                
            nome_origem = conteudo.get('input_source')
            label = conteudo.get('feedback', {}).get('label', 'Unknown')
            json_generated = ConverterJsonTexto(conteudo.get('prediction_output') or conteudo.get('text_content'))
            expertComment = conteudo.get('feedback', {}).get('expert_comment', '')

            jsonReferenceExamples = referencias.get(nome_origem, "")
            
            bloco_a.append({
                'Artigo_Example': nome_origem,
                'JSON_Example': json_generated,
                'Label_Example': label,
                'Comment_Example': expertComment,
                'JSON_Reference_Example': jsonReferenceExamples
            })
            
        except Exception as e:
            print(f"Erro ao processar exemplo {arquivo}: {e}")
    
    bloco_b = []
    if testEval and arquivos_candidatos:
        for arquivo in arquivos_candidatos:
            try:
                with open(arquivo, 'r', encoding='utf-8') as f:
                    conteudo = json.load(f)
                
                nome_origem = conteudo.get('input_source')
                text_content = conteudo.get('text_content')
                label_candidate = conteudo.get('feedback', {}).get('label', 'Unknown')

                json_reference_candidate = referencias.get(nome_origem, "")
                
                if testEval:    
                    bloco_b.append({
                        'Artigo_Candidate': nome_origem,
                        'JSON_Candidate': ConverterJsonTexto(text_content) if text_content else "",
                        'Actual_Label_Candidate': label_candidate,
                        'JSON_Reference_Candidate': json_reference_candidate
                    })
                else:
                    bloco_b.append({
                        'Artigo_Candidate': nome_origem,
                        'JSON_Candidate': ConverterJsonTexto(text_content) if text_content else "",
                        'Actual_Label_Candidate': label_candidate,
                    })
                
            except Exception as e:
                print(f"Erro ao processar candidato {arquivo}: {e}")
    else:
        bloco_b = [{'JSON_Candidate': '', 'Article_Content': ''}] * len(bloco_a)
    
    for i in range(len(bloco_b)):
        idx_exemplo = i % len(bloco_a)
        item = {**bloco_a[idx_exemplo]}  
        
        item.update(bloco_b[i])  
        
        dados.append(item)
    
    df = pd.DataFrame(dados)
    
    if testEval:
        colunas_ordenadas = [
            'Artigo_Example', 'JSON_Example', 'Label_Example', 'Comment_Example', 'JSON_Reference_Example',
            'Artigo_Candidate', 'JSON_Candidate', 'Actual_Label_Candidate', 'JSON_Reference_Candidate'
        ]
    else:
        colunas_ordenadas = [
            'Artigo_Example', 'JSON_Example', 'Label_Example', 'Comment_Example',
            'JSON_Candidate'
        ]
    df = df[colunas_ordenadas]
    
    return df

def JsonEvalDataset(model, dataset):
    """
    Cria e preenche o dataset final para avaliação JSON, combinando o histórico empírico de calibração 
    (arquivos locais - Bloco A) com as novas predições geradas ativamente pela LLM Extratora (Bloco B).

    :param model: Objeto contendo as configurações do modelo, incluindo o caminho para os XMLs e o batch size.
    :param dataset: Objeto contendo as configurações do dataset, incluindo os caminhos para exemplos
    :return: Um DataFrame consolidado combinando os exemplos e as extrações candidatas a serem julgadas.
    """
    
    pasta_ex = dataset.pasta_ex
    pasta_ref = dataset.pasta_ref
    pasta_xmls = model.pasta_xmls
    batch_size_num = model.batch_size
    resultados_llm = dataset.resultados_llm
    testEval = dataset.testEval

    df_eval = JsonEvalFolderDataset(pasta_ex, pasta_ref, batch_size=batch_size_num, testEval=testEval)
    
    if testEval:
        print("[Modo Teste] Dataset criado com Bloco A (exemplos) e Bloco B (candidatos)")
        return df_eval

    if resultados_llm is None:
        print("Aviso: resultados_llm é None e testEval=False. JSON_Candidate ficará vazio.")
        return df_eval
    
    print(f" ->[INFO] Dataset tem {len(df_eval)} linhas")
    print(f" ->[INFO] LLM processou {len(resultados_llm)} artigos")

    resultados_ordenados = []
    for item in resultados_llm:
        dados = item.get('dados_extraidos')
        resultados_ordenados.append(ConverterJsonTexto(dados) if dados else "")
    
    coluna_gerada = []
    coluna_article_content = []
    
    for index, row in df_eval.iterrows():
        if index < len(resultados_ordenados):
            coluna_gerada.append(resultados_ordenados[index])
        else:
            coluna_gerada.append("")
            print(f" ->[AVISO] Linha {index} não tem resultado correspondente da LLM")
        
        nome_ref = row.get('Artigo_Example')
        
        if nome_ref and pasta_xmls:
            chave_ref = os.path.splitext(nome_ref)[0] 
            nome_xml = chave_ref + '.xml'
            caminho_xml = os.path.join(pasta_xmls, nome_xml)
            
            try:
                texto_artigo = extrair_metodologia_por_numero(caminho_xml, save_txt=False)
                coluna_article_content.append(texto_artigo if texto_artigo else "")
            except Exception as e:
                print(f" ->[AVISO] Erro ao extrair conteúdo de {nome_xml}: {e}")
                coluna_article_content.append("")
        else:
            coluna_article_content.append("")

    df_eval['JSON_Candidate'] = coluna_gerada
    df_eval['Article_Content'] = coluna_article_content

    return df_eval

def EvidentlyJsonEvalFolderDataset(jsonDataset):
    """
    Converte o DataFrame do Pandas para o formato nativo e tipado da biblioteca Evidently AI.
    Mapeia explicitamente quais colunas contêm texto para avaliação e quais contêm rótulos categóricos,
    preparando a estrutura de dados para a injeção do LLM avaliador.

    :param jsonDataset: O objeto contendo o DataFrame do Pandas gerado pela função `JsonEvalDataset`, já estruturado com os exemplos e candidatos.
    :return: Um objeto `Dataset` do Evidently pronto para receber as métricas e descritores do LLM.
    """

    dataset = jsonDataset.jsonDataset
    testEval = jsonDataset.testEval

    if testEval:
        dataDefinitionColumns = [
                'Artigo_Example', 'JSON_Example', 'Comment_Example', 'JSON_Reference_Example',
                'Artigo_Candidate', 'JSON_Candidate', 'JSON_Reference_Candidate'
            ]
    else:
        dataDefinitionColumns = [
                'Artigo_Example', 'JSON_Example', 'Label_Example', 'Comment_Example',
                'JSON_Candidate', 'Article_Content'
            ]

    eval_data = Dataset.from_pandas(
        dataset,
        data_definition = DataDefinition(
            text_columns = dataDefinitionColumns,
            categorical_columns = ["Label_Example"]
        )
    )
    return eval_data

def judgeJSONEval(apiKey, evidentlyDataset, prompt1, testEval=False):
    """
    Orquestra a avaliação LLM-as-a-Judge inserindo descritores no dataset do Evidently.
    Configura os templates de classificação binária, estabelece as diretrizes rígidas de penalidade 
    (Prime Directive) e mapeia as colunas necessárias para o modelo julgar alucinações.

    :param apiKey: Chave de API para autenticação no serviço de LLM (OpenRouter).
    :param evidentlyDataset: O dataset tipado gerado pelo Evidently.
    :param prompt1: O system prompt avaliador principal contendo as regras de auditoria química.
    :param testEval: Se True, anexa uma nota de ancoragem extra no prompt para ensinar o modelo a avaliar.
    :return: O `evidentlyDataset` atualizado, agora contendo as chamadas configuradas para os provedores de LLM.
    """

    if not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = apiKey

    MODELO_JUIZ = "x-ai/grok-4.1-fast"
    provider = "openrouter"

    nota_ancoragemTestEval = """
    ---
    ### DATA MAPPING & ROLES (CRITICAL):
    To perform this evaluation, you must strictly map the inputs as follows:

    1. **BLOCK A: CALIBRATION CONTEXT (Historical Data)**
    - Past Prediction: `{JSON_Example}`
    - Past Ground Truth: `{JSON_Reference_Example}`
    - Expert Verdict: `{Label_Example}` (Reasoning: `{Comment_Example}`)
    *Instruction: Observe the gap between Past Prediction and Past Ground Truth to understand the Expert Verdict.*

    2. **BLOCK B: TARGET EVALUATION (Current Task)**
    - Candidate Prediction: JSON_Candidate Column (the output you are evaluating)
    - Candidate Ground Truth: `{JSON_Reference_Candidate}`
    *Instruction: Compare these two to decide the new label.*
    
    ---
    """

    promptJsonEval = """
        You are a highly meticulous Chemist and Data Auditor.
        Your goal is to verify if a JSON_Candidate extracted by an AI is FAITHFUL
        to the original Scientific Article (Article_Content).


        ══════════════════════════════════════════════════════
        PRIME DIRECTIVE — ANTI-HALLUCINATION FOR THE JUDGE
        ══════════════════════════════════════════════════════
        YOU ARE ALSO SUSCEPTIBLE TO HALLUCINATION.


        MANDATORY RULE — QUOTE-BEFORE-CRITICIZE:
        Before making ANY claim about what JSON_Candidate contains or omits,
        you MUST copy the exact text from JSON_Candidate for that field.
        Format: [JSON QUOTE] "field_name": <exact_value_here>
        If you cannot produce an exact quote from JSON_Candidate, you CANNOT
        make a claim about that field. Silence is better than a fabricated claim.


        Similarly for Article_Content:
        Before claiming the article "says X", copy the relevant sentence.
        Format: [ARTICLE QUOTE] "...exact sentence..."


        Violations of this rule (claiming the JSON says something without quoting it)
        are more harmful than a missed error. Do not do it.


        ══════════════════════════════════════════════════════
        STEP 0 — ALL-NULL VALIDATION (run FIRST)
        ══════════════════════════════════════════════════════
        First, determine whether JSON_Candidate has ALL fields set to null/[MISSING].


        If YES (all-null JSON), apply this decision tree:
        Case A: Article_Content is an error message, empty, or non-scientific text
                → VERDICT: Good. Correct behavior.
        Case B: Article_Content discusses a non-CD material as primary focus
                (dialysed caramel, PDA-PEI dots, PEI NPs, CdSe QDs, etc.)
                → VERDICT: Good. Correct behavior.
        Case C: Article_Content clearly discusses Carbon Dots with rich data
                and the extractor returned all-null with no explanation
                → VERDICT: Bad. Systematic omission.


        Do NOT penalize an all-null JSON as hallucination. An all-null JSON contains
        no fabricated data by definition.


        ══════════════════════════════════════════════════════
        INPUT STRUCTURE
        ══════════════════════════════════════════════════════
        Block A (Calibration):
        {JSON_Example} + {Label_Example} + {Comment_Example}
        → Learn what the expert punishes and rewards.


        Block B (Audit Task):
        {Article_Content} = Source of Truth
        `JSON_Candidate` = Extraction to audit
        → No Ground Truth JSON exists. Verify against Article_Content directly.


        ══════════════════════════════════════════════════════
        ANALYSIS PROCEDURE
        ══════════════════════════════════════════════════════
        For each non-null field in JSON_Candidate:


        1. QUOTE: Copy the exact value from JSON_Candidate.
            [JSON QUOTE] "field": <value>


        2. SEARCH: Find supporting text in Article_Content.
            [ARTICLE QUOTE] "...relevant sentence..."
            If no supporting text exists → potential hallucination.


        3. VERIFY numerically (for any numerical value):
            Compare digit-by-digit: JSON says X, article says Y.
            Even a single digit difference (e.g., 421 vs 2718) = hallucination.


        4. VERIFY units: text says micrometers, JSON says nm → Bad.


        5. VERIFY attribution: does this data refer to Carbon Dots,
            or to a precursor / reference dye / other nanomaterial?


        ══════════════════════════════════════════════════════
        EVALUATION CRITERIA
        ══════════════════════════════════════════════════════
        1. HALLUCINATION (Critical — always Bad)
        A value appears in JSON but has NO textual support in Article_Content.
        Exception: scientifically deducible terms (e.g., "Hydrothermal" when
        the text says "heated in autoclave at 200°C") are valid extractions.


        2. FACTUAL ACCURACY (Critical — always Bad if violated)
        Unit mismatch (μm vs nm), wrong number, wrong reference material.


        3. TARGET MATERIAL VERIFICATION (Critical — always Bad if violated)
        Data in JSON_Candidate must refer to Carbon Dots specifically,
        NOT to: precursors, reference dyes (Rhodamine B, Quinine sulfate),
        other nanoparticles (Graphene Oxide, TiO2), Polymer Dots (PDs),
        or Copolymer Dots (PDA-PEI etc.).
        Correct extraction for non-CD data: null or [MISSING].


        4. COMPLETENESS (Important — Bad if critical data omitted)
        If Article_Content EXPLICITLY states a value (e.g., "average size 3.5 nm")
        and JSON says null → Omission error (Bad).
        If article does NOT mention the value → null is correct (Good).


        SEVERITY TIERS for omissions:
        TIER 1 (Critical): Synthesis conditions, optical peaks, size avg, QY value
        TIER 2 (Important): Surface groups, elemental composition, morphology
        TIER 3 (Minor): Alternative names, qualitative distribution descriptors
        Note: Tier 3 omissions alone should NOT flip a verdict to Bad.


        5. NULL/MISSING HANDLING
        Treat null, [MISSING], None, Null as equivalent.
        Do NOT penalize for using [MISSING] instead of null.
        Do NOT penalize [MISSING] when a property is mentioned but not quantified
        (e.g., "lifetime was measured" with no numerical value → [MISSING] is correct).


        6. MORPHOLOGY SPECIAL RULE
        "Monodisperse", "uniform", "equally distributed", "well-dispersed"
        describe size DISTRIBUTION, NOT geometric morphology.
        If the text uses only these words (no "spherical", "nanosheet", etc.),
        then [MISSING] for morphology is strictly correct. Do NOT penalize it.


        ══════════════════════════════════════════════════════
        VERDICT RULES
        ══════════════════════════════════════════════════════
        Bad: Any CRITICAL violation (hallucination, unit error, wrong material,
            Tier 1 omission, Tier 2 omission with explicit article statement).


        Good: No critical violations. Minor Tier 3 omissions alone do not trigger Bad.


        ══════════════════════════════════════════════════════
        OUTPUT FORMAT
        ══════════════════════════════════════════════════════
        Reasoning: [for each checked field: JSON QUOTE → ARTICLE QUOTE → verdict]
        Verdict: Good | Bad


        Use \n for line breaks inside strings. Keep reasoning on a single logical line.
        Escape all special characters properly.

    """
    
    additionalColumnsJsonEval = {
            "JSON_Example"              : "JSON_Example",  
            "Label_Example"             : "Label_Example",  
            "Comment_Example"           : "Comment_Example",                  
            "Article_Content"           : "Article_Content", 
    }

    additionalColumnsTestEval = {
            "JSON_Example"              : "JSON_Example",  
            "Label_Example"             : "Label_Example",  
            "Comment_Example"           : "Comment_Example",                  
            "JSON_Reference_Example"    : "JSON_Reference_Example",    
            "JSON_Reference_Candidate"  : "JSON_Reference_Candidate", 
    }

    extractionEval1 = BinaryClassificationPromptTemplate(
        pre_message=[("system", "You are a top-tier AI evaluator specializing in assessing the quality of JSON data extraction from scientific texts. Your task is to compare two JSON outputs generated by different prompts and determine which one better captures the necessary information for chemical analysis of Carbon Dots.")],
        criteria = (prompt1 + nota_ancoragemTestEval) if testEval else promptJsonEval,
        target_category = "Good",
        non_target_category = "Bad",
        uncertainty = "unknown",
        include_reasoning = True,
        include_score = False,
        additional_columns = additionalColumnsTestEval if testEval else additionalColumnsJsonEval
    )

    evidentlyDataset.add_descriptors(descriptors=[
        LLMEval("JSON_Candidate",
                template=extractionEval1,
                provider = provider,
                model = MODELO_JUIZ,
                alias=f"Evaluation_Modelo_1",
                additional_columns = additionalColumnsTestEval if testEval else additionalColumnsJsonEval,
        )
    ])
    return evidentlyDataset

# endregion

# region [Prompt create]
def create_prompt(model, prompt_sistema):
    """
    Comunica-se com a API de inferência (OpenRouter) para gerar instruções baseadas em um prompt de sistema.
    Utilizada primariamente no pipeline paralelo de 'Auto-Prompting', onde um modelo atua como 
    Engenheiro de Prompts e cria diretrizes de extração otimizadas de forma autônoma.

    :param model: Identificador do modelo gerador (ex: 'openrouter/tngtech/deepseek-r1t2-chimera').
    :param prompt_sistema: A instrução inicial base (system prompt) que guia a criação do novo prompt mestre.
    :return: Uma string contendo o prompt gerado, estruturado e otimizado pelo modelo de linguagem.
    """
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {model.apiKey}",
        "Content-Type": "application/json"
    }

    modelo = model.modelo
    payload = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": prompt_sistema},
        ],
        "temperature": 0.5
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == requests.codes.ok:
            conteudo = response.json()['choices'][0]['message']['content']
            return conteudo
        else:
            print(f"Erro na API ({response.status_code}): {response.text}")
            return None
            
    except Exception as e:
        print(f"Erro de conexão: {e}")
        return None

# endregion

# region [Prompt eval]
def EvidentlyPromptEval(dict_prompts_gerados):
    """
    Converte o dicionário contendo os prompts gerados pelos diferentes modelos em um formato tabular 
    e o tipifica como um 'Dataset' nativo da biblioteca Evidently AI. Prepara a infraestrutura de dados 
    para a etapa de avaliação comparativa dos prompts candidatos.

    :param dict_prompts_gerados: Dicionário contendo os nomes dos modelos como chaves e os prompts gerados como valores.
    :return: Um objeto Dataset do Evidently, mapeado com as colunas prontas para a injeção do LLM Juiz.
    """
    rows = []
    
    for modelo, prompt in dict_prompts_gerados.items():
        if prompt is None:
            continue
        
        row = {
            "model": modelo,
            "prompt_generated": prompt,
        }

        rows.append(row)
    
    eval_data = Dataset.from_pandas(
        pd.DataFrame(rows),
        data_definition=DataDefinition(
            text_columns=["model", "prompt_generated"]
        )
    )
    return eval_data

def judge_evidently_prompt_eval(apiKey, evidently_dataset):
    """
    Configura e executa a avaliação automatizada (LLM-as-a-Judge) focada estritamente na qualidade dos prompts.
    Aplica templates de classificação para julgar simultaneamente as quatro métricas fundamentais do projeto:
    Structure, Scientific Relevance, Constraints e Robustness, acoplando esses descritores ao dataset.

    :param apiKey: Chave de autenticação (API Key) para autorizar a chamada ao modelo juiz no OpenRouter.
    :param evidently_dataset: O dataset estruturado gerado previamente pela função `EvidentlyPromptEval`.
    :return: O dataset do Evidently atualizado com os resultados completos (veredito e raciocínio) de cada métrica.
    """

    if not os.environ.get("OPENROUTER_API_KEY"):
         os.environ["OPENROUTER_API_KEY"] = apiKey

    MODELO_JUIZ = "openrouter/xiaomi/mimo-v2-flash:free"
    provider = "openrouter"

    prompt_base = """
        You are an Expert Prompt Engineer with a strong background in Material Science.

        Your goal is to evaluate a System Prompt generated by another AI.
        The generated prompt is intended to extract scientific data about 'Carbon Dots' from papers into JSON format.
        Crucially, it must capture the specific parameters that a Chemist would expect to fully characterize the material.
    """
    structure = BinaryClassificationPromptTemplate(
        pre_message=[prompt_base],
        criteria = """
            A STRICT JSON prompt is any instruction that explicitly:
            - Demands the output format to be a valid JSON object.
            - Defines the expected Schema, Keys, or Structure (e.g., specifying fields like 'precursor', 'method', 'yield').
            - Disallows conversational filler or explicitly asks for "raw data" / "only JSON".
            -The prompt explicitly demands valid JSON output AND defines the specific keys/schema to use. It forbids conversational text.

            A VAGUE prompt is any instruction that:
            - Asks for a summary, list, or free-text description.
            - Mentions "extract data" without specifying the exact JSON keys or format.
            - Allows the model to output explanations or chatty introductions alongside the data.
            -The prompt asks for summaries, lists, or plain text. It fails to specify the exact JSON structure or allows conversational filler."
        """,
        target_category = "STRICT JSON",
        non_target_category = "VAGUE",
        uncertainty = "unknown",
        include_reasoning = True,
    )

    scientific_relevance = MulticlassClassificationPromptTemplate(
        pre_message=[prompt_base],
        criteria = """
            Evaluate if the prompt extracts all necessary fields for a chemist to reproduce the synthesis or understand the properties of Carbon Dots.
        """,
        category_criteria = {
            "1" : """
                - Focuses primarily on metadata (Authors, Affiliations, Dates) rather than chemistry.
                - Fails to identify the core precursors or the synthesis method.
                - Is too vague to understand what material is being synthesized.
            """,
            "2" : """
                - Requests to identify the main material ('Carbon Dots') and the general technique (e.g., 'Hydrothermal').
                - Misses critical numerical conditions (e.g., forgets to ask for 'Temperature' or 'Reaction Time').
                - Requests lists of materials but lacks specificity needed for a lab experiment.
            """,
            "3" : """
                - Explicitly asks for all critical synthesis parameters for Carbon Dots.
                - Demands quantitative conditions necessary for reproduction (e.g., Temperature, Time, pH, Quantum Yield, ...).
                - Ensures the extraction of the material's identity and properties.
            """
        },
        uncertainty = "UNKNOWN",
        include_reasoning = True,
        include_score = False
    )

    constraints = MulticlassClassificationPromptTemplate(
        pre_message=[prompt_base],
        criteria = """
            "Does it include rules to avoid hallucination (e.g. \"Return null if not found\")?"
        """,
        category_criteria = {
                "1": "No negative constraints. The model might hallucinate or guess missing values.",
                "2": "Contains generic phrases like \"Be truthful\" or \"Don't make up info\".",
                "3": "Contains explicit handling rules (e.g. \"If some information is missing, return null\", \"Convert K to C\")."
        },
        uncertainty = "UNKNOWN",
        include_reasoning = True,
        include_score = False
    )

    robustness = MulticlassClassificationPromptTemplate(
        pre_message=[prompt_base],
        criteria = """
            "Evaluate the prompt's robustness against model misinterpretation or confusion. 
            Is it concise enough not to confuse the model?"

            Here is the length of the prompt to consider: 
            {Prompt_Length}
        """,
        additional_columns={"Prompt Length" : "Prompt_Length"},
        category_criteria = {
                "1": "Confusing, contradictory instructions, or excessive boilerplate text that wastes tokens.",
                "2": "Clear instructions but verbose.",
                "3": "Highly efficient, concise, and uses specific keywords that steer the model effectively."
        },
        uncertainty = "UNKNOWN",
        include_reasoning = True,
        include_score = False
    )
    
    evidently_dataset.add_descriptors(descriptors=[
        TextLength(column_name="prompt_generated", alias="Prompt_Length"),
        LLMEval("prompt_generated",
                template=structure,
                provider = provider,
                model = MODELO_JUIZ,
                alias="Structure_Evaluation"
        ),
        LLMEval("prompt_generated",
                template=scientific_relevance,
                provider = provider,
                model = MODELO_JUIZ,
                alias="Scientific_Relevance"
        ),
        LLMEval("prompt_generated",
                template=constraints,
                provider = provider,
                model = MODELO_JUIZ,
                alias="Constraints_Evaluation"
        ),
        LLMEval("prompt_generated",
                template=robustness,
                additional_columns={"Prompt_Length": "Prompt_Length"},
                provider = provider,
                model = MODELO_JUIZ,
                alias="Robustness_Evaluation"
        ),
    ])

    return evidently_dataset
# endregion

# region [Metrics]
def calculateMetrics(df, coluna_predicao, coluna_real, nome_modelo):
    """
    Calcula os indicadores estatísticos de desempenho computacional e qualitativo do pipeline.
    Cruza os dados da predição do modelo (JSON extraído ou nota do prompt) com o Gabarito (Ground Truth) 
    para derivar métricas de eficácia (como Acertos, Erros, ou Taxa de Concordância).

    :param df: DataFrame pandas contendo os resultados compilados do Evidently.
    :param coluna_predicao: Nome da coluna contendo as saídas ou vereditos gerados pela IA.
    :param coluna_real: Nome da coluna contendo a verdade absoluta estabelecida pelo especialista (Golden Dataset).
    :param nome_modelo: Identificador do modelo analisado, utilizado para categorizar a saída dos logs/gráficos.
    :return: Um dicionário ou relatório resumido com os indicadores numéricos e percentuais calculados.
    """
    df_clean = df.dropna(subset=[coluna_predicao, coluna_real])

    df_clean = df_clean[df_clean[coluna_predicao] != "unknown"]
    
    y_true = df_clean[coluna_real].values
    y_pred = df_clean[coluna_predicao].values
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='Good', zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label='Good', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label='Good', zero_division=0)
    
    labels = ['Bad', 'Good'] 
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    cm_df = pd.DataFrame(cm, 
                         index=['Real: Bad', 'Real: Good'], 
                         columns=['Pred: Bad', 'Pred: Good'])
    
    total = len(df_clean)
    acertos = (y_true == y_pred).sum()
    erros = total - acertos

    print(f"Model: {nome_modelo}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f" ->Success: {acertos} / {total}")
    print(f" ->Errors: {erros} / {total}")
    print("Confusion Matrix:")
    print(cm_df)
    print(20*"=")
    
    return {
        'modelo': nome_modelo,
        'total_samples': int(total),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'acertos': int(acertos),
        'erros': int(erros),
        'confusion_matrix': cm.tolist(),
    }

def countGoodBad(df):
    """
    Função agregadora utilitária para contagem de rótulos categóricos resultantes da auditoria do Juiz.
    Varre o DataFrame de resultados e contabiliza a distribuição quantitativa das extrações classificadas 
    como 'Good' (Boas), 'Bad' (Ruins) ou 'Unknown' (Desconhecidas/Malformadas). 
    Esta função alimenta diretamente a plotagem dos gráficos de Benchmarking.

    :param df: DataFrame pandas contendo a coluna de veredito final da auditoria.
    :return: Um dicionário ou série (pandas) contendo as contagens absolutas para cada categoria de avaliação.
    """

    dfClean = df[
        (df['Evaluation_Modelo_1'].str.lower() != "unknown") 
    ]

    total = len(dfClean)
    contModel1 = dfClean["Evaluation_Modelo_1"].value_counts()

    for label, count in contModel1.items():
        print(f" [{label}]: {count}/{total}")

# endregion

# region [Main]
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    promptJsonEval = """
        Act as a Senior Prompt Engineer and Data Scientist specializing in NLP for Material Science.

        Project Context: 
            I am building a pipeline to extract structured data about "Carbon Dots" (nanomaterials) from scientific papers into a strict JSON schema.
            I have a "Golden Dataset" (manually extracted JSONs serving as Ground Truth) and a "Generated Dataset" (JSONs extracted by an LLM).
            I need to set up an "LLM-as-a-Judge" system to evaluate the quality of the extraction using the Evidently Python library.

        Your Task: 
            Write a highly robust, Chain-of-Thought (CoT) based **System Prompt** (to be used as the `criteria` argument in Evidently).

        **Crucial Constraint:**
        Include the list of labels (Good/Bad) or their specific definitions in the generated prompt text and focus entirely on the **reasoning logic** and the **comparison rules**.

        Requirements for the System Prompt:

        1. **Role:** The Judge must act as a strict Chemist and Data Auditor.

        2. **Input Structure:**
        - **Block A (Learning Context):** Historical example (`[[Json_Example]]`, `[[Expert_Label]]`, `[[Expert_Comment]]`). Use this strictly for calibration.
        - **Block B (Current Task):** Compare `[[Json_Reference]]` (Ground Truth) vs `[[Json_Candidate]]`.

        3. **Evaluation Criteria (The "How-To"):**

        - **Factual Accuracy (Quantitative):** - Numerical values (e.g., "5.4 nm", "80%") and Units MUST match the Reference exactly or be within a scientifically negligible margin. 
            - *Strict Rule:* If the Reference has a number (e.g., "50%"), and the Candidate has a different number (e.g., "90%"), this is a Critical Error.

        - **Descriptive Accuracy (Qualitative - FLEXIBLE):**
            - **Information Gain:** If the Candidate includes valid chemical adjectives or functional descriptions NOT present in the Reference (e.g., Reference: "Carbon Dots"; Candidate: "Fluorescent Carbon Dots with peroxidase activity"), treat this as **CORRECT/ENRICHMENT**, not hallucination, provided it does not explicitly contradict the Reference.
            - **Semantic Equivalence:** "N-doped" == "Nitrogen-doped". "Hydrothermal" == "Hydrothermal synthesis".

        - **Hallucination Check (Fabrication):**
            - **PENALIZE ONLY IF:** The Candidate invents **Quantitative Data** (numbers/formulas) that are missing in the Reference, OR if it adds descriptions that imply a completely different material (e.g., saying "Graphene Oxide" instead of "Carbon Dots").
            - **DO NOT PENALIZE:** If the Candidate fills a `null` field in the Reference with scientifically plausible data (Discovery).

        - **Omission Check:** - Penalize if the Candidate has `null` where the Reference has specific data.

        4. **Analysis Steps:**
        - Step 1: Analyze Block A to understand the human expert's tolerance.
        - Step 2: Compare Block B. Distinguish between **"Contradiction"** (Bad) and **"Enrichment"** (Good).
        - Step 3: Determine severity.

        Output: Generate only the System Prompt text.

        JSON_Schema's fields for Reference and Candidate:
        ```
            carbon_dots: {
                general_info: {
                    definition,
                    iupac_name,
                    alternative_names
                },
                synthesis: {
                    method: {
                        type, 
                        conditions, 
                        catalysts
                    },
                    precursors: {
                        source, 
                        dopants
                    },
                    post_treatment: 
                },
                properties: {
                    optical: {
                        quantum_yield: {
                            value, 
                            reference_material
                        },
                        excitation_max,
                        emission_max,
                        fluorescence_lifetime
                    },
                    structural: {
                        size: {
                            avg, 
                            distribution
                        },
                        morphology,
                        crystallinity
                    },
                    chemical: {
                        surface_groups,
                        elemental_composition: {C, N, O}
                    }
                },
                applications: [
                    {
                    domain:,
                    performance:,
                    comparative_advantage
        ```
        5. **Note:** 
        - Don't use quotation marks("") or curly braces({}) in the prompt text. Use placeholders like `[[Json_Example]]`, `[[value]]` instead of `{JSON_Example}`, `{value}` or "JSON_Example", "value" to prevent parsing issues.
    """

    modelExtractorTest = modelExtractor(modelo="minimax/minimax-m2.5", 
                                        apiKey="SUA_API_KEY_AQUI",
                                        batch_size=5, 
                                        pasta_xmls="carbon_dots")
    
    jsonDataset = evalDataset(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset', 'dataset_examples'),
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset', 'GoldenDataset'))

    jsonDataset.resultados_llm, _ = processar_lote_artigos(modelExtractorTest) 

    print("Preparando dataset para avaliação JSON...")

    jsonDataset.jsonDataset = JsonEvalDataset(modelExtractorTest,
                                                jsonDataset)
                    
    jsonDataset.evidentlyJsonDataset = EvidentlyJsonEvalFolderDataset(jsonDataset)
    
    print("Iniciando avaliação JSON com Evidently...")
    
    evalData = judgeJSONEval(modelExtractorTest.apiKey, jsonDataset.evidentlyJsonDataset, None)
    evalData_df = evalData.as_dataframe()
    evalData_df.to_csv(os.path.join("Dataset", "JsonEvalDataset_Judge", f"jsonEvalDataset_{timestamp}.csv"), index=False, sep=';', encoding='utf-8-sig')

    print("Gerando Report...")
    report = Report([
        TextEvals()
    ])
    my_eval = report.run(evalData)
    print("Finalizando...")

    print("Resultados da Avaliação JSON:")
    countGoodBad(evalData.as_dataframe())

# endregion