import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from tqdm import tqdm
import re
import time
import uuid
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. Configuración (Adaptada para el nuevo dataset) ---
INTERNAL_LLM_BASE_URL = "********"
INTERNAL_LLM_API_KEY = "dummy-key"
MODEL_NAME = "gpt-4.1-2025-04-14"
USER_ALIAS = "****"
NAMESPACE = "sdk_test"
MAX_WORKERS = 16

# --- MODIFICADO: Rutas para el nuevo dataset ---
DATA_DIR = "./dignosis_dataset/"
LABEL_FILE = os.path.join(DATA_DIR, "label.txt")
LOG_SESSIONS_DIR = os.path.join(DATA_DIR, "log_analysis_runs")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SAMPLES_PER_CATEGORY = 20 # Muestras por categoría para la visualización

# RAW_LABEL_TO_CATEGORY y FAULT_CATEGORIES se descubrirán dinámicamente

# --- 2. LLM 调用函数 ---
def call_llm(prompt: str, system_prompt: str, session_id: str, is_json_output: bool = False) -> str:
    try:
        client = OpenAI(base_url=INTERNAL_LLM_BASE_URL, api_key=INTERNAL_LLM_API_KEY)
    except Exception as e:
        raise ConnectionError(f"无法初始化OpenAI客户端: {e}")
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                extra_headers={"x-extra-session-id": session_id, "x-extra-user-alias": USER_ALIAS, "x-extra-namespace": NAMESPACE},
                model=MODEL_NAME,
                messages=messages,
                response_format={"type": "json_object"} if is_json_output else None,
                temperature=0.0)
            return completion.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                tqdm.write(f"LLM调用失败 (尝试 {attempt+1}/{max_retries}) for {session_id}. 错误: {e}. 5秒后重试...")
                time.sleep(5)
            else:
                tqdm.write(f"LLM调用最终失败 for {session_id}. 错误: {e}")
                raise
    return ""

# --- 3. 绘图风格配置 ---
def setup_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman'], 'axes.labelsize': 16,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
        'axes.titlesize': 18, 'figure.dpi': 300, 'savefig.format': 'pdf',
    })
    print("双栏论文绘图风格已配置。")

# --- 4. 辅助函数 ---
def log_parsing_helper(log_df: pd.DataFrame) -> dict:
    unique_templates = set()
    for _, row in log_df.iterrows():
        log_message = str(row.get('msg', ''))
        if not log_message: continue
        template_parts = ['<*>' if re.search(r'\d', token) else token for token in log_message.split()]
        unique_templates.add(' '.join(template_parts))
    return {"unique_templates": list(unique_templates)}

# get_logs_for_task_helper ya no es necesario

# --- 5. 数据加载与处理 (REESCRITO para el nuevo dataset) ---
def load_session_data_new(n_samples):
    print("--- [Data Preparation] Starting for new dataset structure...")
    
    # 1. Leer todas las tareas desde label.txt
    tasks = []
    if not os.path.exists(LABEL_FILE):
        raise FileNotFoundError(f"El archivo de etiquetas no se encuentra en: {LABEL_FILE}")
        
    with open(LABEL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ',' not in line: continue
            session_id, category = line.split(',', 1)
            # Verificar si el archivo de log correspondiente existe
            if os.path.exists(os.path.join(LOG_SESSIONS_DIR, f"{session_id}.log")):
                tasks.append({'session_id': session_id, 'category': category})
            else:
                print(f"Advertencia: Archivo de log para la sesión {session_id} no encontrado. Omitiendo.")

    if not tasks:
        raise ValueError("No se encontraron tareas válidas con archivos de log correspondientes.")

    # 2. Muestrear n_samples por categoría usando pandas
    tasks_df = pd.DataFrame(tasks)
    sampled_tasks_df = tasks_df.groupby('category').head(n_samples)
    sampled_tasks = sampled_tasks_df.to_dict('records')
    
    session_content = {}
    print(f"Extrayendo logs para {len(sampled_tasks)} tareas muestreadas...")
    for task in tqdm(sampled_tasks, desc="Procesando Tareas Muestreadas"):
        session_id, category = task['session_id'], task['category']
        log_file_path = os.path.join(LOG_SESSIONS_DIR, f"{session_id}.log")
        
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                log_lines = [line.strip() for line in f if line.strip()]
            
            if not log_lines: continue

            # Preparar datos para el helper de parsing
            temp_df = pd.DataFrame(log_lines, columns=['msg'])
            parsed_data = log_parsing_helper(temp_df)

            session_content[session_id] = {
                "category": category,
                "raw_logs": " ".join(log_lines),
                "parsed_templates": " ".join(parsed_data['unique_templates'])
            }
        except Exception as e:
            print(f"Error al procesar el archivo {log_file_path}: {e}")

    print(f"Carga de datos completa. Total de {len(session_content)} sesiones válidas cargadas.")
    return session_content

# --- 6. LLM并发总结函数 (Corregido) ---
def generate_llm_summaries_concurrently(session_data):
    print(f"\n--- [LLM Summarization] Empezando a resumir {len(session_data)} sesiones concurrentemente...")
    
    prompt_template = """Context:
Give a summary of the following system logs in 500 words,
it will later be used to classify one kind of anomaly.
The summary needs and only needs to contain all operations
performed by the system, wrapped with [] and separated
by;
For example: [initializing the authorizer];[registering to a
cluster];[flush working memtables]
Input:
{Input_Logs}"""

    system_prompt = "You are an expert system reliability engineer summarizing logs."

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_key = {
            executor.submit(
                call_llm,
                # *** CORRECCIÓN CRÍTICA ***
                # Usar el método format para rellenar la plantilla con los logs reales
                prompt=prompt_template.format(Input_Logs=data['raw_logs']),
                system_prompt=system_prompt,
                session_id=key
            ): key
            for key, data in session_data.items()
        }
        
        for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="LLM Summarizing"):
            key = future_to_key[future]
            try:
                summary = future.result()
                session_data[key]['llm_summary'] = summary
            except Exception as e:
                tqdm.write(f"Fallo al generar resumen para la sesión {key}: {e}")
                session_data[key]['llm_summary'] = ""

    print("--- [LLM Summarization] Completo. ---")
    return session_data

# --- 7. 嵌入、降维与可视化 ---
def get_embeddings(model, session_data, mode):
    print(f"Generando embeddings basados en '{mode}'...")
    texts = [data.get(mode, "") for data in session_data.values()]
    labels = [data['category'] for data in session_data.values()]
    
    filtered_texts, filtered_labels = [], []
    for text, label in zip(texts, labels):
        if text: # Solo incluir si el texto no está vacío
            filtered_texts.append(text)
            filtered_labels.append(label)

    if not filtered_texts: 
        print(f"Advertencia: No se encontraron textos para el modo '{mode}'.")
        return None, None
        
    embeddings = model.encode(filtered_texts, show_progress_bar=True)
    return embeddings, filtered_labels

def reduce_dimensions(embeddings):
    print("Usando t-SNE para reducción de dimensionalidad...")
    # La perplejidad debe ser menor que el número de muestras
    perplexity = min(30, len(embeddings) - 1)
    if perplexity <= 0:
        print("No hay suficientes datos para ejecutar t-SNE.")
        return None
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000, init='pca', learning_rate='auto')
    return tsne.fit_transform(embeddings)

def plot_three_panel_embeddings(data_raw, data_parsed, data_llm, labels, filename):
    # Crear DataFrames para cada tipo de dato
    df_raw = pd.DataFrame({'x': data_raw[:, 0], 'y': data_raw[:, 1], 'category': labels})
    df_parsed = pd.DataFrame({'x': data_parsed[:, 0], 'y': data_parsed[:, 1], 'category': labels})
    df_llm = pd.DataFrame({'x': data_llm[:, 0], 'y': data_llm[:, 1], 'category': labels})
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    palette = sns.color_palette("deep", n_colors=len(set(labels)))
    
    # Panel (a): Raw Logs
    sns.scatterplot(data=df_raw, x='x', y='y', hue='category', palette=palette, style='category',
                    s=80, alpha=0.85, edgecolor='k', linewidth=0.6, ax=axes[0], legend=False)
    axes[0].set_title('(a) Raw Log Sessions', weight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1')
    axes[0].set_ylabel('t-SNE Dimension 2')
    
    # Panel (b): Log Templates
    sns.scatterplot(data=df_parsed, x='x', y='y', hue='category', palette=palette, style='category',
                    s=80, alpha=0.85, edgecolor='k', linewidth=0.6, ax=axes[1], legend=False)
    axes[1].set_title('(b) Log Template Sessions', weight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1')
    axes[1].set_ylabel('')
    axes[1].set_yticklabels([])
    
    # Panel (c): LLM Summaries
    sns.scatterplot(data=df_llm, x='x', y='y', hue='category', palette=palette, style='category',
                    s=80, alpha=0.85, edgecolor='k', linewidth=0.6, ax=axes[2], legend=True)
    axes[2].set_title('(c) LLM-Summarized Sessions', weight='bold')
    axes[2].set_xlabel('t-SNE Dimension 1')
    axes[2].set_ylabel('')
    axes[2].set_yticklabels([])

    # Mover la leyenda a la parte inferior
    handles, labels_legend = axes[2].get_legend_handles_labels()
    axes[2].get_legend().remove()
    fig.legend(handles, labels_legend, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=len(set(labels)), title='Fault Category', title_fontsize='15')

    plt.subplots_adjust(bottom=0.25, wspace=0.18)
    plt.savefig(filename, bbox_inches='tight')
    print(f"El gráfico final de tres paneles ha sido guardado en: {filename}")
    plt.show()

# --- 8. 主程序 ---
if __name__ == "__main__":
    setup_plot_style()
    
    try:
        session_data = load_session_data_new(n_samples=SAMPLES_PER_CATEGORY)
        
        if not session_data:
            print("Error: No se pudieron cargar o procesar datos de sesión, el programa terminará.")
            sys.exit(1)

        session_data_with_summaries = generate_llm_summaries_concurrently(session_data)
        
        print(f"\nCargando modelo de embedding: '{EMBEDDING_MODEL_NAME}'...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        embeddings_raw, labels_raw = get_embeddings(embedding_model, session_data_with_summaries, mode='raw_logs')
        embeddings_parsed, _ = get_embeddings(embedding_model, session_data_with_summaries, mode='parsed_templates')
        embeddings_llm, _ = get_embeddings(embedding_model, session_data_with_summaries, mode='llm_summary')

        # Verificar que todos los embeddings se generaron correctamente
        if all(e is not None for e in [embeddings_raw, embeddings_parsed, embeddings_llm]):
            reduced_raw = reduce_dimensions(embeddings_raw)
            reduced_parsed = reduce_dimensions(embeddings_parsed)
            reduced_llm = reduce_dimensions(embeddings_llm)
            
            if all(r is not None for r in [reduced_raw, reduced_parsed, reduced_llm]):
                plot_three_panel_embeddings(
                    reduced_raw,
                    reduced_parsed,
                    reduced_llm,
                    labels_raw, # labels_raw es la lista de etiquetas filtrada y correcta
                    'visualization_three_panel.pdf'
                )
            else:
                print("Error: Falló la reducción de dimensionalidad para una o más representaciones.")
        else:
            print("Error: No se pudieron crear los embeddings para una o más representaciones, no se puede generar el gráfico.")

    except (FileNotFoundError, ValueError) as e:
        print(f"Se ha producido un error durante la preparación de los datos: {e}")
    except Exception as e:
        print(f"Se ha producido un error inesperado: {e}")