import pandas as pd
import json
import os
import re
import time
import uuid
import openai
import numpy as np
import sys
import ast
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Union
from collections import Counter, deque
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from itertools import combinations
from scipy.spatial.distance import cosine
from tqdm import tqdm

# --- 1. 全局配置 ---
INTERNAL_LLM_BASE_URL = "*******"
INTERNAL_LLM_API_KEY = "dummy-key"
MODEL_NAME = "gpt-4.1-2025-04-14"
USER_ALIAS = "****"
NAMESPACE = "sdk_test"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LOG_DATA_DIR = "LogKG/data/CMCC_case/"
CONFIG_FILE = "LogKG/data/config.json"
ENABLE_MEMORY = True

FAULT_CATEGORY_DESCRIPTIONS = {
    "Categories": "Description",
    "AMQP": "AMQP server unreachable",
    "Mysql": "Mysql lost connection",
    "CreateErrorNovaConductor": "Nova-conductor Lost Connection",
    "Down": "Computing node down",
    "CreateErrorFlavor": "Flavor Disk Space Insufficient",
    "CreateErrorLinuxbridgeAgent": "Linuxbridge-agent Anomalies",
    "Normal": "Normal: The system is operating normally without any of the defined faults."
}

FAULT_CATEGORIES = list(FAULT_CATEGORY_DESCRIPTIONS.keys())
LABEL_TO_INT_MAP = {label: i for i, label in enumerate(FAULT_CATEGORIES)}
INT_TO_LABEL_MAP = {i: label for i, label in enumerate(FAULT_CATEGORIES)}
DEFAULT_LABEL = "Normal"
DEFAULT_LABEL_INT = LABEL_TO_INT_MAP[DEFAULT_LABEL]

MAX_WORKERS = 16
SIMILARITY_THRESHOLD = 0.95
MEMORY_MAX_BYTES = 100 * 1024 * 1024
HIGH_PRIORITY_LEVELS = ['ERROR', 'FATAL', 'CRITICAL', 'WARNING']
LAST_RESORT_FAULT_CATEGORY = "AMQP"

# --- 2. LLM 和 Embedding 函数 ---
def call_llm(prompt: str, system_prompt: str, session_id: str, is_json_output: bool = True) -> Union[str, Dict, List]:
    """
    调用 LLM。如果 is_json_output 为 True，它会从响应中提取并解析 JSON。
    """
    try:
        client = OpenAI(base_url=INTERNAL_LLM_BASE_URL, api_key=INTERNAL_LLM_API_KEY)
    except Exception as e:
        raise ConnectionError(f"Failed to initialize internal OpenAI client: {e}")
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    response_format = {"type": "json_object"} if is_json_output else None
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                extra_headers={"x-extra-session-id": session_id, "x-extra-user-alias": USER_ALIAS, "x-extra-namespace": NAMESPACE},
                model=MODEL_NAME,
                messages=messages,
                response_format=response_format,
            )
            
            response_content = completion.choices[0].message.content
            
            if not is_json_output:
                return response_content.strip()

            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Failed to parse extracted JSON. Extracted string: '{json_str}'. Original content: '{response_content}'") from e
            else:
                try:
                    return json.loads(response_content)
                except json.JSONDecodeError as e:
                    raise ValueError(f"LLM response is not valid JSON and does not contain a JSON block. Content: '{response_content}'") from e
        
        except Exception as e:
            if attempt < max_retries - 1:
                tqdm.write(f"\nLLM call failed (attempt {attempt+1}/{max_retries}). Retrying in 5s... Error: {e}")
                time.sleep(5)
            else:
                raise
    return "" if not is_json_output else {}

def get_embedding(model: SentenceTransformer, texts: List[str]) -> Union[np.ndarray, None]:
    if not texts:
        return None
    try:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.mean(embeddings, axis=0)
    except Exception as e:
        print(f"Warning: Local Embedding generation failed: {e}")
        return None

# --- 3. 核心类 ---
class LogParser:
    def process_logs(self, log_df: pd.DataFrame) -> Dict:
        templates, template_to_id = {}, {}
        high_priority_template_ids, normal_template_ids = set(), set()
        parsed_logs = []
        template_counter = 0
        for _, row in log_df.iterrows():
            template_text = row.get('EventTemplate', '')
            log_level = str(row.get('Level', '')).upper()
            if not template_text or pd.isna(template_text):
                continue
            try:
                variables = ast.literal_eval(row.get('ParameterList', '[]'))
            except (ValueError, SyntaxError):
                variables = []
            if template_text not in template_to_id:
                template_id = f"T{template_counter}"
                templates[template_id] = template_text
                template_to_id[template_text] = template_id
                template_counter += 1
            template_id = template_to_id[template_text]
            if log_level in HIGH_PRIORITY_LEVELS:
                high_priority_template_ids.add(template_id)
            else:
                normal_template_ids.add(template_id)
            parsed_logs.append({'template_id': template_id, 'variables': variables})
        normal_template_ids -= high_priority_template_ids
        return {
            'templates': templates,
            'parsed_logs': parsed_logs,
            'high_priority_template_ids': list(high_priority_template_ids),
            'normal_template_ids': list(normal_template_ids)
        }

class DissimilaritySampler:
    def __init__(self, log_parser: LogParser, embedding_model: SentenceTransformer, log_data_dir: str):
        self.log_parser = log_parser
        self.embedding_model = embedding_model
        self.log_data_dir = log_data_dir
        self.session_vectors = {}

    def _get_session_embedding(self, session_id: str) -> Union[np.ndarray, None]:
        if session_id in self.session_vectors:
            return self.session_vectors[session_id]
        log_file_path = os.path.join(self.log_data_dir, f"{session_id}.csv")
        try:
            log_df = pd.read_csv(log_file_path, on_bad_lines='skip')
        except FileNotFoundError:
            return None
        parsed_data = self.log_parser.process_logs(log_df)
        template_ids = parsed_data['high_priority_template_ids'] or list(parsed_data['templates'].keys())
        templates_to_embed = [parsed_data['templates'][tid] for tid in template_ids]
        if not templates_to_embed:
            return None
        session_vector = get_embedding(self.embedding_model, templates_to_embed)
        self.session_vectors[session_id] = session_vector
        return session_vector

    def select_most_dissimilar_pair(self, session_ids: List[str]) -> Tuple[str, str]:
        if len(session_ids) < 2:
            return tuple(session_ids)
        print(f"--- [Sampler] Vectorizing {len(session_ids)} sessions for dissimilarity check...")
        for sid in tqdm(session_ids, desc="Vectorizing Sampler Sessions", ncols=100, leave=False):
            self._get_session_embedding(sid)
        valid_sessions = {sid: vec for sid, vec in self.session_vectors.items() if sid in session_ids and vec is not None}
        if len(valid_sessions) < 2:
            print("--- [Sampler] Not enough valid sessions to compare. Returning first two.")
            return tuple(session_ids[:2])
        max_distance = -1
        best_pair = (None, None)
        for sid1, sid2 in combinations(valid_sessions.keys(), 2):
            distance = cosine(valid_sessions[sid1], valid_sessions[sid2])
            if distance > max_distance:
                max_distance = distance
                best_pair = (sid1, sid2)
        print(f"--- [Sampler] Selected pair {best_pair} with max dissimilarity score: {max_distance:.4f}")
        return best_pair

class L0_SummarizationAgent:
    def run(self, category: str, example_sessions: List[str], session_id: str) -> str:
        system_prompt = "You are a master analyst. Your job is to find the common ground between two examples of the same fault and create an initial summary. Focus only on shared characteristics."
        code_block = "```"
        prompt = f"""### Task: Infer Potential Fault Characteristics
You are given two examples for the fault category **{category}**.
**Basic description of this fault category:** {FAULT_CATEGORY_DESCRIPTIONS[category]}
**Example 1:**
{code_block}
{example_sessions[0]}
{code_block}
**Example 2:**
{code_block}
{example_sessions[1]}
{code_block}
### Your Output
Based on the commonalities between these examples, infer the **potential fault characteristics** of this category. Focus on identifying the **shared and recurring traits** that could define this type of fault. Avoid mentioning details that only appear in one example.
**Format:**
#### {category}
- **Possible Symptom Features:** [Summarize the likely common external manifestations]
- **Possible Diagnostic Clues:** [Summarize the likely shared log signatures, variables, or error patterns]"""
        try:
            return call_llm(prompt, system_prompt, session_id, is_json_output=False)
        except Exception as e:
            tqdm.write(f"\n--- [L0 Agent] Failed to create draft for '{category}': {e}")
            return f"#### {category}\n- **Error:** Failed to generate draft summary."

class L0_5_RefinementAgent:
    def run(self, all_draft_guides_str: str, all_examples_str: str, session_id: str) -> str:
        system_prompt = "You are a 'Chief SRE Diagnostician'. Your mission is to refine a full report of draft diagnostic guides, making each one uniquely identifiable and distinct from the others."
        code_block = "```"
        prompt = f"""### Mission: Differentiate Fault Categories by Key Distinctions
You are given a full report containing draft diagnostic guides for **all** fault categories, along with a reference example for each. Your task is to **rewrite the entire report**, making each guide highlight the **most distinctive features** that set its category apart.
**Basic description of fault categories:** {FAULT_CATEGORY_DESCRIPTIONS}
**1. Full Report of Draft Guides to Refine:**
{code_block}markdown
{all_draft_guides_str}
{code_block}
**2. All Reference Examples (one for each category):**
{code_block}
{all_examples_str}
{code_block}
### Your Output
Rewrite and output the **complete, final report** in a single markdown block. For each category, you must:
1. Identify the **most discriminative differences** (not just common traits).
2. Rewrite the guide to emphasize these distinctive signatures.
Your final report must keep the same structure, containing:
- **Refined Symptom Summary:** Focused on outward signs that clearly differentiate this fault.
- **Refined Diagnostic Instructions:** Focused on unique log signatures that make this fault type easy to distinguish."""
        try:
            return call_llm(prompt, system_prompt, session_id, is_json_output=False)
        except Exception as e:
            tqdm.write(f"\n--- [L0.5 Agent] Global refinement failed: {e}")
            return all_draft_guides_str

class L1_CandidateGenerationAgent:
    def run(self, enriched_logs: List[str], session_id: str, final_guides: Dict[str, str]) -> Dict:
        system_prompt = "You are an AI diagnostic assistant. Your task is to analyze system logs and propose the top 3 most likely fault categories based on a set of expert-written guides."
        demonstration_block = "### Final Diagnostic Guides:\nUse these guides as your primary reference.\n"
        for category, guide in final_guides.items():
            demonstration_block += f"\n---\n{guide}\n"
        description_block = "\n".join([f"- **{name}**: {desc}" for name, desc in FAULT_CATEGORY_DESCRIPTIONS.items()])
        code_block = "```"
        prompt = f"""### Fault Category Definition Manual:
{description_block}

{demonstration_block}
---
### Your Diagnostic Task
Analyze the following **enriched log templates** from the CURRENT case.
#### Current Case: Enriched Log Templates with Embedded Statistics:
{code_block}
{json.dumps(enriched_logs, indent=2)}
{code_block}
### Your Triage Report
Based on all evidence, identify the **three** most probable fault categories. Strictly return your report in the following JSON format, ordered from most likely to least likely:
{code_block}json
{{
  "top_3_categories": ["Most likely category", "Second most likely", "Third most likely"]
}}
{code_block}"""
        try:
            return call_llm(prompt, system_prompt, session_id, is_json_output=True)
        except Exception as e:
            tqdm.write(f"\n--- [L1 Agent] Candidate generation for {session_id} failed: {e}")
            return {"top_3_categories": [LAST_RESORT_FAULT_CATEGORY]}

class L2_SelectionAgent:
    def run(self, enriched_logs: List[str], top_3_candidates: List[str], final_guides: Dict[str, str], session_id: str) -> Dict:
        system_prompt = "You are a decisive, senior SRE. Your task is to make the final call from a shortlist of potential faults and provide a detailed rationale, refuting the incorrect options."
        code_block = "```"
        relevant_guides_block = "### Relevant Diagnostic Guides:\n"
        for category in top_3_candidates:
            if category in final_guides:
                relevant_guides_block += f"\n---\n{final_guides[category]}\n"
        prompt = f"""### Your Final Decision Task
You have a shortlist of three potential fault categories. Your job is to make the final, definitive diagnosis.
#### 1. Current Case Evidence (Enriched Logs):
{code_block}
{json.dumps(enriched_logs, indent=2)}
{code_block}
#### 2. Candidate Faults (from L1 Analyst):
- {top_3_candidates[0]} (Top Candidate)
- {top_3_candidates[1]}
- {top_3_candidates[2] if len(top_3_candidates) > 2 else "N/A"}
{relevant_guides_block}
### Your Final Judgement
1.  Critically re-evaluate the evidence against the guides.
2.  Select the **single best-fitting** fault category.
3.  Provide a clear justification explaining **why your choice is correct** and **why the others are not**.
Strictly return your verdict in JSON format:
{code_block}json
{{
  "fault_category": "The single, final chosen category",
  "justification": "My choice is X because [evidence]. Y is incorrect because [counter-evidence]. Z is less likely because [counter-evidence]."
}}
{code_block}"""
        try:
            return call_llm(prompt, system_prompt, session_id, is_json_output=True)
        except Exception as e:
            tqdm.write(f"\n--- [L2 Agent] Final selection for {session_id} failed: {e}")
            return {"fault_category": top_3_candidates[0], "justification": f"L2 Agent failed: {e}. Defaulting to L1's top candidate."}

# --- 4. 核心控制器和任务处理 ---
def process_event_task(framework_instance, session_id: str, true_label_str: str, diagnostic_guides: Dict[str, str]) -> Tuple[str, str, str]:
    related_logs_df = framework_instance._get_related_logs(session_id)
    if related_logs_df is None or related_logs_df.empty:
        return session_id, true_label_str, LAST_RESORT_FAULT_CATEGORY
    try:
        parsed_data = framework_instance.log_parser.process_logs(related_logs_df)
        templates = parsed_data['templates']
        if not templates:
            return session_id, true_label_str, LAST_RESORT_FAULT_CATEGORY
        
        ids_for_analysis = parsed_data['high_priority_template_ids'] or list(templates.keys())
        variable_stats = framework_instance._calculate_variable_stats(ids_for_analysis, parsed_data['parsed_logs'])
        enriched_logs_for_agent = [framework_instance._enrich_template_with_stats(templates[tid], tid, variable_stats) for tid in ids_for_analysis]
        
        candidate_report = framework_instance.l1_candidate_agent.run(enriched_logs=enriched_logs_for_agent, session_id=f"l1-cand-{session_id}", final_guides=diagnostic_guides)
        top_3_candidates = candidate_report.get('top_3_categories', [])
        
        if not top_3_candidates:
             return session_id, true_label_str, LAST_RESORT_FAULT_CATEGORY
        
        while len(top_3_candidates) < 3:
            padding_candidate = "Normal" if "Normal" not in top_3_candidates else "Mysql"
            if padding_candidate not in top_3_candidates: top_3_candidates.append(padding_candidate)
            else: top_3_candidates.append("AMQP")

        final_report = framework_instance.l2_selection_agent.run(enriched_logs=enriched_logs_for_agent, top_3_candidates=top_3_candidates, final_guides=diagnostic_guides, session_id=f"l2-select-{session_id}")
        predicted_label_str = final_report.get('fault_category', LAST_RESORT_FAULT_CATEGORY)
        
        if predicted_label_str == "Normal":
            predicted_label_str = top_3_candidates[0] if top_3_candidates[0] != "Normal" else LAST_RESORT_FAULT_CATEGORY
            
        return session_id, true_label_str, predicted_label_str
    except Exception as e:
        tqdm.write(f"\n!!! Error processing task for session {session_id}: {e}")
        return session_id, true_label_str, "Error"

class HierarchicalAnalysisFramework:
    def __init__(self):
        self.log_data_dir = LOG_DATA_DIR
        self.config_file = CONFIG_FILE
        self.l0_agent = L0_SummarizationAgent()
        self.l0_5_agent = L0_5_RefinementAgent()
        self.l1_candidate_agent = L1_CandidateGenerationAgent()
        self.l2_selection_agent = L2_SelectionAgent()
        self.log_parser = LogParser()
        self.tasks_df = None
        self.enable_memory = ENABLE_MEMORY
        self.embedding_model = None
        self.sampler = None
        self.demonstration_sessions_map: Dict[str, List[str]] = {}
        self.raw_demonstrations: Dict[str, List[str]] = {}
        self.output_dir = None

        if self.enable_memory:
            try:
                print(f"--- Loading local embedding model: '{EMBEDDING_MODEL_NAME}'...")
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                print("--- Local embedding model loaded successfully! ---")
                self.sampler = DissimilaritySampler(self.log_parser, self.embedding_model, self.log_data_dir)
                print("--- Dissimilarity Sampler is enabled. ---")
            except Exception as e:
                print(f"Critical Error: Failed to load embedding model! Sampler disabled. Error: {e}")
                self.enable_memory = False; self.sampler = None
        else:
            print("--- [Info] Sampler is disabled, skipping embedding model loading. ---")
    
    def _setup_output_dir(self):
        self.output_dir = f"results_hierarchical_framework_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"--- Results will be saved in: {self.output_dir} ---")

    def _prepare_data(self):
        print("--- [Data Preparation] Starting to load tasks from config.json...")
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found.")
        if not os.path.isdir(self.log_data_dir):
            raise FileNotFoundError(f"Log directory not found.")
        
        with open(self.config_file, 'r') as f:
            config_data = json.load(f)
            
        tasks = []
        print("--- [ICL] Selecting example sessions for ICL...")
        for label_str, session_ids in config_data.items():
            if label_str not in FAULT_CATEGORIES:
                continue
            valid_sessions = [sid for sid in session_ids if os.path.exists(os.path.join(self.log_data_dir, f"{sid}.csv"))]
            selected_sessions = []
            if self.sampler and len(valid_sessions) >= 2:
                selected_sessions = list(self.sampler.select_most_dissimilar_pair(valid_sessions))
            elif len(valid_sessions) >= 2:
                print(f" -> Category '{label_str}': Sampler disabled, taking first two sessions.")
                selected_sessions = valid_sessions[:2]
            else:
                print(f" -> Warning: Not enough valid examples for '{label_str}' (need at least 2), skipping for ICL.")
            
            if selected_sessions:
                self.demonstration_sessions_map[label_str] = selected_sessions
            
            if label_str == "Normal":
                continue
            
            for session_id in valid_sessions:
                tasks.append({"session_id": session_id, "label_str": label_str})
                
        self.tasks_df = pd.DataFrame(tasks)
        print(f"Successfully loaded {len(self.tasks_df)} valid [faulty] events.")
        print("--- [Data Preparation] Complete. ---")

    def _enrich_template_with_stats(self, template_text: str, template_id: str, all_stats: Dict) -> str:
        parts = re.split(r'(<\*>)', template_text)
        result_parts = []
        var_counter = 1
        for part in parts:
            if part == '<*>':
                var_key = f"variable_{var_counter}"
                template_stats = all_stats.get(template_id)
                if template_stats and var_key in template_stats:
                    stats_str = json.dumps(template_stats[var_key], ensure_ascii=False)
                    result_parts.append(stats_str)
                else:
                    result_parts.append("{}")
                var_counter += 1
            else:
                result_parts.append(part)
        return "".join(result_parts)

    def _prepare_raw_demonstrations(self):
        print("\n--- [ICL] Preparing raw, enriched examples for each session...")
        if not self.demonstration_sessions_map:
            print("Warning: No demonstration sessions loaded.")
            return
            
        for category, session_id_list in self.demonstration_sessions_map.items():
            category_examples = []
            for session_id in session_id_list:
                demo_df = self._get_related_logs(session_id)
                if demo_df is None or demo_df.empty:
                    continue
                parsed_data = self.log_parser.process_logs(demo_df)
                templates, parsed_logs = parsed_data['templates'], parsed_data['parsed_logs']
                ids_to_process = parsed_data['high_priority_template_ids'] or list(templates.keys())
                stats = self._calculate_variable_stats(ids_to_process, parsed_logs)
                session_enriched_logs = [self._enrich_template_with_stats(templates[tid], tid, stats) for tid in ids_to_process]
                category_examples.append("\n".join(session_enriched_logs))
            if category_examples:
                self.raw_demonstrations[category] = category_examples
        print("--- [ICL] Raw examples prepared. ---")

    def _generate_l0_draft_guides(self) -> Dict[str, str]:
        print("\n--- [Guide Generation] Starting L0 Draft Generation...")
        draft_guides = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_category = {
                executor.submit(self.l0_agent.run, category, examples, f"l0-draft-gen-{category}"): category
                for category, examples in self.raw_demonstrations.items() if len(examples) >= 2
            }
            for future in tqdm(as_completed(future_to_category), total=len(future_to_category), desc="L0 Draft Generation", ncols=100, leave=False):
                category = future_to_category[future]
                try:
                    draft_guides[category] = future.result()
                except Exception as e:
                    tqdm.write(f"\nL0 Agent failed for {category}: {e}")
        print("--- [Guide Generation] L0 Drafts created. ---")
        return draft_guides

    def _generate_l0_5_refined_guides(self, draft_guides: Dict[str, str]) -> Dict[str, str]:
        if not draft_guides:
            print("--- [Guide Generation] No draft guides to refine. Aborting L0.5 step.")
            return {}
        
        print("\n--- [Guide Generation] Preparing for global L0.5 refinement...")
        all_draft_guides_str = "\n---\n".join(draft_guides.values())
        all_examples_str = ""
        for category, examples in self.raw_demonstrations.items():
            if examples:
                all_examples_str += f"#### Example for: {category}\n{examples[0]}\n\n---\n"
        
        session_id = "l0.5-global-refinement"
        final_report_str = self.l0_5_agent.run(all_draft_guides_str, all_examples_str, session_id)
        
        final_guides = self._parse_final_report(final_report_str, draft_guides)
        print("--- [Guide Generation] L0.5 global refinement complete. ---")
        return final_guides
    
    def _create_basic_guides(self) -> Dict[str, str]:
        """Creates simple guides directly from the global descriptions."""
        print("\n--- [Guide Generation] Creating Basic Guides from descriptions...")
        basic_guides = {}
        for category, description in FAULT_CATEGORY_DESCRIPTIONS.items():
            if category == "Categories": continue
            basic_guides[category] = f"#### {category}\n- **Description:** {description}"
        print("--- [Guide Generation] Basic Guides created. ---")
        return basic_guides

    def _parse_final_report(self, report_str: str, fallback_guides: Dict) -> Dict[str, str]:
        guides = {}
        pattern = re.compile(r"####\s+([^\n]+)")
        matches = list(pattern.finditer(report_str))
        
        if not matches:
            print("Warning: L0.5 did not return a parsable report. Using drafts as fallback.")
            return fallback_guides
            
        for i, match in enumerate(matches):
            category_name = match.group(1).strip()
            start_pos = match.end(0)
            end_pos = matches[i+1].start(0) if i + 1 < len(matches) else len(report_str)
            guide_content = report_str[start_pos:end_pos].strip()
            full_guide = f"#### {category_name}\n{guide_content}"
            if category_name in FAULT_CATEGORIES:
                guides[category_name] = full_guide
        return guides

    def _get_related_logs(self, session_id: str) -> Union[pd.DataFrame, None]:
        try:
            return pd.read_csv(os.path.join(self.log_data_dir, f"{session_id}.csv"), on_bad_lines='skip')
        except FileNotFoundError:
            return None

    def _calculate_variable_stats(self, key_template_ids: List[str], all_parsed_logs: List[Dict]) -> Dict:
        stats = {}
        for tid in key_template_ids:
            relevant_logs = [log for log in all_parsed_logs if log['template_id'] == tid]
            if not relevant_logs: continue
            first_log_with_vars = next((log for log in relevant_logs if log.get('variables')), None)
            if not first_log_with_vars: continue
            num_variables = len(first_log_with_vars['variables'])
            stats[tid] = {}
            for i in range(num_variables):
                var_values = [str(log['variables'][i]) for log in relevant_logs if len(log.get('variables', [])) > i and log['variables'][i] is not None]
                if not var_values: continue
                var_counts = Counter(var_values)
                total = len(var_values)
                stats[tid][f"variable_{i+1}"] = {v: round(c / total, 2) for v, c in var_counts.most_common(3)}
        return stats
    
    def _save_detailed_results(self, results: List[Tuple], output_dir: str):
        if not results:
            print("--- No results to save. ---")
            return
        df = pd.DataFrame(results, columns=['session_id', 'true_label', 'predicted_label'])
        output_path = os.path.join(output_dir, 'detailed_predictions.csv')
        df.to_csv(output_path, index=False)
        print(f"--- Detailed predictions saved to: {output_path} ---")

    def run_analysis_and_evaluate(self, ablation_mode: bool = False):
        """主入口点，根据模式分派任务。"""
        self._setup_output_dir()
        self._prepare_data()
        self._prepare_raw_demonstrations()

        if ablation_mode:
            self.run_ablation_study()
        else:
            print("\n--- Running in Standard Mode (Full Pipeline) ---")
            draft_guides = self._generate_l0_draft_guides()
            final_guides = self._generate_l0_5_refined_guides(draft_guides)
            self._run_single_analysis_mode("Standard_Run", final_guides, self.output_dir)

    def run_ablation_study(self):
        """执行消融实验，对比三种指南模式。"""
        print("\n" + "="*80)
        print(" " * 25 + "STARTING ABLATION STUDY")
        print("="*80)
        
        # --- 模式 1: 基础指南 ---
        basic_guides = self._create_basic_guides()
        self._run_single_analysis_mode(
            mode_name="Ablation_Basic",
            diagnostic_guides=basic_guides,
            output_dir=os.path.join(self.output_dir, "ablation_basic")
        )

        # --- 模式 2: L0 草稿指南 ---
        l0_draft_guides = self._generate_l0_draft_guides()
        self._run_single_analysis_mode(
            mode_name="Ablation_L0_Drafts",
            diagnostic_guides=l0_draft_guides,
            output_dir=os.path.join(self.output_dir, "ablation_l0_drafts")
        )

        # --- 模式 3: 完整流程 (L0.5 优化指南) ---
        final_guides = self._generate_l0_5_refined_guides(l0_draft_guides)
        self._run_single_analysis_mode(
            mode_name="Ablation_Full_Pipeline",
            diagnostic_guides=final_guides,
            output_dir=os.path.join(self.output_dir, "ablation_full_pipeline")
        )
        
        print("\n" + "="*80)
        print(" " * 27 + "ABLATION STUDY COMPLETE")
        print(f"All results and reports are saved in separate subdirectories inside: {self.output_dir}")
        print("="*80)

    def _run_single_analysis_mode(self, mode_name: str, diagnostic_guides: Dict[str, str], output_dir: str):
        """对给定的指南集运行一次完整的分析和评估。"""
        os.makedirs(output_dir, exist_ok=True)
        print("\n" + "-"*80)
        print(f"Executing Analysis Mode: [ {mode_name} ]")
        print(f"Outputting results to: {output_dir}")
        print("-"*80)

        guides_path = os.path.join(output_dir, 'diagnostic_guides_used.md')
        with open(guides_path, 'w', encoding='utf-8') as f:
            if not diagnostic_guides:
                f.write(f"!!! No diagnostic guides were available for mode: {mode_name} !!!")
            else:
                f.write(f"### DIAGNOSTIC GUIDES FOR MODE: {mode_name} ###\n\n")
                for category, guide in diagnostic_guides.items():
                    f.write(f"--- [Guide for: {category}] ---\n{guide}\n\n")
        print(f"--- Diagnostic guides for this mode saved to: {guides_path} ---")
            
        results = []
        total_tasks = len(self.tasks_df)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {
                executor.submit(process_event_task, self, row['session_id'], row['label_str'], diagnostic_guides): row['session_id']
                for index, row in self.tasks_df.iterrows()
            }
            print(f"\n--- Starting analysis of {total_tasks} events with {MAX_WORKERS} workers for mode '{mode_name}' ---")
            for future in tqdm(as_completed(future_to_task), total=total_tasks, desc=f"Analyzing Events ({mode_name})", ncols=100):
                session_id = future_to_task[future]
                try:
                    results.append(future.result())
                except Exception as e:
                    tqdm.write(f"\n!!! Critical Error in task for session {session_id}: {e}")
                    results.append((session_id, self.tasks_df[self.tasks_df['session_id'] == session_id]['label_str'].iloc[0], "Error"))

        self._save_detailed_results(results, output_dir)
        
        print(f"\n--- Analysis for mode '{mode_name}' complete. Calculating metrics... ---")
        valid_results = [r for r in results if r and r[2] != "Error"]
        
        self.calculate_metrics(
            [r[1] for r in valid_results], # y_true
            [r[2] for r in valid_results], # y_pred
            output_dir,
            mode_name
        )

    def calculate_metrics(self, y_true: List[str], y_pred: List[str], output_dir: str, mode_name: str):
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')

        report_lines = []
        report_lines.append(f"### Evaluation Report for Mode: [ {mode_name} ] ###\n")

        if not y_true or not y_pred:
            report_lines.append("No valid results to evaluate.")
            with open(report_path, 'w') as f: f.write("\n".join(report_lines))
            print("\n".join(report_lines))
            return
        
        report_lines.append("--- Performance Metrics (excluding 'Normal' class) ---")
        labels = sorted([cat for cat in FAULT_CATEGORIES if cat != "Normal"])
        
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
        
        header = f"{'Category':<30} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        for i, label in enumerate(labels):
            report_lines.append(f"{label:<30} | {p[i]:<10.4f} | {r[i]:<10.4f} | {f1[i]:<10.4f} | {s[i]:<10}")
        report_lines.append("-" * len(header))
        
        micro_f1 = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='micro')[2]
        macro_f1 = np.mean(f1)
        weighted_f1 = np.average(f1, weights=s if sum(s) > 0 else None)
        
        report_lines.append("\n--- Overall Metrics (excluding 'Normal' class) ---")
        report_lines.append(f"Micro F1-Score:    {micro_f1:.6f}")
        report_lines.append(f"Macro F1-Score:    {macro_f1:.6f}")
        report_lines.append(f"Weighted F1-Score: {weighted_f1:.6f}")

        final_report = "\n".join(report_lines)
        print("\n" + final_report)
        with open(report_path, 'w') as f:
            f.write(final_report)
        print(f"\n--- Full evaluation report for '{mode_name}' saved to: {report_path} ---")

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.ylabel('True Label'); plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - Mode: {mode_name}')
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(cm_path)
        print(f"--- Confusion matrix for '{mode_name}' saved to: {cm_path} ---")

# --- 5. 入口点 ---
if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Config file not found at '{CONFIG_FILE}'.")
    elif not os.path.isdir(LOG_DATA_DIR):
        print(f"Error: Log directory not found at '{LOG_DATA_DIR}'.")
    else:
        # 检查是否传入了消融实验标志
        ablation_mode = "--ablation" in sys.argv
        
        if ablation_mode:
            print("--- Detected '--ablation' flag. Running in Ablation Study Mode. ---")
        else:
            print("--- Running in Standard Mode. To run the ablation study, use the '--ablation' flag. ---")

        framework = HierarchicalAnalysisFramework()
        framework.run_analysis_and_evaluate(ablation_mode=ablation_mode)