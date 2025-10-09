# LogCKF: Enhancing Log-based Fault Diagnosis with LLM through Contrastive Knowledge Refinement
### Artifact for WWW'26 Industry Submission

We sincerely thank the Program Committe/ Artifact Evaluation Committee/ Reviewers for their time and expertise in reviewing our work. 

---

## 1. Artifact Overview

This artifact enables the reproduction of the following key results:

*   **RQ1 (Baseline Experiment:)** The performance of baselines (logCKF, Scalalog, directly applying various general-purpose LLMs, ...) for log-based fault diagnosis.
*   **RQ2 (HITL Evolution):** The performance evolution of LogCKF with a simulated human feedback loop, compared to a standard version without feedback.
*   **RQ3 (Ablation Study):** The performance of LogCKF variants to demonstrate the contribution of each component in our Contrastive Knowledge Refinement mechanism.


> **Note on Private APIs and Data:**
> *   The scripts for fetching log data from our internal production environment (`keywords_search.py`, `get_search_result.py`, `get_logid.py`) are included for completeness. However, they are **not runnable** by evaluators as they rely on private, internal Volcano Engine APIs and credentials.
> *   The experiments are designed to run on publicly available datasets as described below.

## 2. Environment Setup

### 2.1. System Requirements
*   **OS:** Linux (Recommended)
*   **Python:** `3.9.6` or a compatible `3.9.x` version.
*   **Hardware:** A modern CPU with at least 8 cores and 16GB of RAM is recommended for running experiments in parallel.

### 2.2. Python Dependencies
Install the required Python packages using pip:

    pip install pandas openai scikit-learn seaborn matplotlib sentence-transformers tqdm

### 2.3. Dataset Setup
Our experiments use two different public datasets. Please set up both as described below.

#### 2.3.1. LogKG Dataset (for Ablation Study)
The ablation study (`logCKF.py`) uses the **LogKG** dataset.

1.  Clone the LogKG repository into a temporary location:

        git clone https://github.com/SycIsDD/LogKG.git

2.  If an empty `LogKG` directory exists in the artifact, remove it first. Then, move the cloned repository into the artifact's root directory:

        mv LogKG/ /path/to/your/artifact/

    This ensures the script can find the data at the expected path: `LogKG/data/CMCC_case/`.

#### 2.3.2. Dataset2 / CSV Files (for HITL and Baseline Studies)
The Human-in-the-Loop evolution study (`evolution.py`) and the baseline study (`Direct_dataset1_llm.py`) use a set of CSV files referred to as "Dataset2" in the paper. Ensure the following files are present in the root directory of the artifact:
*   `preliminary_sel_log_dataset.csv`
*   `preliminary_train_label_dataset.csv`
*   `preliminary_train_label_dataset_s.csv`

### 2.4. LLM API Configuration

> **IMPORTANT: Action Required**
> Our code accesses LLMs through a private, internal Bytedance API endpoint. You **must** modify each experiment script to use your own LLM provider (e.g., OpenAI, Azure).

For each of the main scripts (`logCKF.py`, `evolution.py`, `Direct_dataset1_llm.py`), perform the following steps:

1.  Open the script and locate the global configuration section at the top.
2.  Modify the API settings to point to your OpenAI-compatible API endpoint and provide your API key. For example, for OpenAI:

        # --- Example for OpenAI ---
        INTERNAL_LLM_BASE_URL = "https://api.openai.com/v1" # Or your proxy
        INTERNAL_LLM_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" # Your OpenAI API Key
        MODEL_NAME = "gpt-4-turbo-preview" # A model you have access to

## 3. Reproducing Key Results (RQ3: Ablation Study on LogKG)

The main script, `logCKF.py`, runs the ablation study on the **LogKG dataset**. The results for each run are saved in a timestamped directory (e.g., `results_hierarchical_framework_...`).

### 3.1. Running the Full LogCKF Model
To run the standard experiment with the complete LogCKF framework:

    python3 logCKF.py

This will generate the main results in a directory named `results_.../Standard_Run`.

### 3.2. Running the Ablation Study (RQ3)
To automatically run the full pipeline and both ablation variants described in the paper, use the `--ablation` flag:

    python3 logCKF.py --ablation

This command will execute three separate experimental runs and save the results in distinct subdirectories within a new results folder:
*   `ablation_basic/`: **(w/o Refinement)** Results using only basic descriptions.
*   `ablation_l0_drafts/`: **(w/ Intra-Class Distillation)** Results using draft guides.
*   `ablation_full_pipeline/`: **(Full LogCKF)** The complete model with all refinements.

## 4. Reproducing Human-in-the-Loop Evolution Results (RQ2 on Dataset2)

The `evolution.py` script runs the experiment described in RQ2 on **Dataset2 (the CSV files)**. It compares the LogCKF framework with and without its human feedback loop.

### 4.1. Prerequisites
*   Ensure the CSV files mentioned in section 2.3.2 are in the root directory.
*   Ensure you have configured the LLM API credentials in `evolution.py` as described in section 2.4.

### 4.2. Running the Experiment

    python3 evolution.py

### 4.3. Expected Output
The script will create a new results directory (e.g., `results_<timestamp>`) containing:
*   **`standard/` directory:** Contains the results for the LogCKF branch without human feedback.
*   **`hitl/` directory:** Contains the results for the LogCKF branch with the simulated Human-in-the-Loop feedback mechanism.
*   **`periodic_metrics_log.csv`:** A crucial file at the top level of the results directory. It logs the performance metrics for both branches after each batch of tasks, providing the data needed to plot the evolution graphs.
*   **`initial_diagnostic_guides.md`:** The initial set of guides used at the start of the experiment.

## 5. Reproducing Baseline Results 

We provide `Direct_dataset1_llm.py` and a shell script to evaluate the performance of directly prompting various LLMs for fault diagnosis on **Dataset2 (the CSV files)**.

### 5.1. Running a Single Model
You can test a single LLM by running the Python script directly:

    # Make sure to configure the API credentials in Direct_dataset1_llm.py first!
    python3 Direct_dataset1_llm.py --model-name "gpt-4-turbo-preview" --task-limit 50

### 5.2. Running All Baselines in Batch
The `experiment.sh` script automates the process of running experiments for multiple LLMs, three times each.

1.  **Configure Models:** Edit `experiment.sh` to include the model names you have access to.

        # In experiment.sh
        MODELS=(
          "gpt-4-turbo-preview" # Replace with your models
          "gpt-3.5-turbo"
        )

2.  **Execute the Script:**

        bash experiment.sh

## 6. Evaluating Results and Generating Metrics

We provide a utility script, `metric.py`, to calculate detailed performance metrics from any of the `*.csv` files generated by the experiments.

**Usage:**

    # Example for an Ablation Study run
    python3 metric.py results_hierarchical_framework_.../ablation_full_pipeline/detailed_predictions.csv

    # Example for an HITL run
    python3 metric.py results_.../hitl/detailed_predictions.csv

The script will print a detailed report to the console and save a `confusion_matrix.png` in the current directory.

## 7. Reproducing Figures

Scripts are provided to regenerate the main figures from the paper using the hardcoded data from our experiments.

*   **Ablation Study Figure:** To generate the bar chart from the RQ3 study:

        python3 ablation_figure.py

*   **Evolution Figure:** To generate the performance evolution charts from the RQ2 study:

        python3 evolution_figure.py

## 8. Supplementary Scripts (For Context Only)

The following scripts are included to show our data acquisition process from **internal, private systems**.

*   `keywords_search.py`: Initiates a log search task.
*   `get_search_result.py`: Retrieves the results of a search task.
*   `get_logid.py`: Fetches detailed log contexts for specific log IDs.

> **Disclaimer:** These scripts are **not runnable** by artifact evaluators. They are provided for informational purposes only. All key experiments are fully reproducible using the public datasets as described in Section 2.3.