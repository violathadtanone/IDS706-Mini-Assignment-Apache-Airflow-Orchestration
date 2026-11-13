# IDS 706 Mini Assignment: Data Orchestration with Apache Airflow

## Project Description
This project aims to build an end-to-end healthcare data pipeline using Apache Airflow and use the cleaned data in an agent-based machine learning workflow. The project focuses on practicing data ingestion, transformation, task orchestration, workflow automation, and agent coordination.

## Project Structure

```text
IDS706-Mini-Assignment-Apache-Airflow-Orchestration/
├── dags/                     # Airflow DAGs for ETL and the fixed ML baseline
├── dataset/                  # CMS healthcare source datasets
├── ml_workflow/              # Part 2 LangGraph ML workflow and backend logic
├── frontend/                 # Live UI for visualizing Part 2 agent execution
├── adaptive_workflow/        # Part 3 adaptive multi-agent workflow and routing
├── evaluation/               # Evaluation cases and three-system comparison
├── tests/                    # Unit, integration, and workflow validation tests
├── audit/                    # Runtime and assignment-requirement verification
├── screenshots/              # Screenshots used as submission evidence
├── output/                   # Generated metrics, plots, predictions, and results
├── images/                   # Images used in the README and documentation
├── Dockerfile                # Base Docker image configuration
├── docker-compose.yaml       # Core Airflow and PostgreSQL services
├── compose.ml.yaml           # Part 2 ML workflow and UI services
├── compose.part3.yaml        # Part 3 adaptive workflow services
├── compose.evaluation.yaml   # Evaluation and comparison services
└── README.md                 # Project overview, architecture, and instructions
```

### Overall Architecture

![Diagram](images/architecture.png)
This project uses two datasets from the Centers for Medicare & Medicaid Services (CMS):
- [Hospital General Information](https://data.cms.gov/provider-data/dataset/xubh-q36u) — This contains general information about hospitals, including facility identifiers, hospital type, location, ownership, and overall hospital ratings.
- [Healthcare Associated Infections—Hospital](https://data.cms.gov/provider-data/dataset/77hc-ibv8) — This contains hospital-level measures related to healthcare-associated infections, including infection measure types, reported scores, and reporting periods.
- These two datasets can be linked using the shared Facility ID, allowing hospital characteristics to be combined with infection-related outcomes for analysis.
<br><br>

### Overall Architecture

The project is organized into three stages:
- Part 1 - Prepare and store the healthcare data using Apache Airflow
- Part 2 - Use the cleaned data in an agent-based machine learning workflow
- Part 3 - extends the workflow with adaptive routing and recovery

### Part 1 — Airflow Pipeline
- This step builds an end-to-end data pipeline using Apache Airflow. Two related healthcare datasets are ingested and cleaned independently, then merged using Facility ID.
- The merged data is stored in PostgreSQL and read back for analysis. Independent tasks run in parallel where appropriate, and the pipeline is executed on a scheduled DAG.
- Cleaning normalizes missing values, converts numeric scores/ratings and dates, and validates facility and infection-record keys before merging.
- The DAG runs daily at **00:00 UTC**. Hospital and infection ingestion/cleaning branches run independently before the merge.
- Analysis reads PostgreSQL back and saves a CSV and visualization of average central-line infection score by hospital rating. Differences between rating groups describe an association, not a causal effect.

### Part 2 — Agent-Based ML Workflow
- This part uses the cleaned data from Part 1 in an agent-based machine learning workflow.
- Different agents handle data preparation, model training, evaluation, and model adjustment. The workflow can loop through multiple model attempts and uses mean absolute error (MAE) to select the best validated predictor.
- A live graph interface shows the actual agent execution and loop transitions.
- Ridge regression estimates `HAI_1_SIR` using hospital rating, type, and state. The median predictor beat Ridge on validation; selected validation MAE was **0.426202**, and test MAE was **0.448833**.

### Part 3 — Adaptive Multi-Agent Workflow
- This part extends the workflow by introducing multiple specialist agents with different responsibilities.
- A dispatcher selects the next agent based on intermediate results. The system can route tasks dynamically, retry temporary failures through a recovery agent, and stop appropriately when a permanent error occurs.
- This adaptive workflow is later compared against a fixed Airflow workflow and the simpler agent-based workflow from Part 2.
<br><br>

### Baseline Comparison

Part 3 was compared against healthcare adaptations of the **two baseline workflow patterns demonstrated in class**:

| Workflow | Classroom baseline it represents | Key behavior |
|---|---|---|
| **Fixed Airflow Baseline** | Fixed Airflow workflow | Executes predefined steps with one fixed Ridge model and no adaptive model selection. |
| **Part 2 Agent Workflow** | Simple agentic workflow | Uses agents and a bounded model-adjustment loop, but has limited routing and no database-recovery agent. |
| **Part 3 Adaptive Workflow** | Proposed adaptive system | Dynamically selects specialist agents based on intermediate results and can route recoverable failures through a Recovery Coordinator. |

All three healthcare workflows use the same prediction task, cohort, train/validation/test splits, predictors, and MAE metric so their results can be compared fairly.

| Healthcare Workflow | Validation MAE | Test MAE | Failure Recovery |
|---|---:|---:|---|
| Fixed Airflow Baseline | 0.432464 | **0.442157** | Standard task retry |
| Part 2 Agent Workflow | **0.426202** | 0.448833 | No |
| Part 3 Adaptive Workflow | **0.426202** | 0.448833 | Adaptive recovery agent |

The fixed Airflow workflow achieved the lowest test MAE, while the Part 3 workflow provided the strongest coordination and recovery behavior. The adaptive workflow passed all nine evaluation cases, including routing, recovery, invalid-input handling, test-set isolation, and execution-budget checks.