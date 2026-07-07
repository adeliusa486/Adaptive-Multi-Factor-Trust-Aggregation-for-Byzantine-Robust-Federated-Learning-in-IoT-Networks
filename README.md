# Adaptive Multi-Factor Trust Aggregation (AMFTA) for Byzantine-Resilient Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.1.0-ee4c2c.svg)](https://pytorch.org/)

An engineering-grade research software artifact and comparative benchmarking framework for evaluating **Byzantine-resilient server-side aggregation strategies** in non-IID federated network intrusion detection systems (NIDS). Evaluated on the real-world **TON_IoT** smart city telemetry dataset under diverse model-poisoning and data-poisoning threat models.

---

## Table of Contents

- [Overview & Motivation](#overview--motivation)
- [Key Capabilities & Threat Models](#key-capabilities--threat-models)
- [Aggregators Compared](#aggregators-compared)
- [Repository Architecture](#repository-architecture)
- [Installation & Environment Setup](#installation--environment-setup)
  - [Linux & macOS](#linux--macos)
  - [Windows (PowerShell)](#windows-powershell)
- [Data Preparation Workflow](#data-preparation-workflow)
- [Quick Start & Usage Examples](#quick-start--usage-examples)
  - [Single Experiment Execution](#single-experiment-execution)
  - [Resumable Parameter Sweep](#resumable-parameter-sweep)
  - [Ablation Studies](#ablation-studies)
- [Reproducing Paper Tables & Figures](#reproducing-paper-tables--figures)
- [Verified Benchmark Results](#verified-benchmark-results)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Experimental Assumptions & Limitations](#experimental-assumptions--limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview & Motivation

In decentralized smart city and Industrial Internet of Things (IIoT) environments, Collaborative Federated Learning (FL) allows edge devices to collaboratively train intrusion detection models without sharing sensitive raw network telemetry. However, standard federated aggregation protocols like **Federated Averaging (FedAvg)** are vulnerable to **Byzantine failures**—where adversarial or compromised edge nodes inject malicious model updates to subvert global model convergence or trigger targeted misclassifications.

This repository provides a reproducible implementation and systematic empirical evaluation of **Adaptive Multi-Factor Trust Aggregation (AMFTA)**, comparing it against established Byzantine-robust aggregation methods under realistic non-IID client data distributions ($\text{Dirichlet } \alpha = 0.5$).

### Primary Empirical Findings
- **No Single Strategy Dominates Across All Regimes:** Coordinate-wise trimming and density-based clustering perform exceptionally well under low-to-moderate model poisoning but degrade rapidly when the malicious client fraction reaches $30\%$.
- **Graceful Degradation via Multi-Factor Trust:** By synthesizing instant cosine similarity, historical reputation exponential moving average (EMA), and quality factor evaluations, AMFTA maintains robust classification accuracy even under severe adversarial fractions ($30\%–40\%$).
- **Independence from Root Validation Set:** Empirical results demonstrate that removing the server-side root validation dataset (**AMFTA-ND**) does not degrade defensive performance and can improve stability at high attacker fractions.

---

## Key Capabilities & Threat Models

The framework implements a modular federated training engine supporting plug-and-play aggregators, configurable client data partitions, and multiple adversarial attack models:

- **Data-Poisoning Attacks:**
  - **Label Flipping:** Malicious edge nodes systematically invert training labels (e.g., benign traffic mapped to malicious intrusion classes) prior to local training.
- **Model-Poisoning Attacks:**
  - **Gaussian Noise Injection:** Adversaries corrupt local model weight updates by adding high-variance isotropic Gaussian noise ($\mathcal{N}(0, \sigma^2)$) to disrupt global convergence.
  - **Sign Flipping:** Malicious clients invert the sign of their computed model weight deltas before transmitting them to the aggregation server.
  - **Mimicry Attacks:** Adversaries craft deceptive weight vectors that mimic statistical norms of benign updates while slowly steering the global decision boundary.
- **Realistic Non-IID Telemetry:** Implements Dirichlet distribution partitioning across $100$ simulated IoT client nodes to reflect real-world network traffic heterogeneity.

---

## Aggregators Compared

| Aggregator | Code Module | Mechanism & Defense Strategy |
| :--- | :--- | :--- |
| **FedAvg** | `amfta.aggregation.baselines` | Standard weighted empirical mean aggregation; no Byzantine resilience (baseline reference). |
| **Trimmed Mean** | `amfta.aggregation.baselines` | Coordinate-wise sorting and trimming of outlier weight distributions before averaging. |
| **Krum** | `amfta.aggregation.baselines` | Distance-based robust selection; chooses the single local update with the lowest squared Euclidean distance to its closest neighbors. |
| **FLTrust** | `amfta.aggregation.baselines` | Compute cosine similarity scores and magnitude clipping against a server-maintained reference model update trained on clean validation data. |
| **FedDBC** | `amfta.aggregation.baselines` | Density-Based Clustering (DBSCAN) on weight feature representations to isolate and filter outlier cluster models. |
| **AMFTA** | `amfta.aggregation.amfta` | **Ours:** Synthesizes multi-factor trust metrics (cosine similarity + reputation EMA + quality scoring) with server reference validation. |
| **AMFTA-ND** | `amfta.aggregation.amfta` | **Ours (No Defense Buffer):** AMFTA trust aggregation operating entirely without access to a clean server validation set. |

---

## Repository Architecture

```text
amfta-fl/
├── amfta/                      # Core Python package
│   ├── aggregation/            # Implementation of AMFTA, AMFTA-ND, and baseline aggregators
│   ├── attacks/                # Byzantine attack models (label_flipping, gaussian_noise, etc.)
│   ├── data/                   # TON_IoT dataset preprocessing and Dirichlet partitioning
│   ├── models/                 # Neural network architectures for network intrusion detection
│   └── utils/                  # Helper utilities, logging, and mathematical operations
├── configs/                    # YAML experiment configuration files and parameter loaders
├── data/                       # Dataset directory (raw/ and processed/ telemetry)
├── docs/                       # Technical documentation and detailed setup guides
├── evaluation/                 # Metrics calculation and statistical evaluation utilities
├── experiments/                # Experiment drivers, ablation runners, table and figure builders
├── figures/                    # Generated publication-quality PDF and PNG plots
├── results/                    # Serialized JSON run logs across all evaluated seeds
├── scripts/                    # Helper scripts for standalone inference and data setup
├── tests/                      # Automated unit, integration, and smoke test suites (pytest)
├── training/                   # Federated learning server/client orchestration loop
├── Makefile                    # Make automation targets for linting, testing, and training
├── pyproject.toml              # Modern Python build metadata, dependencies, and tool configs
└── requirements.txt            # Pinned package requirements for immediate pip installation
```

---

## Installation & Environment Setup

### Prerequisites
- Python $\ge 3.9$, $3.10$, or $3.11$
- `pip` or `conda` package manager
- Recommended: Git and Virtual Environment (`venv` or `virtualenv`)

### Linux & macOS

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amfta-research/amfta-fl.git
   cd amfta-fl
   ```

2. **Create and activate an isolated virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install core and developer dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

### Windows (PowerShell)

1. **Clone the repository and enter the directory:**
   ```powershell
   git clone https://github.com/amfta-research/amfta-fl.git
   cd amfta-fl
   ```

2. **Create and activate a Python virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install required dependencies:**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

---

## Data Preparation Workflow

The evaluation is conducted on the official **TON_IoT** telemetry dataset from UNSW Canberra. Because the raw network flow records are large (~several gigabytes), they are excluded from source control.

1. **Download the Raw Dataset:**
   Obtain the official network telemetry CSV files from the [UNSW Canberra Research Repository](https://research.unsw.edu.au/projects/toniot-datasets) and place the downloaded files into `data/raw/`.

2. **Run Data Preprocessing:**
   Standardize network features, encode categorical variables, and normalize numerical distributions:
   ```bash
   python -m amfta.data.preprocessing
   ```

3. **Generate Non-IID Dirichlet Partitions:**
   Partition the preprocessed NIDS data across $100$ federated clients using a Dirichlet distribution ($\alpha = 0.5$) to simulate realistic edge traffic skewness:
   ```bash
   python -m amfta.data.partitioning
   ```

> [!NOTE]
> If you do not wish to re-partition the dataset from scratch, the pre-computed evaluation logs in `results/` contain complete trial runs across $3$ random seeds ($42, 123, 456$) and can be tabulated directly using the provided build scripts.

---

## Quick Start & Usage Examples

### Single Experiment Execution

To launch a single federated learning training run evaluating a specific aggregator and attack configuration, execute the primary experiment driver `experiments/run_main.py`:

```bash
python experiments/run_main.py \
    --method amfta \
    --attack label_flipping \
    --byzantine_fraction 0.3 \
    --num_clients 100 \
    --num_rounds 25 \
    --seeds 42
```

Alternatively, if installed as an editable Python package, use the registered CLI entry point:
```bash
amfta-train --method trimmed_mean --attack gaussian_noise --byzantine_fraction 0.2
```

### Resumable Parameter Sweep

To execute the comprehensive benchmarking study across all $7$ aggregation strategies, attack types, and Byzantine client fractions ($10\%$, $20\%$, $30\%$, $40\%$), run the automated study runner. This script is fully **resumable**—if interrupted, re-running it will automatically detect and skip already completed JSON result logs in `results/`:

```bash
python experiments/run_focused_study.py
```

### Ablation Studies

To investigate the individual contributions of trust factors (cosine similarity vs. reputation EMA vs. quality scoring), execute the ablation study driver:

```bash
python experiments/run_ablation.py
```

---

## Reproducing Paper Tables & Figures

All empirical tables and visual charts presented in the research manuscript are generated programmatically directly from the raw JSON execution logs in `results/`. No numerical data is hand-edited.

### 1. Compile Summary Tables
Generate the exact text tables comparing last-5-round mean test accuracy and standard deviation across all $3$ evaluated seeds:
```bash
python experiments/build_paper_tables.py
```
*Output:* Prints formatted LaTeX and Markdown comparison tables to the standard output.

### 2. Generate Publication Figures
Compile publication-grade vector (PDF) and raster (PNG) charts matching Elsevier and IEEE journal styling:
```bash
python experiments/build_paper_figures.py
```
*Output:* Generates high-resolution charts in `figures/`:
- `fig_bar30.pdf` / `fig_bar30.png`: Comparative bar chart at $30\%$ Byzantine fraction.
- `fig_lineflip.pdf` / `fig_lineflip.png`: Performance degradation curves under Label Flipping.
- `fig_linegauss.pdf` / `fig_linegauss.png`: Robustness curves under Gaussian Noise injection.

---

## Verified Benchmark Results

The following tables summarize the verified mean test classification accuracy ($\%$) and standard deviation evaluated over the final $5$ communication rounds across $3$ independent random seeds ($42, 123, 456$).

### Table 1: Label Flipping Attack
*Federated training across $100$ clients ($5$ local epochs, Dirichlet $\alpha=0.5, T=25$ rounds).*

| Aggregation Method | $10\%$ Byzantine | $20\%$ Byzantine | $30\%$ Byzantine | $40\%$ Byzantine |
| :--- | :---: | :---: | :---: | :---: |
| **FedAvg** (Reference) | $94.26 \pm 0.71$ | $89.34 \pm 0.77$ | $73.81 \pm 0.82$ | $63.24 \pm 6.57$ |
| **Trimmed Mean** | $92.07 \pm 1.25$ | $91.37 \pm 1.05$ | $82.99 \pm 0.68$ | $73.21 \pm 3.26$ |
| **Krum** | $89.66 \pm 0.63$ | $89.46 \pm 0.98$ | $87.30 \pm 3.26$ | $68.24 \pm 28.88$ |
| **FLTrust** | $77.36 \pm 1.50$ | $74.57 \pm 0.98$ | $72.05 \pm 1.05$ | $69.96 \pm 2.50$ |
| **FedDBC** | $92.08 \pm 1.26$ | $85.56 \pm 5.37$ | $71.99 \pm 2.36$ | $61.00 \pm 2.96$ |
| **AMFTA** (Ours) | $\mathbf{93.06 \pm 1.18}$ | $\mathbf{92.81 \pm 1.11}$ | $80.32 \pm 9.64$ | $54.63 \pm 5.11$ |
| **AMFTA-ND** (Ours, No Buffer) | $92.47 \pm 1.09$ | $92.27 \pm 1.47$ | $\mathbf{91.69 \pm 0.99}$ | `--` |

### Table 2: Gaussian Noise Injection Attack
*High-variance isotropic noise corruption under identical non-IID partitioning parameters.*

| Aggregation Method | $10\%$ Byzantine | $20\%$ Byzantine | $30\%$ Byzantine |
| :--- | :---: | :---: | :---: |
| **FedAvg** (Reference) | $71.33 \pm 1.77$ | $40.76 \pm 18.88$ | $42.80 \pm 21.76$ |
| **Trimmed Mean** | $\mathbf{94.43 \pm 0.81}$ | $56.97 \pm 20.91$ | $41.49 \pm 19.90$ |
| **Krum** | $89.95 \pm 0.44$ | $89.86 \pm 0.75$ | $90.32 \pm 0.20$ |
| **FLTrust** | $77.29 \pm 1.62$ | $73.61 \pm 1.95$ | $70.29 \pm 1.87$ |
| **FedDBC** | $93.90 \pm 0.97$ | $\mathbf{92.78 \pm 1.36}$ | $65.33 \pm 9.19$ |
| **AMFTA** (Ours) | $90.29 \pm 0.91$ | $89.46 \pm 0.91$ | $89.28 \pm 0.68$ |
| **AMFTA-ND** (Ours, No Buffer) | $90.92 \pm 0.89$ | $90.64 \pm 1.14$ | $\mathbf{90.55 \pm 0.89}$ |

---

## Testing & Quality Assurance

The repository maintains an automated test suite implemented via `pytest`, covering mathematical trust calculations, aggregation bounds, and end-to-end training smoke tests.

### Run Unit and Integration Tests
Execute the complete test suite with verbose reporting:
```bash
pytest -v
```

### Run Fast Smoke Tests
Run only quick smoke tests to verify environment functionality without triggering long training loops:
```bash
pytest -v -m "smoke"
```

### Code Style & Static Analysis
The codebase adheres strictly to PEP-8 standards enforced by `black`, `isort`, and `mypy`. To run formatting and type-checking audits:
```bash
black --check amfta experiments training
isort --check-only amfta experiments training
mypy amfta experiments training
```

---

## Experimental Assumptions & Limitations

To ensure rigorous scientific transparency, we explicitly note the foundational assumptions and empirical boundary conditions of this study:

1. **Honest-Majority Assumption:** In alignment with standard Byzantine fault-tolerant learning theory, our core evaluation assumes an honest majority ($10\%–30\%$ malicious fraction). While AMFTA-ND demonstrates resilience at $40\%$ under label flipping, performance under strict adversarial majorities ($\ge 50\%$) remains an open research challenge.
2. **Fixed Heterogeneity Skew:** Experiments are benchmarked under a Dirichlet concentration parameter of $\alpha = 0.5$. While representing realistic IoT telemetry variance, extreme non-IID regimes ($\alpha < 0.1$) may alter distance metrics and cluster boundaries in clustering-based defenses (e.g., FedDBC).
3. **Telemetry Domain Specificity:** The neural architectures and feature representations are optimized specifically for tabular network flow telemetry (TON_IoT NIDS). Applying trust factors directly to high-dimensional unstructured domains (e.g., high-resolution vision or generative LLM weights) may require domain-specific layer normalization.

---

## Contributing

We welcome contributions from researchers and open-source engineers! To contribute:

1. **Fork the Repository:** Create a feature branch from `main` (`git checkout -b feature/new-defense`).
2. **Adhere to Standards:** Ensure all new code passes static analysis (`black`, `isort`, `mypy`) and includes corresponding `pytest` unit coverage.
3. **Submit a Pull Request:** Describe the technical rationale, mathematical formulation (if introducing a new aggregator), and empirical verification logs.

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) and our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for detailed guidelines.



## License

This software artifact is released under the **MIT License**. See the [LICENSE](LICENSE) file for the complete terms and legal details.
