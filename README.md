# AMFTA-FL: Adaptive Multi-Factor Trust Aggregation for Byzantine-Resilient Federated Learning

[![CI](https://github.com/amfta-research/amfta-fl/actions/workflows/ci.yml/badge.svg)](https://github.com/amfta-research/amfta-fl/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Federated Intrusion Detection for Smart City IoT with 30% Byzantine Tolerance**

Official implementation of the AMFTA framework described in:

> *"AMFTA: Adaptive Multi-Factor Trust Aggregation for Byzantine-Resilient Federated Learning in Non-IID Smart City IoT Networks"* — submitted to IEEE Internet of Things Journal.

---

## Overview

AMFTA is a Byzantine-robust federated learning aggregation framework designed for heterogeneous IoT deployments. Unlike prior methods that rely on clean server-side data (FLTrust) or reject legitimate non-IID updates (Krum), AMFTA uses three complementary trust signals:

| Factor | Symbol | Description |
|--------|--------|-------------|
| **I — Gradient Similarity** | S_i | Cosine similarity of client update to population centroid |
| **II — Historical Reputation** | H_i | EMA of per-client consistency across rounds (β=0.9) |
| **III — Contribution Quality** | Q_i | Leave-one-out accuracy delta on 500-sample server buffer |

**Final trust score:**
```
T_i = α_s · S_i + α_h · H_i + α_q · Q_i
    = 0.4 · S_i + 0.3 · H_i + 0.3 · Q_i
```

Quality evaluation (Factor III) is applied **selectively** to borderline clients only (τ_l=0.35 < T̂ < τ_u=0.55), reducing computational overhead while maintaining accuracy.

### Key Results (TON_IoT, 30% Byzantine, Label Flipping)

| Method | Accuracy | F1 | Precision | Recall |
|--------|----------|----|-----------|--------|
| FedAvg | 0.712 | 0.698 | 0.683 | 0.714 |
| Krum | 0.744 | 0.731 | 0.756 | 0.708 |
| Trimmed Mean | 0.768 | 0.759 | 0.771 | 0.748 |
| FLTrust | 0.801 | 0.793 | 0.812 | 0.775 |
| **AMFTA** | **0.923** | **0.918** | **0.931** | **0.906** |

---

## Architecture

```
amfta-fl/
├── amfta/                      # Core package
│   ├── models/
│   │   └── local_mlp.py        # 3-layer MLP (45→64→32→1)
│   ├── aggregation/
│   │   ├── trust_factors.py    # Factors I, II, III computation
│   │   ├── amfta.py            # AMFTAAggregator (full pipeline)
│   │   └── baselines.py        # FedAvg, Krum, TrimmedMean, FLTrust
│   ├── attacks/
│   │   ├── label_flipping.py   # Label flip Byzantine attack
│   │   └── gaussian_noise.py   # Gaussian noise Byzantine attack
│   ├── data/
│   │   ├── preprocessing.py    # TON_IoT data pipeline
│   │   └── partitioning.py     # Dirichlet non-IID partitioning
│   └── utils/
│       ├── metrics.py          # Evaluation metrics
│       ├── reproducibility.py  # Seeding utilities
│       └── logging_utils.py    # Experiment logging
├── training/
│   └── federated_runner.py     # Full FL training orchestrator
├── experiments/
│   ├── run_main.py             # Table II: all methods comparison
│   ├── run_ablation.py         # Table III: ablation study
│   └── run_scalability.py      # Figure 4: Byzantine rate sweep
├── evaluation/
│   └── visualize.py            # Figure generation
├── api/
│   └── main.py                 # FastAPI inference service
├── tests/                      # Unit + integration tests
├── configs/
│   └── config.yaml             # Experiment configuration
├── deployment/k8s/             # Kubernetes manifests
└── monitoring/                 # Prometheus + Grafana configs
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/amfta-research/amfta-fl.git
cd amfta-fl
pip install -r requirements.txt
```

### 2. Quick Smoke Test (No Data Download Required)

```bash
# Uses synthetic data — verifies the full pipeline works
make smoke
# or
python experiments/run_main.py \
  --method amfta \
  --use_synthetic \
  --num_rounds 5 \
  --num_clients 10
```

### 3. Full Experiment (TON_IoT Dataset)

**Download the dataset:**
```bash
# Place NF-TON-IoT.csv in data/raw/
# Source: https://research.unsw.edu.au/projects/toniot-datasets
```

**Preprocess:**
```bash
make preprocess       # Normalize, split, extract server buffer
make partition        # Dirichlet partition (α=0.5, N=100 clients)
```

**Run all experiments:**
```bash
make train            # All methods × all Byzantine rates
make ablation         # Ablation study
make figures          # Generate paper figures
```

---

## Dataset

This implementation uses the **TON_IoT Network Traffic Dataset** (Network Flow variant):

- **Source:** [UNSW Canberra](https://research.unsw.edu.au/projects/toniot-datasets)
- **Samples:** 461,043 flow records (after deduplication)
- **Features:** 45 network flow features (min-max normalised to [0,1])
- **Labels:** Binary — Normal (38%) / Attack (62%)

Place `NF-TON-IoT.csv` in `data/raw/` before running preprocessing.

---

## API Usage

Start the inference server:
```bash
make api
# or: uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Send a prediction request:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ...]}' # 45 min-max normalised values
```

Response:
```json
{
  "attack_probability": 0.9234,
  "is_attack": true,
  "threshold": 0.5,
  "latency_ms": 1.23
}
```

API Documentation: http://localhost:8000/docs (Swagger UI)

---

## Configuration

All experiment parameters are in `configs/config.yaml`. Key settings:

```yaml
federated:
  num_clients: 100
  num_rounds: 100
  local_epochs: 5

byzantine:
  fraction: 0.30
  attack_type: "label_flipping"

amfta:
  alpha_s: 0.4    # Factor I weight
  alpha_h: 0.3    # Factor II weight
  alpha_q: 0.3    # Factor III weight
  beta: 0.9       # EMA decay
  tau_lower: 0.35
  tau_upper: 0.55
```

---

## Testing

```bash
make test-unit         # Fast unit tests (~10 seconds)
make test-integration  # Full FL round simulation (~2 minutes)
make test-coverage     # With coverage report
```

---

## Docker

```bash
make docker-build      # Build image
make docker-up         # Start all services (API + monitoring)
make docker-test       # Run tests in container
make docker-train      # Run quick training in container
```

---

## Reproducibility

All experiments use 5 random seeds `{42, 123, 456, 789, 1024}` with:
- PyTorch `manual_seed` + CUDA seed
- NumPy `np.random.seed`
- `torch.use_deterministic_algorithms(True)`

Results are reported as mean ± std across 5 seeds.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{amfta2024,
  title   = {AMFTA: Adaptive Multi-Factor Trust Aggregation for
             Byzantine-Resilient Federated Learning in Non-IID Smart City IoT Networks},
  journal = {IEEE Internet of Things Journal},
  year    = {2024},
  note    = {Under Review}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting pull requests,
reporting issues, and extending the framework with new aggregation methods or attacks.
