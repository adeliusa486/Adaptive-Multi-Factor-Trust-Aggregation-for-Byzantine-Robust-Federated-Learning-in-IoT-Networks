# Byzantine-robust federated intrusion detection on TON_IoT

Code for our study comparing Byzantine-robust aggregation methods for federated
network intrusion detection under realistic non-IID conditions. We benchmark
seven server-side strategies on the real TON_IoT dataset and look at how each one
holds up as the fraction of malicious clients grows.

The short version of what we found: no single method wins everywhere. Density
clustering and trimming do best under mild model-poisoning but fall apart at a 30%
attacker fraction, while a trust-momentum aggregator degrades gracefully. We also
found that the trust-based method does *not* need a trusted validation set on the
server — dropping it actually helps at high attacker fractions.

## Methods compared

- FedAvg (no defense, reference point)
- Trimmed Mean (coordinate-wise)
- Krum
- FLTrust (server reference update)
- FedDBC (density-based clustering)
- AMFTA — multi-factor trust (similarity + reputation EMA + optional quality)
- AMFTA-ND — AMFTA without the trusted validation buffer

## Results (last-5-round mean accuracy, 3 seeds)

Label flipping:

| Method       | 10%  | 20%  | 30%  |
|--------------|------|------|------|
| FedAvg       | 94.3 | 89.3 | 73.8 |
| Trimmed Mean | 92.1 | 91.4 | 83.0 |
| Krum         | 89.7 | 89.5 | 87.3 |
| FLTrust      | 77.4 | 74.6 | 72.1 |
| FedDBC       | 92.1 | 85.6 | 72.0 |
| AMFTA        | 93.1 | 92.8 | 80.3 |
| AMFTA-ND     | 92.5 | 92.3 | 91.7 |

Gaussian noise:

| Method       | 10%  | 20%  | 30%  |
|--------------|------|------|------|
| FedAvg       | 71.3 | 40.8 | 42.8 |
| Trimmed Mean | 94.4 | 57.0 | 41.5 |
| Krum         | 90.0 | 89.9 | 90.3 |
| FLTrust      | 77.3 | 73.6 | 70.3 |
| FedDBC       | 93.9 | 92.8 | 65.3 |
| AMFTA        | 90.3 | 89.5 | 89.3 |
| AMFTA-ND     | 90.9 | 90.6 | 90.6 |

Numbers are produced by `experiments/build_paper_tables.py` directly from the
per-run JSON files under `results/`. Nothing in the tables is hand-edited.

## Setup

```bash
pip install -r requirements.txt
```

The TON_IoT network-flow CSV is not included (it is large). Download it from
UNSW Canberra (https://research.unsw.edu.au/projects/toniot-datasets) and place it
under `data/raw/`, then build the partitions:

```bash
python -m amfta.data.preprocessing
python -m amfta.data.partitioning   # Dirichlet alpha=0.5, 100 clients
```

## Running the experiments

A single configuration:

```bash
python experiments/run_main.py --method amfta --attack label_flipping \
    --byzantine_fraction 0.3 --num_clients 100 --num_rounds 25 --seeds 42
```

The full study (resumable — re-run it and it skips finished configs):

```bash
python experiments/run_focused_study.py
```

Then build the result tables and publication figures:

```bash
python experiments/build_paper_tables.py
python experiments/build_paper_figures.py
```

## Layout

```
amfta/aggregation/   trust factors, AMFTA, and the baseline aggregators
amfta/attacks/       label flipping, gaussian noise, sign flipping, mimicry
amfta/data/          TON_IoT preprocessing and Dirichlet partitioning
training/            the federated training loop
experiments/         experiment drivers and the table builder
```

## Notes

- Seeds used throughout: 42, 123, 456. Each cell is the mean over those three.
- Rounds: 25, local epochs: 5, clients: 100, Dirichlet alpha 0.5.
- We evaluate up to a 30% attacker fraction, consistent with the usual
  honest-majority assumption.
