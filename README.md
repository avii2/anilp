# BBB-FLIDS (Clustered)

Basic Blockchain-Based Federated Learning Intrusion Detection System with optional clustered federated averaging.

- Federated IDS built on PyTorch and NumPy
- Dummy or Ethereum blockchain backends (via `eth-tester`)
- Optional data-standardization coordination
- Configurable clustered FL (3–4 clients per cluster by default) to reduce on-chain writes

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r fl-requirements.txt   # or requirements.txt for full stack
python install_solc.py               # only if using Ethereum backend
python main.py
```

Key settings live in `config.ini`:

| Section | Field | Notes |
| --- | --- | --- |
| `[FL]` | `platform` | `dummy` for fast local runs, `eth` for on-chain |
|  | `num users` | Total simulated clients |
|  | `cluster size` | Clients per cluster (`1` disables clustering) |
|  | `training fraction` | Fraction of (clusters) training per round |
|  | `preprocessing fraction` | Fraction sending means/stds |
| `[MODEL]` | `name` | Class inside `ModelConfig.py` |

## How It Works

1. **Data prep** – dataset is split evenly across clients (`dataset/`, `split_data_equal`). Clients optionally participate in a two-stage mean/std aggregation step to standardize features globally.
2. **Local training** – each client trains the configured neural network (`ModelConfig`) using SGD for `LOCAL_EPOCHS`, normalizing batches with the shared statistics.
3. **Cluster aggregation** – when `cluster size > 1`, `ClientCluster` groups a few clients, averages their local model weights offline, and submits a single weighted update to the blockchain.
4. **Blockchain coordination** – the server (contract owner) collects `LocalUpdate` events, weights them by reported sample counts, averages the models, and writes the new global model back via `globalUpdate`. The smart contract also stores global means/stds for reuse.
5. **Evaluation** – optionally measure validation accuracy/loss after each round and append results to `results.json`. `plot.py` can visualize past runs.

## Scripts

- `main.py` – orchestrates federated/clustered training
- `FL.sol` – Solidity smart contract
- `DummyPlatform.py` / `EthPlatform.py` – blockchain backends
- `plot.py` – plot historical accuracies/losses
- `TestContract.py`, `StressTestContract.py` – basic contract tests

## Extending

- Add new models in `ModelConfig.py` and reference them in `config.ini`
- Adjust clustering or turn it off (`cluster size = 1`)
- Swap datasets by pointing `dataset path` to your CSV
- Plug in a different platform by implementing the same interface as `DummyPlatform`
