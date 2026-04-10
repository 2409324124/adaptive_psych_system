# CAT-Psych

Dynamic adaptive psychological assessment system based on MIRT/CAT, PyTorch, local Ollama LLM integration, and a lightweight Tkinter desktop chat-flow UI.

## Current Status

- Conda environment: `IPIP`
- PyTorch: CUDA-enabled build verified on RTX 4060 Laptop GPU
- MVP item bank: public-domain IPIP Big-Five 50-item starter set
- Full pulled item bank: official IPIP 3,320-item alphabetical list in `data/ipip_full_item_bank.json`
- Item assignment table: `data/ipip_item_assignment_table.json`
- Mock MIRT/2PL parameters: `data/mock_params.pt`
- Adaptive routing: coverage-aware item selection keeps early short tests from ignoring low-evidence traits
- Architecture and project memory: `memory/`

## Environment

```powershell
conda env create -f environment.yml
conda activate IPIP
```

For an existing environment:

```powershell
conda env update -n IPIP -f environment.yml --prune
```

## Web App

Start the local FastAPI app:

```powershell
conda activate IPIP
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Data Preparation

Official raw IPIP files are stored under `data/raw/`.

Regenerate normalized JSON/CSV files:

```powershell
conda run -n IPIP python scripts\prepare_ipip_data.py
```

Generated outputs:

- `data/ipip_full_item_bank.json`
- `data/ipip_full_item_bank.csv`
- `data/ipip_item_assignment_table.json`
- `data/ipip_item_assignment_table.csv`

## Adaptive Engine

The first engine pass supports two scoring modes:

- `binary_2pl`: stable MVP path that maps Likert 1-2 to 0, 4-5 to 1, and skips theta updates for neutral 3.
- `grm`: experimental graded response path that derives four ordered thresholds from the current mock `b` parameter at runtime.
- `classical_big5`: traditional IPIP-style baseline scorer that reverse-keys items and averages Likert scores per trait for comparison.

The router now applies a small coverage guard before pure maximum-information selection. By default, it tries to gather at least two answered items per Big-Five dimension before fully relaxing into global Fisher-information routing. The Web result view marks classical trait comparisons with `low evidence` when a trait has fewer than two answered items.

Smoke test:

```powershell
conda run -n IPIP pytest -q
```

Run a small simulation comparing both engine paths:

```powershell
conda run -n IPIP python scripts\simulate_adaptive_sessions.py --max-items 12
```

Run a manual terminal assessment:

```powershell
conda run -n IPIP python scripts\run_cli_assessment.py --model binary_2pl --max-items 12
```

The CLI prints an ASCII-safe disclaimer for Windows `conda run` compatibility; JSON output keeps the original Chinese disclaimer.

Run the CLI in non-interactive demo mode:

```powershell
conda run -n IPIP python scripts\run_cli_assessment.py --demo-responses 5,4,3,2,1 --max-items 5
```

## Ethics Notice

This project uses open IPIP data for MVP development. It must not scrape, copy, reconstruct, or reverse-engineer protected MMPI items, proprietary norm tables, or protected clinical scoring mechanisms.

Required user-facing disclaimer:

```text
本系统仅作为心理特质筛查与辅助参考工具，绝对不可替代专业精神科临床诊断。
```
