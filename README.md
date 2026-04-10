# CAT-Psych

Dynamic adaptive psychological assessment system based on MIRT/CAT, PyTorch, local Ollama LLM integration, and a lightweight Tkinter desktop chat-flow UI.

## Current Status

- Conda environment: `IPIP`
- PyTorch: CUDA-enabled build verified on RTX 4060 Laptop GPU
- MVP item bank: public-domain IPIP Big-Five 50-item starter set
- Full pulled item bank: official IPIP 3,320-item alphabetical list in `data/ipip_full_item_bank.json`
- Item assignment table: `data/ipip_item_assignment_table.json`
- Mock MIRT/2PL parameters: `data/mock_params.pt`
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

Smoke test:

```powershell
conda run -n IPIP pytest -q
```

## Ethics Notice

This project uses open IPIP data for MVP development. It must not scrape, copy, reconstruct, or reverse-engineer protected MMPI items, proprietary norm tables, or protected clinical scoring mechanisms.

Required user-facing disclaimer:

```text
本系统仅作为心理特质筛查与辅助参考工具，绝对不可替代专业精神科临床诊断。
```
