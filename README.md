# CAT-Psych

Dynamic adaptive psychological assessment system based on MIRT/CAT, PyTorch, local Ollama LLM integration, and a lightweight Tkinter desktop chat-flow UI.

## Current Status

- Conda environment: `IPIP`
- PyTorch: CUDA-enabled build verified on RTX 4060 Laptop GPU
- MVP item bank: public-domain IPIP Big-Five 50-item starter set
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

## Ethics Notice

This project uses open IPIP data for MVP development. It must not scrape, copy, reconstruct, or reverse-engineer protected MMPI items, proprietary norm tables, or protected clinical scoring mechanisms.

Required user-facing disclaimer:

```text
本系统仅作为心理特质筛查与辅助参考工具，绝对不可替代专业精神科临床诊断。
```
