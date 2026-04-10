# Adaptive Psych System Project Memory

Last updated: 2026-04-10 19:36 Asia/Taipei

## Project Goal

Build an adaptive psychological assessment system based on multidimensional item response theory (MIRT/CAT). The long-term reference target is MMPI-style clinical personality assessment, but the project must not use protected MMPI items or proprietary scoring mechanisms. The MVP uses public-domain IPIP/IPIP-NEO-style data.

The product goal is to reduce scale fatigue for young technical users, especially computer-related practitioners, by turning long linear questionnaires into an adaptive chat-flow experience.

The MVP combines:

- A PyTorch-based adaptive item selection engine.
- An IPIP public-domain personality item pool.
- A local Ollama-backed LLM semantic layer for interpreting free-text user input.
- A lightweight Tkinter chat-style UI.

## Workspace

Project root:

```text
D:\IPIP\adaptive_psych_system
```

The root `D:\IPIP` directory is not currently a Git repository.

## Current Directory Structure

```text
adaptive_psych_system/
+-- data/
|   +-- ipip_items.json
|   +-- mock_params.pt
|   +-- sft_dataset.jsonl
+-- engine/
|   +-- __init__.py
|   +-- irt_model.py
|   +-- math_utils.py
+-- llm/
|   +-- __init__.py
|   +-- ollama_client.py
|   +-- prompt_templates.py
+-- memory/
|   +-- PROJECT_MEMORY.md
|   +-- ARCHITECTURE_HANDOFF.md
+-- tests/
|   +-- test_irt.py
|   +-- test_llm.py
+-- ui/
|   +-- __init__.py
|   +-- app.py
|   +-- components.py
+-- AGENT_INSTRUCTIONS.md
+-- environment.yml
+-- main.py
```

## Environment Status

Conda environment name:

```text
IPIP
```

Activation command:

```powershell
conda activate IPIP
```

The environment is managed by Anaconda/Conda only. The current `environment.yml` uses:

```yaml
name: IPIP
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pytorch=2.5.1
  - pytorch-cuda=12.4
  - numpy
  - requests
  - pytest
  - pyyaml
```

## GPU Status

GPU detected:

```text
NVIDIA GeForce RTX 4060 Laptop GPU
```

Validation command result:

```text
torch: 2.5.1
torch cuda build: 12.4
cuda available: True
device count: 1
device name: NVIDIA GeForce RTX 4060 Laptop GPU
gpu matmul ok: (1024, 1024) cuda:0
```

Implementation preference for future engine code:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Use GPU for PyTorch tensor operations when available, but keep CPU fallback for portability.

## Data Status

Prepared data files:

- `data/ipip_items.json`
- `data/mock_params.pt`
- `data/sft_dataset.jsonl`

`ipip_items.json` contains a 50-item IPIP Big-Five starter set with these dimensions:

```text
extraversion
agreeableness
conscientiousness
emotional_stability
intellect
```

Source reference:

```text
https://ipip.ori.org/newBigFive5broadKey.htm
```

`mock_params.pt` contains mock MIRT/2PL parameters:

```text
a shape: (50, 5)
b shape: (50,)
metadata.device: cpu
```

Note: `mock_params.pt` was generated as portable CPU tensors. Runtime code should load it with `map_location=device` or move tensors to the selected device after loading.

`sft_dataset.jsonl` is intentionally empty for now and will be filled later for Qwen/GLM fine-tuning examples.

## Implementation Status

Completed:

- Created the project folder skeleton.
- Created the `IPIP` Conda environment.
- Installed core dependencies.
- Switched PyTorch from CPU-only to CUDA build.
- Verified CUDA is available from PyTorch.
- Added initial IPIP item JSON data.
- Generated mock item parameter tensor file.
- Added this persistent memory document.
- Added architecture handoff document with project vision and ethics constraints.

Not implemented yet:

- `engine/math_utils.py`
- `engine/irt_model.py`
- `llm/ollama_client.py`
- `llm/prompt_templates.py`
- `ui/components.py`
- `ui/app.py`
- `main.py`
- `tests/test_irt.py`
- `tests/test_llm.py`

## Architecture Constraints To Preserve

- Keep algorithmic logic inside `engine/`.
- Keep UI rendering and interaction logic inside `ui/`.
- Keep LLM/Ollama integration inside `llm/`.
- Do not let Tkinter UI directly implement IRT/CAT business logic.
- Use `pathlib` for all filesystem paths.
- Keep code compatible with Windows 11 and Ubuntu.
- Prefer small, testable modules.

## Data And Ethics Constraints

Mandatory red lines:

- Do not scrape, copy, reconstruct, or reverse-engineer protected MMPI items.
- Do not use copyrighted or Level C clinical content.
- Do not implement proprietary MMPI norm tables or official clinical scoring mechanisms.
- Do not modify original clinical scales such as F/L/K validity scales or their scoring weights.
- Keep MVP development on open IPIP data and mock/calibrated open item parameters.
- Treat `data/mock_params.pt` as non-clinical mock data.

Required UI disclaimer:

```text
本系统仅作为心理特质筛查与辅助参考工具，绝对不可替代专业精神科临床诊断。
```

System outputs should be framed as tendency scores or auxiliary screening results, not clinical diagnoses.

## Next Development Steps

1. Implement `engine/math_utils.py`:
   - Logistic sigmoid utilities.
   - Multidimensional 2PL probability calculation.
   - Fisher information approximation.
   - Simple theta update helper.

2. Implement `engine/irt_model.py`:
   - `AdaptiveMMPIRouter` class.
   - Load IPIP item metadata and mock tensors.
   - Track answered item IDs.
   - Select next item by expected information.
   - Update theta from Likert responses.

3. Add tests in `tests/test_irt.py`:
   - Tensor shape validation.
   - Probability range validation.
   - Next-item selection excludes answered items.
   - GPU/CPU device fallback smoke test.

4. Implement `llm/ollama_client.py`:
   - Simple `requests` wrapper for local Ollama.
   - JSON parsing for trait-weight interpretation.
   - Graceful fallback when Ollama is not running.

5. Implement Tkinter MVP:
   - Chat bubbles.
   - Option buttons for Likert responses.
   - Optional free-text input.
   - Main loop in `main.py`.

## Useful Commands

Activate environment:

```powershell
conda activate IPIP
```

Run a GPU smoke test:

```powershell
conda run -n IPIP python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Recreate environment from file:

```powershell
conda env create -f environment.yml
```

Update existing environment from file:

```powershell
conda env update -n IPIP -f environment.yml --prune
```
