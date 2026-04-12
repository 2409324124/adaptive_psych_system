# Adaptive Psych System Project Memory

Last updated: 2026-04-12 15:20 Asia/Taipei

## Project Goal

Build an adaptive psychological assessment system based on multidimensional item response theory (MIRT/CAT). The long-term reference target is MMPI-style clinical personality assessment, but the project must not use protected MMPI items or proprietary scoring mechanisms. The MVP uses public-domain IPIP/IPIP-NEO-style data.

The product goal is to reduce scale fatigue for young technical users, especially computer-related practitioners, by turning long linear questionnaires into an adaptive chat-flow experience.

The MVP combines:

- A PyTorch-based adaptive item selection engine.
- An IPIP public-domain personality item pool.
- A local Ollama-backed LLM semantic layer for interpreting free-text user input.
- A FastAPI + Web MVP for browser-based interaction, with Tkinter kept as a later desktop target.

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
|   +-- ipip_full_item_bank.json
|   +-- ipip_full_item_bank.csv
|   +-- ipip_item_assignment_table.json
|   +-- ipip_item_assignment_table.csv
|   +-- benchmark_stopping_rules_current.json
|   +-- benchmark_stopping_rules_keyed.json
|   +-- mock_params.pt
|   +-- mock_params_keyed.pt
|   +-- simulation_matrix_current.json
|   +-- simulation_matrix_keyed.json
|   +-- raw/
|   +-- sft_dataset.jsonl
+-- engine/
|   +-- __init__.py
|   +-- classical_scoring.py
|   +-- irt_model.py
|   +-- math_utils.py
+-- llm/
|   +-- __init__.py
|   +-- ollama_client.py
|   +-- prompt_templates.py
+-- services/
|   +-- __init__.py
|   +-- assessment_session.py
+-- api/
|   +-- __init__.py
|   +-- app.py
+-- docs/
|   +-- adaptive_measurement_vision.md
+-- web/
|   +-- index.html
|   +-- style.css
|   +-- app.js
+-- memory/
|   +-- PROJECT_MEMORY.md
|   +-- ARCHITECTURE_HANDOFF.md
+-- scripts/
|   +-- benchmark_stopping_rules.py
|   +-- generate_key_aware_mock_params.py
|   +-- prepare_ipip_data.py
|   +-- run_cli_assessment.py
|   +-- simulate_adaptive_sessions.py
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
  - fastapi
  - uvicorn
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
- `data/ipip_full_item_bank.json`
- `data/ipip_full_item_bank.csv`
- `data/ipip_item_assignment_table.json`
- `data/ipip_item_assignment_table.csv`
- `data/mock_params.pt`
- `data/mock_params_keyed.pt`
- `data/benchmark_stopping_rules_current.json`
- `data/benchmark_stopping_rules_keyed.json`
- `data/simulation_matrix_current.json`
- `data/simulation_matrix_keyed.json`
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

The full official IPIP alphabetical item list was pulled into:

```text
data/raw/ipip_alphabetical_item_list.html
```

Normalized outputs:

```text
data/ipip_full_item_bank.json
data/ipip_full_item_bank.csv
```

Validation:

```text
item_count: 3320
source_url: https://ipip.ori.org/AlphabeticalItemList.htm
```

The IPIP item assignment table was pulled into:

```text
data/raw/TedoneItemAssignmentTable30APR21.xlsx
```

Normalized outputs:

```text
data/ipip_item_assignment_table.json
data/ipip_item_assignment_table.csv
```

Validation:

```text
row_count: 3805
headers: instrument, alpha, key, text, label
source_url: https://ipip.ori.org/TedoneItemAssignmentTable30APR21.xlsx
```

`mock_params.pt` contains mock MIRT/2PL parameters:

```text
a shape: (50, 5)
b shape: (50,)
metadata.device: cpu
```

Note: `mock_params.pt` was generated as portable CPU tensors. Runtime code should load it with `map_location=device` or move tensors to the selected device after loading.

`mock_params_keyed.pt` contains a newer mock parameter set with metadata flag:

```text
metadata.key_aligned: True
```

This means reverse-keyed items are handled at the parameter layer, so runtime code must not also flip the response a second time.

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
- Pulled official IPIP raw item files into `data/raw/`.
- Added `scripts/prepare_ipip_data.py` to normalize official IPIP HTML/XLSX files.
- Generated full 3,320-item IPIP JSON/CSV files and 3,805-row assignment table JSON/CSV files.
- Implemented the first adaptive routing engine in `engine/`.
- Added both `binary_2pl` and experimental `grm` scoring modes.
- Added `ClassicalBigFiveScorer`, a traditional IPIP-style baseline that reverse-keys Likert items and averages per trait.
- Added IRT engine tests and `pytest.ini`.
- Added `scripts/simulate_adaptive_sessions.py` to compare `binary_2pl`, `grm`, and classical Big Five outputs on fixed simulated personas.
- Added `scripts/run_cli_assessment.py`, a manual terminal assessment demo with disclaimer, 1-5 input, IRT scores, classical Big Five comparison, and optional JSON output.
- Added `services/assessment_session.py` as the shared assessment workflow layer for CLI/API/Web.
- Added `api/app.py` with FastAPI session endpoints.
- Added minimal Web app under `web/`.
- Added FastAPI/API tests.
- Added coverage-aware routing so short adaptive sessions collect minimum early evidence across Big-Five dimensions before pure max-information selection.
- Added Web result labeling for classical comparison traits with low answered-item evidence.
- Added `SessionStore` with in-memory and optional JSON-backed persistence modes.
- Added session timeout cleanup, export endpoint, summary endpoint, restart endpoint, and delete endpoint.
- Added session snapshot/replay support by rebuilding router state from answered-path history.
- Added a rules-based result interpretation layer with overview, higher/lower tendency summaries, and low-evidence cautions.
- Result page now shows interpretation, coverage cards, IRT/classical score panels, export, and restart flow in the Web UI.
- Confirmed local export flow works with generated JSON result files.
- Added `docs/adaptive_measurement_vision.md` as the canonical design note for product thesis, ethics boundary, technical direction, and roadmap.
- Added Fisher information matrix accumulation, covariance approximation, standard errors, and uncertainty summaries.
- Added early stopping based on `min_items`, coverage, and mean standard error threshold.
- Added a Web confidence panel showing mean standard error, readiness flag, and per-trait standard errors.
- Fixed reverse-key handling in the IRT update path so reverse-keyed items now affect theta in the correct direction.
- Fixed session flow so the current routed item is cached as `active_item` instead of re-calling `select_next_item()` during response validation.
- Fixed snapshot restore consistency so replayed `theta_after` values match the live router state.
- Added `scripts/benchmark_stopping_rules.py` for reproducible stopping-rule experiments.
- Added `scripts/generate_key_aware_mock_params.py` to create parameter files aligned with reverse-key metadata.
- Added `data/mock_params_keyed.pt` as a key-aware mock parameter baseline.
- Added JSON benchmark/simulation result snapshots for both the old and keyed parameter files.
- Fixed a double-flip bug by making response flipping conditional on `param_metadata.key_aligned`.
- Added a shared parameter-mode resolver with dual-track support:
  - `legacy` -> `mock_params.pt`
  - `keyed` -> `mock_params_keyed.pt`
- Added API/session/export visibility for `param_mode`, `param_path`, `key_aligned`, and raw parameter metadata.
- Added stop-rule breakdown in session progress:
  - `min_items_met`
  - `coverage_ready`
  - `standard_error_ready`
  - `stopped_by`
- Unified benchmark/simulation outputs with timestamp, script version, parameter metadata, and richer persona summaries.
- Added `scripts/compare_param_modes.py` for fixed legacy-vs-keyed comparison bundles.
- Web results now show experiment context, parameter mode, and stop-rule breakdown.
- Current automated test status after the latest fixes: `35 passed`.

Implemented engine modules:

- `engine/math_utils.py`
- `engine/irt_model.py`
- `engine/classical_scoring.py`

Not implemented yet:

- `llm/ollama_client.py`
- `llm/prompt_templates.py`
- `ui/components.py`
- `ui/app.py`
- `main.py`
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

Encoding note:

- Some Windows terminal reads inside the agent environment may display UTF-8 Chinese text as mojibake.
- Treat this as a console/display issue unless the user reports that the file itself is broken in GitHub or the browser.
- Do not "fix" stored Chinese disclaimer text based only on agent-side terminal output.

## Next Development Steps

1. Use simulation output to tune model defaults:
   - `binary_2pl` currently remains the default adaptive path.
   - Re-run comparisons with both `mock_params.pt` and `mock_params_keyed.pt` when evaluating stopping behavior.
   - `classical_big5` remains the sanity-check baseline on routed item subsets.
   - Preserve coverage-aware routing and early stopping summaries in all future experiments.

2. Continue tuning the local Web app:
   - `uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload`
   - Open `http://127.0.0.1:8000`
   - Check item flow, score display, model choice, low-evidence labels, session restart/export behavior, interpretation wording, and confidence/early-stop messaging.

3. Refine the human-readable interpretation layer:
   - Make wording less engineering-heavy and more like a psychological feedback summary.
   - Consider trait-pair or style-pattern summaries instead of only per-trait highs/lows.
   - Keep the current rules-based layer as the fallback baseline.
   - Optionally blend with LLM output later, but preserve deterministic fallback.

4. Implement `llm/ollama_client.py`:
   - Simple `requests` wrapper for local Ollama.
   - JSON parsing for trait-weight interpretation.
   - Graceful fallback when Ollama is not running.

5. Keep the current dual-track parameter strategy:
   - `legacy` remains the default for compatibility.
   - `keyed` is the more realistic experimental baseline.
   - Revisit flipping the default only after benchmark/output stability and Web experiment visibility are both in good shape.

6. Implement Tkinter MVP:
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

Run tests:

```powershell
conda run -n IPIP pytest -q
```

Run FastAPI Web app:

```powershell
conda activate IPIP
uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

Regenerate IPIP normalized data:

```powershell
conda run -n IPIP python scripts\prepare_ipip_data.py
```

Run adaptive-routing simulation:

```powershell
conda run -n IPIP python scripts\simulate_adaptive_sessions.py --max-items 12
```

Run manual CLI assessment:

```powershell
conda run -n IPIP python scripts\run_cli_assessment.py --model binary_2pl --max-items 12
```

Recreate environment from file:

```powershell
conda env create -f environment.yml
```

Update existing environment from file:

```powershell
conda env update -n IPIP -f environment.yml --prune
```
