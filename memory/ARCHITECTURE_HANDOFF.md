# CAT-Psych Architecture Handoff

Last updated: 2026-04-10 Asia/Taipei

## 1. Project Vision And Positioning

This project uses algorithm engineering and large language models to rethink traditional linear psychological questionnaires. MMPI is the long-term reference target, while the MVP uses the public-domain IPIP/IPIP-NEO item ecosystem.

The system should use multidimensional item response theory (MIRT) for dynamic routing. The goal is to turn long, repetitive questionnaires into a high-dimensional adaptive chat-flow interaction, reducing scale fatigue for young technical users, especially computer-related practitioners.

Core deliverable:

- A cross-platform desktop architecture for Windows 11 and Ubuntu 24 LTS.
- Local-first deployment.
- Python-centered open-source implementation.
- PyTorch for psychometric computation.
- Ollama/local 1.5B model for semantic interpretation.
- Tkinter for lightweight desktop UI.

## 2. Core Technical Architecture

The system must follow strict separation of concerns.

### Algorithm Core Layer: Backend/Math, PyTorch

Responsibilities:

- Represent psychological item banks as high-dimensional tensor matrices.
- Implement the two-parameter logistic model (2PL):

```text
P(theta) = 1 / (1 + exp(-a(theta - b)))
```

- For multidimensional traits, use vectorized PyTorch operations.
- Implement dynamic routing using Fisher Information maximization.
- Select the next item that best reduces uncertainty about the current latent trait estimate theta.
- Treat this as algorithmic pruning rather than a static questionnaire.

Expected location:

```text
engine/
```

### Semantic Parsing Layer: LLM Engine, Ollama + 1.5B Open-Source Base Model

Responsibilities:

- Run locally through Ollama.
- Fit within RTX 4060-class consumer GPU memory constraints.
- Parse free-form user comments, complaints, or explanations from the chat UI.
- Convert natural language into trait-relevant probability weights over Big Five/IPIP dimensions.
- Eventually support SFT examples stored in `data/sft_dataset.jsonl`.

Expected location:

```text
llm/
```

### Interaction Layer: Frontend/UI, Tkinter

Responsibilities:

- Replace rigid web forms with a lightweight desktop chat-flow.
- Render left-aligned system bubbles and right-aligned user bubbles.
- Render Likert-style option buttons and optional free-text input.
- Stay presentation-only.
- Never implement IRT, CAT routing, scoring, or LLM business logic inside UI code.

Expected location:

```text
ui/
```

## 3. Data And Ethics Red Lines

These constraints are mandatory and should be treated as project-level safety requirements.

### Data Isolation

- Do not scrape, copy, reconstruct, or reverse-engineer copyrighted MMPI items.
- Do not scrape or reproduce protected Level C clinical content.
- Do not implement proprietary MMPI norm tables or official clinical scoring mechanisms.
- Do not modify original clinical scales such as F/L/K validity scales or their scoring weights.

### MVP Data Strategy

- Use open IPIP data during MVP development and validation.
- Use random or mock PyTorch tensors for item parameters during engine integration.
- Current mock parameter target can be small, such as `(50, 5)` or expanded later to `(120, 5)`.
- Keep `mock_params.pt` clearly labeled as mock/non-clinical data.

### Medical And Psychological Disclaimer

The UI must visibly hard-code this disclaimer:

```text
本系统仅作为心理特质筛查与辅助参考工具，绝对不可替代专业精神科临床诊断。
```

System outputs should be framed as tendency scores or auxiliary screening results, not clinical diagnoses.

## 4. Current MVP Direction

Short-term MVP target:

- Use the current IPIP 50-item starter bank.
- Use MIRT/2PL-inspired adaptive item routing.
- Use CUDA when available and CPU fallback otherwise.
- Use local Ollama as optional semantic augmentation.
- Build a Tkinter chat interface with clear disclaimer.

Long-term direction:

- Expand from mock `(a, b)` tensors to calibrated item parameters.
- Expand from IPIP starter data to larger open item pools.
- Add SFT examples for Qwen/GLM or another small local model.
- Add validation experiments comparing adaptive routing length against linear questionnaire length.
