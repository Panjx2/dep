# Schema-Strict Information Extraction Under Perturbations

Local-first demo for **Synthetic Data Engineering + structured JSON extraction + robustness evaluation**.

## What this does

- Extracts support-ticket fields into strict JSON.
- Validates against a JSON Schema.
- Applies 10 perturbation types (typos, distractors, prompt injection, overload, etc.).
- Compares two modes:
  - `vanilla`: one-shot extraction
  - `repair`: extraction + repair retry on parse/schema failure
- Produces evaluation artifacts:
  - `results/results.csv`
  - `results/summary.csv`
  - `results/failure_taxonomy.csv`

## Schema

The schema lives at `src/schema.json` and enforces:
- enums (`product`, `priority`)
- type constraints
- summary max length
- no additional properties

## Dataset

Base data is in `data/base_tickets.jsonl` with labeled expectations.
Each record has:
- `id`
- `input`
- `expected` (gold JSON)

## Run

```bash
python -m pip install -r requirements.txt
python src/demo.py --backend echo --mode vanilla --out-dir results/vanilla
python src/demo.py --backend echo --mode repair --out-dir results/repair
```

With a local GGUF model via llama.cpp (optional dependency):

```bash
python -m pip install llama-cpp-python
python src/demo.py \
  --backend llama_cpp \
  --model-path /path/to/model.gguf \
  --mode repair \
  --temperature 0.2 \
  --out-dir results/llama_repair
```

## Metrics computed

- JSON parse rate
- Schema compliance
- Field-level scoring (exact for enums/bools, normalized match for strings)
- Set-based F1 for `actions_requested`
- Macro field score
- Failure taxonomy counts

## Perturbations

Implemented perturbation tags:

1. `none`
2. `typos`
3. `unicode_noise`
4. `distractor`
5. `prompt_injection`
6. `contradictory_instruction`
7. `field_aliasing`
8. `order_shuffle`
9. `format_bait`
10. `long_context_overload`

## Suggested experiment story

1. Baseline (`vanilla`) under perturbations.
2. Add repair pipeline (`repair`) and show compliance lift.
3. Compare robustness drops by perturbation type.
4. Inspect `failure_taxonomy.csv` to document dominant failure modes.

## Notes

- Default backend `echo` is deterministic and useful for local pipeline testing without loading a model.
- For portfolio-quality results, run with a local instruct GGUF model and repeat across temperatures/seeds.
