# LLM Evaluation

Evaluate `data.jsonl` by calling one LLM to answer each item and another LLM as an external judge.

## Run

From the repository root:

```bash
python3 llm_eval/evaluate_llm.py
```

By default, the script reads `llm_eval/data.jsonl`, image paths under `llm_eval/images/`, and writes outputs to `llm_eval/eval_outputs/`.

The script reads `OPENAI_API_KEY` from the environment or the repository `.env` file. It defaults to the existing OpenAI-compatible gateway:

- `base_url`: `https://antchat.alipay.com/v1`
- `model`: `TEXT_MODEL` from `.env`, or `Qwen3.5-397B-A17B`
- `judge_model`: `JUDGE_MODEL`, or `TEXT_MODEL`

## Metrics

- `exact_success_rate`: average exact final-answer success.
- `component_partial_score`: average intermediate-entity accuracy. If an item has no intermediate entities, this falls back to exact success.

## Test

```bash
python3 -m pytest llm_eval
```
