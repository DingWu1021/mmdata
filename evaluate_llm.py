#!/usr/bin/env python3
import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_BASE_URL = "https://antchat.alipay.com/v1"
DEFAULT_MODEL = "Qwen3.5-397B-A17B"
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class EvaluationScores:
    exact_success: float
    component_partial: float


def load_env_file(start: Path) -> None:
    for directory in [start, *start.parents]:
        env_path = directory / ".env"
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
            return


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_id(record: Dict[str, Any], fallback_index: int) -> str:
    value = record.get("id")
    return str(value if value is not None else fallback_index)


def answer_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def accepted_answers(record: Dict[str, Any]) -> List[str]:
    answers = [answer_to_text(record.get("answer", ""))]
    for value in record.get("accepted_answers") or []:
        text = answer_to_text(value)
        if text and text not in answers:
            answers.append(text)
    return answers


def expected_components(record: Dict[str, Any]) -> List[str]:
    values = record.get("intermediate_answers") or []
    return [answer_to_text(value) for value in values if answer_to_text(value)]


def parse_json_object(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("judge response JSON must be an object")
    return obj


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "correct", "pass"}
    return False


def component_bools(value: Any, expected_components: List[str]) -> List[bool]:
    if isinstance(value, list):
        bools = [coerce_bool(item) for item in value]
    elif isinstance(value, dict):
        bools = [coerce_bool(value.get(component, False)) for component in expected_components]
    else:
        bools = []

    if len(bools) < len(expected_components):
        bools.extend([False] * (len(expected_components) - len(bools)))
    return bools[: len(expected_components)]


def compute_scores_from_judge(
    judge_result: Dict[str, Any], expected_components: List[str]
) -> EvaluationScores:
    exact_raw = judge_result.get("exact_success", judge_result.get("exact_match", False))
    exact = 1.0 if coerce_bool(exact_raw) else 0.0
    if not expected_components:
        return EvaluationScores(exact_success=exact, component_partial=exact)

    bools = component_bools(
        judge_result.get("component_correct", judge_result.get("components_correct", [])),
        expected_components,
    )
    partial = sum(1 for item in bools if item) / len(expected_components)
    return EvaluationScores(exact_success=exact, component_partial=partial)


def aggregate_scores(scores: List[EvaluationScores]) -> Dict[str, float]:
    if not scores:
        return {
            "num_examples": 0,
            "exact_success_rate": 0.0,
            "component_partial_score": 0.0,
        }
    return {
        "num_examples": len(scores),
        "exact_success_rate": sum(score.exact_success for score in scores) / len(scores),
        "component_partial_score": sum(score.component_partial for score in scores) / len(scores),
    }


def image_to_data_url(path_or_url: str, base_dir: Path) -> str:
    if re.match(r"^https?://", path_or_url):
        return path_or_url
    path = Path(path_or_url)
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_generation_messages(
    record: Dict[str, Any], question_field: str, image_base_dir: Path, include_images: bool
) -> List[Dict[str, Any]]:
    question = answer_to_text(record.get(question_field) or record.get("question_en") or record.get("question_zh"))
    content: List[Dict[str, Any]] = [{"type": "text", "text": question}]
    if include_images:
        for image_path in record.get("images") or []:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(answer_to_text(image_path), image_base_dir)},
                }
            )

    return [
        {
            "role": "system",
            "content": (
                "You answer multimodal benchmark questions. Use the provided image(s) when present. "
                "Give only the final answer. If the question asks for counted entities, include the "
                "number and the entity names."
            ),
        },
        {"role": "user", "content": content},
    ]


def build_judge_messages(
    record: Dict[str, Any],
    model_answer: str,
    question_field: str,
    expected: List[str],
    components: List[str],
) -> List[Dict[str, str]]:
    question = answer_to_text(record.get(question_field) or record.get("question_en") or record.get("question_zh"))
    payload = {
        "question": question,
        "gold_answers": expected,
        "expected_intermediate_entities": components,
        "candidate_answer": model_answer,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an external benchmark judge. Compare the candidate answer to the gold answer. "
                "exact_success is true only when the final answer is semantically equivalent to one of "
                "the gold answers. For component_correct, return one boolean per expected_intermediate_entities "
                "item, in the same order, marking whether the candidate includes or clearly implies that entity. "
                "If no expected entities are provided, return an empty component_correct list. "
                "Return only JSON with keys: exact_success, component_correct, rationale."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def chat_completion(
    messages: List[Dict[str, Any]],
    *,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    json_mode: bool = False,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"chat completion failed after {retries} attempts: {last_error}") from last_error


def existing_result_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for row in read_jsonl(path):
        if "id" in row:
            ids.add(str(row["id"]))
    return ids


def evaluate_record(
    record: Dict[str, Any],
    index: int,
    args: argparse.Namespace,
    api_key: str,
    judge_api_key: str,
) -> Dict[str, Any]:
    rid = record_id(record, index)
    expected = accepted_answers(record)
    components = expected_components(record)
    generation_messages = build_generation_messages(
        record, args.question_field, args.data.parent, not args.no_images
    )
    model_answer = chat_completion(
        generation_messages,
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.retries,
    )

    judge_messages = build_judge_messages(
        record, model_answer, args.question_field, expected, components
    )
    judge_raw = chat_completion(
        judge_messages,
        model=args.judge_model,
        api_key=judge_api_key,
        base_url=args.judge_base_url,
        temperature=0,
        max_tokens=args.judge_max_tokens,
        retries=args.retries,
        json_mode=True,
    )
    judge_result = parse_json_object(judge_raw)
    scores = compute_scores_from_judge(judge_result, components)
    return {
        "id": rid,
        "question_field": args.question_field,
        "question": answer_to_text(record.get(args.question_field)),
        "gold_answers": expected,
        "expected_intermediate_entities": components,
        "model_answer": model_answer,
        "judge_result": judge_result,
        "scores": asdict(scores),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a multimodal data.jsonl with an answer model and an external LLM judge."
    )
    parser.add_argument("--data", type=Path, default=SCRIPT_DIR / "data.jsonl")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "eval_outputs" / "results.jsonl")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=SCRIPT_DIR / "eval_outputs" / "summary.json",
    )
    parser.add_argument("--question-field", default="question_en")
    parser.add_argument("--model", default=os.getenv("TEXT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL") or os.getenv("TEXT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--judge-base-url", default=os.getenv("JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-api-key-env", default=os.getenv("JUDGE_API_KEY_ENV", "OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--judge-max-tokens", type=int, default=768)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0, help="Zero-based start offset in the JSONL file.")
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in --output.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-images", action="store_true", help="Evaluate as text-only by omitting images.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    load_env_file(Path.cwd())
    load_env_file(Path(__file__).resolve().parent)
    args = parse_args(argv)

    args.data = args.data.resolve()
    args.output = args.output.resolve()
    args.summary_output = args.summary_output.resolve()

    api_key = os.getenv(args.api_key_env)
    judge_api_key = os.getenv(args.judge_api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key: set {args.api_key_env} in environment or .env")
    if not judge_api_key:
        raise SystemExit(f"Missing judge API key: set {args.judge_api_key_env} in environment or .env")
    if not args.data.exists():
        raise SystemExit(f"data file not found: {args.data}")

    seen = existing_result_ids(args.output) if args.resume else set()
    scores: List[EvaluationScores] = []
    attempted = 0
    skipped = 0

    records = list(read_jsonl(args.data))
    selected = records[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    for offset, record in enumerate(selected, start=args.start + 1):
        rid = record_id(record, offset)
        if rid in seen:
            skipped += 1
            continue
        try:
            row = evaluate_record(record, offset, args, api_key, judge_api_key)
            score = EvaluationScores(**row["scores"])
            scores.append(score)
            append_jsonl(args.output, row)
            print(
                f"[{attempted + 1}] id={rid} exact={score.exact_success:.0f} "
                f"component={score.component_partial:.3f}",
                flush=True,
            )
        except Exception as exc:
            error_row = {
                "id": rid,
                "error": str(exc),
                "scores": asdict(EvaluationScores(0.0, 0.0)),
            }
            append_jsonl(args.output, error_row)
            scores.append(EvaluationScores(0.0, 0.0))
            print(f"[{attempted + 1}] id={rid} ERROR: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise
        attempted += 1

    summary = aggregate_scores(scores)
    summary.update(
        {
            "data": str(args.data),
            "output": str(args.output),
            "skipped_existing": skipped,
            "answer_model": args.model,
            "judge_model": args.judge_model,
            "question_field": args.question_field,
            "images_included": not args.no_images,
        }
    )
    write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
