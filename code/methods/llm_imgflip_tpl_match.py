#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm.auto import tqdm

from meme_research_eval_utils import (
    compute_metrics,
    finalize_run_timing,
    load_rgb_image,
    load_split_rows,
    now_iso,
    parse_int_list,
    set_seed,
    summarize_batch_timings,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INVALID_TEMPLATE_LABEL = "__INVALID_TEMPLATE__"
DEFAULT_MODEL_ID = "google/gemini-2.5-flash-lite"
# The answer is now a small integer, so this is generous headroom. It exists mainly
# to keep any reasoning/preamble tokens from truncating the JSON. Watch finish_reason
# in the outputs: if you see "length", raise this or disable thinking for the run.
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_RUN_ROOT = SCRIPT_DIR / "SEED" / "runs" / "llm_imgflip_tpl_match"
DEFAULT_TRAIN_PARQUET = SCRIPT_DIR / "splits" / "imgflip_80_20" / "train.parquet"
DEFAULT_TEST_PARQUET = SCRIPT_DIR / "splits" / "imgflip_80_20" / "test.parquet"


SYSTEM_TEXT = """You are evaluating a strict closed-set ImgFlip meme-template classification task.

You will receive one query image and a numbered list of allowed template labels. You may also receive a few
labeled in-context examples before the query image. Use those examples only to understand the task format.

Return ONLY valid JSON with exactly this schema:
{"template_id": <integer index of one allowed template>}

Rules:
1. The template_id must be an integer index taken from the numbered allowed template list.
2. Return only the integer index, not the template name.
3. Do not invent indices outside the allowed range.
4. Do not return NO_TEMPLATE, NON_MEME, unknown, none, or any explanation.
5. Even if uncertain, choose the single allowed template index that best matches the query image.
"""


@dataclass(frozen=True)
class InContextExample:
    image_path: str
    template: str


@dataclass
class LabelSpace:
    """Ordered template vocabulary plus everything needed to build prompts and resolve replies."""

    template_names: list[str]
    list_text: str
    template_to_index: dict[str, int]
    index_to_template: dict[int, str]
    canonical_names: set[str]
    lenient_name_lookup: dict[str, str]


@dataclass
class RunConfig:
    train_parquet: Path
    test_parquet: Path
    output_dir: Path
    image_root: Path | None = None
    path_prefix_from: str | None = None
    path_prefix_to: str | None = None
    model_id: str = DEFAULT_MODEL_ID
    openrouter_api_base: str = DEFAULT_OPENROUTER_API_BASE
    icl_shots: int = 1
    random_seed: int = 42
    seeds: tuple[int, ...] | None = None
    max_test_samples: int | None = None
    max_templates: int | None = None
    request_timeout_retries: int = 5
    request_timeout_seconds: float = 120.0
    retry_sleep_seconds: float = 5.0
    temperature: float = 0.0
    top_p: float = 0.95
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an OpenRouter multimodal LLM as a strict closed-set ImgFlip meme-template classifier."
        )
    )
    parser.add_argument("--train-parquet", default=str(DEFAULT_TRAIN_PARQUET))
    parser.add_argument("--test-parquet", default=str(DEFAULT_TEST_PARQUET))
    parser.add_argument("--output-dir", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--path-prefix-from", default=None)
    parser.add_argument("--path-prefix-to", default=None)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--openrouter-api-base", default=DEFAULT_OPENROUTER_API_BASE)
    parser.add_argument("--icl-shots", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--seeds", type=parse_int_list, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--request-timeout-retries", type=int, default=5)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retry-sleep-seconds", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    args = parser.parse_args()
    return RunConfig(
        train_parquet=Path(args.train_parquet).expanduser().resolve(),
        test_parquet=Path(args.test_parquet).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        image_root=None if args.image_root is None else Path(args.image_root).expanduser().resolve(),
        path_prefix_from=args.path_prefix_from,
        path_prefix_to=args.path_prefix_to,
        model_id=args.model_id,
        openrouter_api_base=args.openrouter_api_base,
        icl_shots=args.icl_shots,
        random_seed=args.random_seed,
        seeds=args.seeds,
        max_test_samples=args.max_test_samples,
        max_templates=args.max_templates,
        request_timeout_retries=args.request_timeout_retries,
        request_timeout_seconds=args.request_timeout_seconds,
        retry_sleep_seconds=args.retry_sleep_seconds,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def discover_openrouter_api_key() -> str:
    for dotenv_path in dict.fromkeys([PROJECT_ROOT / ".env", Path.cwd() / ".env", SCRIPT_DIR / ".env"]):
        load_dotenv_file(dotenv_path)

    value = os.getenv("OPENROUTER_API_KEY")
    if value:
        return value
    raise RuntimeError(
        "Could not find OPENROUTER_API_KEY. Add it to .env or export it in the shell before running this script."
    )


def maybe_limit_templates(df: pd.DataFrame, max_templates: int | None) -> pd.DataFrame:
    if max_templates is None:
        return df.reset_index(drop=True)
    keep_templates = sorted(df["template"].unique().tolist())[:max_templates]
    return df[df["template"].isin(keep_templates)].reset_index(drop=True)


def filter_readable_rows(df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    keep_rows: list[int] = []
    for idx, image_path in enumerate(df["image_path"].tolist()):
        if load_rgb_image(image_path) is not None:
            keep_rows.append(idx)
            if limit is not None and len(keep_rows) >= limit:
                break
    return df.iloc[keep_rows].reset_index(drop=True)


def sample_icl_examples(train_df: pd.DataFrame, shots: int, seed: int) -> list[InContextExample]:
    if shots <= 0:
        return []

    rng = random.Random(seed)
    examples: list[InContextExample] = []
    by_template = {template: group.reset_index(drop=True) for template, group in train_df.groupby("template")}
    template_names = sorted(by_template)
    rng.shuffle(template_names)

    for template in template_names:
        if len(examples) >= shots:
            break
        group = by_template[template]
        indices = list(range(len(group)))
        rng.shuffle(indices)
        for idx in indices:
            image_path = group.iloc[idx]["image_path"]
            if load_rgb_image(image_path) is None:
                continue
            examples.append(InContextExample(image_path=image_path, template=template))
            break

    if len(examples) < shots:
        raise ValueError(f"Requested {shots} ICL shots, but only found {len(examples)} readable examples.")
    return examples


def normalize_label(text: str) -> str:
    """Casefold and collapse every run of non-alphanumeric characters to a single space."""
    lowered = text.strip().casefold()
    collapsed = re.sub(r"[^a-z0-9]+", " ", lowered)
    return collapsed.strip()


def build_lenient_name_lookup(template_names: list[str]) -> dict[str, str]:
    """Map normalized label -> canonical label. Drop keys that two distinct labels share, so we never mismap."""
    lookup: dict[str, str] = {}
    ambiguous: set[str] = set()
    for name in template_names:
        key = normalize_label(name)
        if not key:
            continue
        if key in lookup and lookup[key] != name:
            ambiguous.add(key)
        else:
            lookup[key] = name
    for key in ambiguous:
        lookup.pop(key, None)
    return lookup


def build_label_space(template_names: list[str]) -> LabelSpace:
    ordered = list(template_names)
    list_text = "\n".join(f"[{idx}] {name}" for idx, name in enumerate(ordered))
    return LabelSpace(
        template_names=ordered,
        list_text=list_text,
        template_to_index={name: idx for idx, name in enumerate(ordered)},
        index_to_template={idx: name for idx, name in enumerate(ordered)},
        canonical_names=set(ordered),
        lenient_name_lookup=build_lenient_name_lookup(ordered),
    )


def image_content_from_path(image_path: str | Path) -> dict[str, Any]:
    image_path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}",
        },
    }


def build_messages(
    query_image_path: str,
    label_space: LabelSpace,
    icl_examples: list[InContextExample],
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_TEXT}]

    for shot_idx, example in enumerate(icl_examples, start=1):
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"In-context example {shot_idx}. This image's correct template label is shown "
                            "in the assistant reply."
                        ),
                    },
                    image_content_from_path(example.image_path),
                ],
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps({"template_id": label_space.template_to_index[example.template]}),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Allowed templates (index: label):\n"
                        f"{label_space.list_text}\n\n"
                        "Classify the query image into exactly one allowed template. "
                        "Return JSON with the integer template_id only."
                    ),
                },
                image_content_from_path(query_image_path),
            ],
        }
    )
    return messages


def parse_response_json(response_text: str) -> dict[str, Any]:
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def coerce_template_id(raw_value: Any, index_to_template: dict[int, str]) -> tuple[str, bool]:
    """Resolve an integer template_id to its canonical label. Only accepts a clean integer."""
    if isinstance(raw_value, bool):
        return INVALID_TEMPLATE_LABEL, False
    if isinstance(raw_value, int):
        idx = raw_value
    elif isinstance(raw_value, float) and raw_value.is_integer():
        idx = int(raw_value)
    elif isinstance(raw_value, str):
        stripped = raw_value.strip().strip("[](){}").strip()
        if re.fullmatch(r"-?\d+", stripped):
            idx = int(stripped)
        else:
            return INVALID_TEMPLATE_LABEL, False
    elif isinstance(raw_value, list) and len(raw_value) == 1:
        return coerce_template_id(raw_value[0], index_to_template)
    else:
        return INVALID_TEMPLATE_LABEL, False

    if idx in index_to_template:
        return index_to_template[idx], True
    return INVALID_TEMPLATE_LABEL, False


def resolve_prediction(payload: dict[str, Any], label_space: LabelSpace) -> tuple[str, bool, str]:
    """Return (canonical_label, valid, match_mode).

    Preferred path is the integer template_id. If that is missing or out of range, fall back to matching a
    template name (from "template", or a name accidentally placed in "template_id"): exact first, then a
    normalized lenient match that tolerates case and punctuation/whitespace drift.
    """
    if "template_id" in payload:
        label, valid = coerce_template_id(payload.get("template_id"), label_space.index_to_template)
        if valid:
            return label, True, "index"

    name_candidates: list[str] = []
    for key in ("template", "template_id"):
        value = payload.get(key)
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if isinstance(value, str):
            name_candidates.append(value.strip())

    for candidate in name_candidates:
        if candidate in label_space.canonical_names:
            return candidate, True, "name_exact"
    for candidate in name_candidates:
        norm = normalize_label(candidate)
        if norm and norm in label_space.lenient_name_lookup:
            return label_space.lenient_name_lookup[norm], True, "name_lenient"

    return INVALID_TEMPLATE_LABEL, False, "invalid"


def extract_openrouter_text(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter response did not contain any choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "\n".join(text_parts)
    return str(content)


def extract_finish_reason(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    return str(choices[0].get("finish_reason", "") or "")


def call_openrouter(api_key: str, cfg: RunConfig, messages: list[dict[str, Any]]) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://example.org/"),
        "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "meme-template-id"),
    }
    request_payload = {
        "model": cfg.model_id,
        "messages": messages,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_output_tokens,
    }
    response = requests.post(
        cfg.openrouter_api_base,
        headers=headers,
        json=request_payload,
        timeout=cfg.request_timeout_seconds,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


def generate_prediction(
    api_key: str,
    cfg: RunConfig,
    query_image_path: str,
    label_space: LabelSpace,
    icl_examples: list[InContextExample],
) -> dict[str, Any]:
    messages = build_messages(query_image_path, label_space, icl_examples)
    last_error: Exception | None = None
    last_response_text = ""
    last_usage: dict[str, Any] = {}
    last_finish_reason = ""

    for attempt in range(1, cfg.request_timeout_retries + 1):
        try:
            response_payload = call_openrouter(api_key, cfg, messages)
            last_usage = response_payload.get("usage", {}) if isinstance(response_payload.get("usage"), dict) else {}
            last_finish_reason = extract_finish_reason(response_payload)
            response_text = extract_openrouter_text(response_payload)
            last_response_text = response_text
            payload = parse_response_json(response_text)
            predicted_template, valid_exact, match_mode = resolve_prediction(payload, label_space)
            raw_template_id = payload.get("template_id", "")
            return {
                "raw_response_text": response_text,
                "raw_template_id": "" if raw_template_id is None else raw_template_id,
                "raw_template": payload.get("template", "") or "",
                "predicted_template": predicted_template,
                "template_valid_exact": bool(valid_exact),
                "match_mode": match_mode,
                "finish_reason": last_finish_reason,
                "prompt_tokens": int(last_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(last_usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(last_usage.get("total_tokens", 0) or 0),
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == cfg.request_timeout_retries:
                break
            time.sleep(cfg.retry_sleep_seconds * attempt)

    return {
        "raw_response_text": last_response_text,
        "raw_template_id": "",
        "raw_template": "",
        "predicted_template": INVALID_TEMPLATE_LABEL,
        "template_valid_exact": False,
        "match_mode": "error",
        "finish_reason": last_finish_reason,
        "prompt_tokens": int(last_usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(last_usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(last_usage.get("total_tokens", 0) or 0),
        "error": "" if last_error is None else repr(last_error),
    }


def run_once(cfg: RunConfig, run_dir: Path) -> dict[str, float | int | str]:
    set_seed(cfg.random_seed)
    api_key = discover_openrouter_api_key()
    run_started_at = now_iso()
    start_perf = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run_dir={run_dir}")
    print(f"model_id={cfg.model_id}")

    train_df, test_df = load_split_rows(
        str(cfg.train_parquet),
        str(cfg.test_parquet),
        image_root=None if cfg.image_root is None else str(cfg.image_root),
        path_prefix_from=cfg.path_prefix_from,
        path_prefix_to=cfg.path_prefix_to,
    )
    train_df = maybe_limit_templates(train_df, cfg.max_templates)
    test_df = test_df[test_df["template"].isin(train_df["template"].unique())].reset_index(drop=True)

    test_df = filter_readable_rows(test_df, limit=cfg.max_test_samples)
    if test_df.empty:
        raise ValueError("No readable test images remain after filtering.")

    template_names = sorted(train_df["template"].unique().tolist())
    label_space = build_label_space(template_names)
    icl_examples = sample_icl_examples(train_df, cfg.icl_shots, cfg.random_seed)

    predictions_rows: list[dict[str, Any]] = []
    for row in tqdm(test_df.itertuples(index=False), total=len(test_df), desc="llm classify"):
        row_start = time.perf_counter()
        prediction = generate_prediction(
            api_key=api_key,
            cfg=cfg,
            query_image_path=row.image_path,
            label_space=label_space,
            icl_examples=icl_examples,
        )
        latency_seconds = round(time.perf_counter() - row_start, 3)
        predictions_rows.append(
            {
                "image_path": row.image_path,
                "template_true": row.template,
                "template_pred": prediction["predicted_template"],
                "raw_template_id": prediction["raw_template_id"],
                "raw_template": prediction["raw_template"],
                "template_valid_exact": prediction["template_valid_exact"],
                "match_mode": prediction["match_mode"],
                "finish_reason": prediction["finish_reason"],
                "raw_response_text": prediction["raw_response_text"],
                "error": prediction["error"],
                "latency_seconds": latency_seconds,
                "prompt_tokens": prediction["prompt_tokens"],
                "completion_tokens": prediction["completion_tokens"],
                "total_tokens": prediction["total_tokens"],
            }
        )

    predictions_df = pd.DataFrame(predictions_rows)
    predictions_df.to_csv(run_dir / "test_predictions.csv", index=False)

    y_true = predictions_df["template_true"].to_numpy()
    y_pred = predictions_df["template_pred"].to_numpy()
    metrics = compute_metrics(y_true, y_pred)
    metrics["evaluated_samples"] = int(len(predictions_df))
    metrics["template_count"] = int(len(template_names))
    metrics["icl_shots"] = int(cfg.icl_shots)
    metrics["invalid_prediction_count"] = int((~predictions_df["template_valid_exact"]).sum())
    metrics["template_valid_exact_rate"] = float(predictions_df["template_valid_exact"].mean())
    metrics["error_count"] = int((predictions_df["error"] != "").sum())
    metrics["length_truncation_count"] = int((predictions_df["finish_reason"] == "length").sum())
    metrics["match_mode_counts"] = {
        str(mode): int(count) for mode, count in predictions_df["match_mode"].value_counts().items()
    }
    # Accuracy over just the predictions that resolved to a valid label, so a high invalid
    # rate does not silently masquerade as low model accuracy.
    valid_mask = predictions_df["template_valid_exact"].to_numpy()
    if valid_mask.any():
        metrics["accuracy_on_valid"] = float((y_true[valid_mask] == y_pred[valid_mask]).mean())
    else:
        metrics["accuracy_on_valid"] = 0.0
    metrics["mean_latency_seconds"] = (
        float(predictions_df["latency_seconds"].mean()) if len(predictions_df) else 0.0
    )
    metrics["prompt_tokens"] = int(predictions_df["prompt_tokens"].sum())
    metrics["completion_tokens"] = int(predictions_df["completion_tokens"].sum())
    metrics["total_tokens"] = int(predictions_df["total_tokens"].sum())

    summary_row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "seed": int(cfg.random_seed),
        "model_id": cfg.model_id,
        "accuracy": float(metrics["accuracy"]),
        "accuracy_on_valid": float(metrics["accuracy_on_valid"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "mcc": float(metrics["mcc"]),
        "cohen_kappa": float(metrics["cohen_kappa"]),
        "evaluated_samples": int(metrics["evaluated_samples"]),
        "template_count": int(metrics["template_count"]),
        "icl_shots": int(metrics["icl_shots"]),
        "invalid_prediction_count": int(metrics["invalid_prediction_count"]),
        "template_valid_exact_rate": float(metrics["template_valid_exact_rate"]),
        "error_count": int(metrics["error_count"]),
        "length_truncation_count": int(metrics["length_truncation_count"]),
        "mean_latency_seconds": float(metrics["mean_latency_seconds"]),
        "total_tokens": int(metrics["total_tokens"]),
    }
    run_metadata = {
        "config": json_ready(asdict(cfg)),
        "dataset": {
            "train_images": int(len(train_df)),
            "test_images": int(len(test_df)),
            "template_count": int(len(template_names)),
        },
        "openrouter": {
            "api_base": cfg.openrouter_api_base,
            "model_id": cfg.model_id,
            "api_key_source": "OPENROUTER_API_KEY",
        },
        "template_names": template_names,
        "icl_examples": [asdict(example) for example in icl_examples],
        "test_metrics": metrics,
    }
    timing = finalize_run_timing(run_metadata, summary_row, run_started_at, start_perf)

    (run_dir / "run_config.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    (run_dir / "test_metrics.json").write_text(
        json.dumps({"test_metrics": metrics, "timing": timing}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([summary_row]).to_csv(run_dir / "metrics_summary.csv", index=False)

    print(f"test_accuracy={metrics['accuracy']:.4f}")
    print(f"test_accuracy_on_valid={metrics['accuracy_on_valid']:.4f}")
    print(f"test_f1={metrics['f1']:.4f}")
    print(f"test_mcc={metrics['mcc']:.4f}")
    print(f"invalid_prediction_count={metrics['invalid_prediction_count']}")
    print(f"error_count={metrics['error_count']}")
    print(f"length_truncation_count={metrics['length_truncation_count']}")
    print(f"match_mode_counts={metrics['match_mode_counts']}")
    print(f"duration_seconds={timing['duration_seconds']:.3f}")
    print(f"predictions_saved={run_dir / 'test_predictions.csv'}")
    return summary_row


def main() -> None:
    cfg = parse_args()
    seed_values = cfg.seeds if cfg.seeds is not None else (cfg.random_seed,)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if len(seed_values) == 1:
        single_cfg = RunConfig(**{**asdict(cfg), "random_seed": int(seed_values[0]), "seeds": None})
        run_once(single_cfg, cfg.output_dir / timestamp)
        return

    batch_dir = cfg.output_dir / f"{timestamp}_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_rows: list[dict[str, float | int | str]] = []
    for seed in seed_values:
        print(f"\n=== Running seed {seed} ===")
        seed_cfg = RunConfig(**{**asdict(cfg), "random_seed": int(seed), "seeds": None})
        batch_rows.append(run_once(seed_cfg, batch_dir / f"seed_{int(seed)}"))

    batch_timing = summarize_batch_timings(batch_rows)
    pd.DataFrame(batch_rows).to_csv(batch_dir / "batch_metrics_summary.csv", index=False)
    (batch_dir / "batch_config.json").write_text(
        json.dumps(
            {
                "output_dir": str(cfg.output_dir),
                "batch_dir": str(batch_dir),
                "seeds": [int(seed) for seed in seed_values],
                "config": json_ready(asdict(cfg)),
                "timing": batch_timing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"average_duration_seconds={batch_timing['average_duration_seconds']:.3f}")
    print(f"\nbatch_summary_saved={batch_dir / 'batch_metrics_summary.csv'}")


if __name__ == "__main__":
    main()