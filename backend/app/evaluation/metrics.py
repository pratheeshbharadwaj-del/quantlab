from __future__ import annotations

import re
from collections import Counter
from typing import Any


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def output_similarity(left: str, right: str) -> float | None:
    a, b = set(normalize(left).split()), set(normalize(right).split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def score_outputs(items: list[dict[str, Any]]) -> dict[str, float | None]:
    usable = [item for item in items if isinstance(item.get("expected_answer"), str)]
    if not usable:
        return {"exact_match": None, "token_f1": None}
    return {
        "exact_match": sum(exact_match(x["output"], x["expected_answer"]) for x in usable) / len(usable),
        "token_f1": sum(token_f1(x["output"], x["expected_answer"]) for x in usable) / len(usable),
    }
