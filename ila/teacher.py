"""A scripted curriculum or one curriculum request to a local Ollama model."""

import json
from urllib.request import Request, urlopen

MODES = ("random", "novelty", "uncertainty")
SCRIPTED = {
    "goal": "多尝试不同观测与动作的组合，收集移动和停留的结果。",
    "first": "novelty",
    "second": "uncertainty",
}


def validate(data):
    if not isinstance(data, dict) or set(data) != set(SCRIPTED):
        raise ValueError("Expected teacher JSON with goal, first, and second.")
    if not isinstance(data["goal"], str) or not data["goal"].strip():
        raise ValueError("Teacher goal must be a nonempty string.")
    if data["first"] not in MODES or data["second"] not in MODES:
        raise ValueError(f"Teacher strategies must be among {MODES}.")
    return data


def ask_ollama(model):
    schema = {
        "type": "object",
        "properties": {
            "goal": {"type": "string"},
            "first": {"type": "string", "enum": list(MODES)},
            "second": {"type": "string", "enum": list(MODES)},
        },
        "required": list(SCRIPTED),
        "additionalProperties": False,
    }
    prompt = (
        "Design a two-stage exploration curriculum for a grid-world student that must "
        "learn which displacement follows a categorical 3x3 observation and opaque action. "
        "The first map is random warm-up. Select one strategy per stage from random "
        "(uniform actions), novelty (rarer observation/action combinations), uncertainty "
        "(higher outcome entropy). Guidance fades during the first half of training. "
        "Give a short learning goal in Chinese and answer using this schema: "
        + json.dumps(schema)
    )
    payload = {
        "model": model, "stream": False, "format": schema, "options": {"temperature": 0},
        "messages": [{"role": "user", "content": prompt}],
    }
    request = Request(
        "http://127.0.0.1:11434/api/chat", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        content = json.load(response)["message"]["content"]
    return validate(json.loads(content))
