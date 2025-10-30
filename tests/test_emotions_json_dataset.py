import json
from pathlib import Path


def test_emotions_jsonl_is_readable_and_nonempty():
    p = Path("/Volumes/AURA_RESOURCES/aura_tpu/data/json/emotions.jsonl")
    assert p.exists(), "emotions.jsonl file is missing"
    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0, "emotions.jsonl appears empty"
    # Parse a few lines to validate JSONL integrity
    for i, line in enumerate(lines[:10]):
        obj = json.loads(line)
        assert isinstance(obj, dict), "Each line should be a JSON object"
        # Optional soft checks on expected keys if present
        # e.g., 'text' and 'label' or similar; keep loose to avoid coupling
