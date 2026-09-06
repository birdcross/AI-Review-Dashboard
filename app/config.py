import json
import logging
import os
from pathlib import Path

DEFAULT_CONFIG = {
    "storage": {
        "raw_path": "data/raw_reviews.jsonl",
        "clean_path": "data/clean_reviews.jsonl",
        "extracts_path": "data/extracts.jsonl"
    },
    "columns": {
        "text": "reviews.text_ko",
        "original_text": "reviews.text",
        "rating": "reviews.rating",
        "date": "reviews.date",
        "product": "name",
        "brand": "brand",
        "source_id": "id"
    },
    "duplicate_policy": "skip",
    "clean": {"min_length": 5},
    "ai": {
        "provider": "gemini",
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/models",
        "model": "gemini-2.5-flash",
        "timeout_seconds": 90,
        "batch_size": 10,
        "extract_limit": 120
    },
    "visualization": {
        "output_dir": "output",
        "font_candidates": ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    },
    "logging": {"level": "INFO", "file": "logs/app.log"},
    "alert": {"recent_days": 30, "negative_ratio_threshold": 0.30}
}

def deep_merge(base, extra):
    result = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def load_config(path="config.json"):
    p = Path(path)
    if not p.exists():
        p.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
        return DEFAULT_CONFIG
    user = json.loads(p.read_text(encoding="utf-8"))
    return deep_merge(DEFAULT_CONFIG, user)

def setup_logging(config):
    cfg = config.get("logging", {})
    level = getattr(logging, str(cfg.get("level", "INFO")).upper(), logging.INFO)
    log_file = Path(cfg.get("file", "logs/app.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
