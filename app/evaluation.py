"""Sampling review and prompt A/B evaluation utilities.

Workflow
--------
1. Export a reproducible sample with ``sample_for_review``.
2. A human fills ``human_sentiment`` with positive/neutral/negative.
3. ``evaluate_review_file`` measures current predictions.
4. ``run_prompt_ab`` sends the same labeled sample through each prompt version
   and stores a per-review comparison CSV plus a JSON summary.
"""

import csv
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

VALID_SENTIMENTS = {'positive', 'neutral', 'negative'}


def _text(row):
    return row.get('review_text') or row.get('original_text') or ''


def _language(text):
    from .offline_sentiment import detect_language
    return detect_language(text)


def sample_for_review(storage, output='output/evaluation/review_sample.csv', sample_size=50, seed=42, filters=None):
    """Export a reproducible, approximately sentiment-stratified review sample."""
    rows = [r for r in storage.filtered(filters or {}) if r.get('sentiment')]
    if not rows:
        raise ValueError('감정 분석이 완료된 리뷰가 없습니다.')

    rng = random.Random(seed)
    groups = {s: [r for r in rows if r.get('sentiment') == s] for s in VALID_SENTIMENTS}
    chosen = []
    target_each = max(1, sample_size // 3)
    for sentiment in ('positive', 'neutral', 'negative'):
        group = groups[sentiment]
        rng.shuffle(group)
        chosen.extend(group[:target_each])

    remaining = [r for r in rows if r not in chosen]
    rng.shuffle(remaining)
    chosen.extend(remaining[:max(0, min(sample_size, len(rows)) - len(chosen))])
    chosen = chosen[:min(sample_size, len(rows))]
    rng.shuffle(chosen)

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'id', 'language', 'product', 'rating', 'review_date', 'text',
        'predicted_sentiment', 'confidence', 'analysis_provider',
        'prompt_version', 'human_sentiment', 'reviewer_note',
    ]
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in chosen:
            writer.writerow({
                'id': row.get('id'),
                'language': row.get('analysis_language') or _language(_text(row)),
                'product': row.get('product'),
                'rating': row.get('rating'),
                'review_date': row.get('review_date'),
                'text': _text(row),
                'predicted_sentiment': row.get('sentiment'),
                'confidence': row.get('confidence'),
                'analysis_provider': row.get('analysis_provider'),
                'prompt_version': row.get('prompt_version') or '',
                'human_sentiment': '',
                'reviewer_note': '',
            })
    return {'path': str(path), 'sample_size': len(chosen), 'seed': seed}


def _accuracy(rows, pred_key):
    labeled = [r for r in rows if r.get('human_sentiment') in VALID_SENTIMENTS and r.get(pred_key) in VALID_SENTIMENTS]
    correct = sum(r[pred_key] == r['human_sentiment'] for r in labeled)
    confusion = Counter((r['human_sentiment'], r[pred_key]) for r in labeled)
    return {
        'labeled': len(labeled),
        'correct': correct,
        'accuracy': correct / len(labeled) if labeled else None,
        'confusion': {f'{truth}->{pred}': n for (truth, pred), n in sorted(confusion.items())},
    }


def evaluate_review_file(path):
    """Evaluate the stored prediction against manually entered labels."""
    with Path(path).open('r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    result = _accuracy(rows, 'predicted_sentiment')
    result['path'] = str(path)
    return result


def run_prompt_ab(config, sample_csv, output_dir='output/evaluation', versions=('v1', 'v2')):
    """Compare prompt versions on the same human-labeled sample without overwriting review data."""
    from .ai_client import AIClient

    with Path(sample_csv).open('r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    labeled = [r for r in rows if r.get('human_sentiment') in VALID_SENTIMENTS]
    if not labeled:
        raise ValueError('human_sentiment가 입력된 검수 샘플이 없습니다.')

    client = AIClient(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    comparison = []
    summary = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'sample_csv': str(sample_csv),
        'provider': client.provider_label,
        'sample_size': len(labeled),
        'versions': {},
    }

    version_predictions = {}
    batch_size = int(config.get('ai', {}).get('batch_size', 10))
    for version in versions:
        preds = {}
        for start in range(0, len(labeled), batch_size):
            batch = labeled[start:start + batch_size]
            api_rows = [{'id': int(r['id']), 'review_text': r['text']} for r in batch]
            for item in client.analyze_batch(api_rows, prompt_version=version):
                if item.get('sentiment') in VALID_SENTIMENTS:
                    preds[str(item['id'])] = item
        version_predictions[version] = preds

        eval_rows = []
        for r in labeled:
            p = preds.get(str(r['id']), {})
            eval_rows.append({**r, 'prediction': p.get('sentiment')})
        summary['versions'][version] = _accuracy(eval_rows, 'prediction')

    for r in labeled:
        item = {
            'id': r['id'], 'language': r.get('language'), 'product': r.get('product'),
            'rating': r.get('rating'), 'text': r.get('text'),
            'human_sentiment': r.get('human_sentiment'),
        }
        for version in versions:
            pred = version_predictions[version].get(str(r['id']), {})
            item[f'{version}_sentiment'] = pred.get('sentiment', '')
            item[f'{version}_confidence'] = pred.get('confidence', '')
        comparison.append(item)

    csv_path = out / f'prompt_ab_{timestamp}.csv'
    json_path = out / f'prompt_ab_{timestamp}.json'
    fields = list(comparison[0].keys())
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison)
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    summary['comparison_csv'] = str(csv_path)
    summary['summary_json'] = str(json_path)
    return summary
