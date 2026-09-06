import logging

from .offline_sentiment import analyze as offline_analyze
from .analytics import stats

logger = logging.getLogger(__name__)


def analyze(storage, config, mode='unanalyzed', review_id=None, force=False,
            limit=None, offline=False, filters=None):
    """Analyze review sentiment and persist sentiment/confidence.

    mode:
      - unanalyzed: only reviews without a sentiment (default)
      - all: all matching reviews when force=True; otherwise only unanalyzed
      - id: one review ID

    filters can narrow by product/brand/date/rating/etc.
    """
    targets = storage.analysis_targets(mode, review_id, force, None)
    if filters:
        targets = [r for r in targets if storage.matches(r, filters)]
    if limit:
        targets = targets[:limit]

    logger.info('분석 대상: %s건', len(targets))
    if not targets:
        return {'success': 0, 'failed': 0, 'target': 0, 'provider': 'none'}

    # Kept only for direct CLI testing. The shopping UI intentionally never
    # falls back to this mode; it requires the real API for AI analysis.
    if offline:
        updates = []
        for i, r in enumerate(targets, 1):
            text = r.get('original_text') or r.get('review_text') or ''
            sentiment, confidence = offline_analyze(text)
            updates.append({
                'id': r['id'],
                'sentiment': sentiment,
                'confidence': confidence,
                'analysis_provider': 'offline_baseline',
            })
            logger.info('[%s/%s] ID=%s 완료: %s (%.2f)',
                        i, len(targets), r['id'], sentiment, confidence)
        storage.save_analysis_many(updates)
        return {
            'success': len(updates), 'failed': 0, 'target': len(targets),
            'provider': 'offline_baseline'
        }

    from .ai_client import AIClient

    client = AIClient(config)
    provider_label = client.provider_label
    batch_size = int(config.get('ai', {}).get('batch_size', 10))
    success = 0
    failed = 0

    for pos in range(0, len(targets), batch_size):
        batch = targets[pos:pos + batch_size]
        try:
            result = client.analyze_batch(batch)
            valid = []
            batch_by_id = {int(r['id']): r for r in batch}

            for item in result:
                try:
                    review_id_value = int(item.get('id', -1))
                    sentiment = item.get('sentiment')
                    confidence = max(0.0, min(1.0, float(item.get('confidence', 0))))
                except (TypeError, ValueError):
                    continue

                if review_id_value not in batch_by_id:
                    continue
                if sentiment not in {'positive', 'negative', 'neutral'}:
                    continue

                valid.append({
                    'id': review_id_value,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'analysis_provider': provider_label,
                })

            storage.save_analysis_many(valid)
            success += len(valid)
            failed += len(batch) - len(valid)

            # Show the user what the AI actually produced, review by review.
            valid_by_id = {x['id']: x for x in valid}
            for offset, row in enumerate(batch, start=1):
                item = valid_by_id.get(int(row['id']))
                current = pos + offset
                if item:
                    logger.info('[%s/%s] ID=%s 분석 완료: %s (%.2f)',
                                current, len(targets), row['id'],
                                item['sentiment'], item['confidence'])
                else:
                    logger.warning('[%s/%s] ID=%s 응답 누락/형식 오류 - 스킵',
                                   current, len(targets), row['id'])

        except Exception as exc:
            failed += len(batch)
            logger.error('API 분석 실패 - 해당 배치 %s건 스킵: %s', len(batch), exc)

    return {
        'success': success,
        'failed': failed,
        'target': len(targets),
        'provider': provider_label,
    }


def extract(storage, config, filters, limit=None, offline=False):
    """Extract keywords/summary/suggestions from matching reviews.

    Real AI extraction uses only reviews that already have sentiment results,
    so the product workflow is sentiment analysis -> insight extraction.
    """
    rows = storage.filtered(filters)
    analyzed_rows = [r for r in rows if r.get('sentiment')]
    if limit:
        analyzed_rows = analyzed_rows[:limit]

    if not analyzed_rows:
        raise ValueError('AI 감정분석이 완료된 리뷰가 없습니다. 먼저 AI 리뷰 분석을 실행하세요.')

    if offline:
        from .analytics import theme_counts
        pos = theme_counts(analyzed_rows, 'positive').most_common(5)
        neg = theme_counts(analyzed_rows, 'negative').most_common(5)
        st = stats(analyzed_rows)
        result = {
            'positive_keywords': [x[0] for x in pos],
            'negative_keywords': [x[0] for x in neg],
            'summary': (
                f"분석된 리뷰 {st['analyzed']}건 기준 긍정 {st['positive_ratio']:.1f}%, "
                f"중립 {st['neutral_ratio']:.1f}%, 부정 {st['negative_ratio']:.1f}%입니다."
            ),
            'suggestions': '부정 리뷰에서 빈도가 높은 제품 품질/연결/배터리/설정 항목을 우선 점검하세요.',
        }
        provider = 'offline_baseline'
    else:
        from .ai_client import AIClient
        client = AIClient(config)
        result = client.extract(analyzed_rows)
        provider = client.provider_label

    item = storage.save_extract(filters, result, provider)
    return item, len(analyzed_rows)
