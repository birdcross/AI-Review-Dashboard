from collections import Counter
from datetime import datetime, timedelta
import re

STOP = set("the and for with this that was are but not have from you your they them its our my i we he she to of in on is it a an as at be been being or if so very just one all can could would should will has had do does did about than then when where what which who how up out over into after before more most some any each other only also really much too very product use used using get got make made bought buy new time because there were still want need".split())


def stats(rows):
    total = len(rows)
    analyzed = [r for r in rows if r.get('sentiment')]
    sentiments = Counter(r.get('sentiment') for r in analyzed)
    ratings = [float(r['rating']) for r in rows if r.get('rating') is not None]
    conf = [float(r['confidence']) for r in analyzed if r.get('confidence') is not None]
    rating_sent = []
    agree = 0
    for r in analyzed:
        if r.get('rating') is None:
            continue
        rt = 'positive' if float(r['rating']) >= 4 else 'negative' if float(r['rating']) <= 2 else 'neutral'
        rating_sent.append(rt)
        agree += int(rt == r['sentiment'])
    return {
        'total': total,
        'analyzed': len(analyzed),
        'analysis_rate': len(analyzed) / total * 100 if total else 0,
        'positive': sentiments['positive'],
        'neutral': sentiments['neutral'],
        'negative': sentiments['negative'],
        'positive_ratio': sentiments['positive'] / len(analyzed) * 100 if analyzed else 0,
        'neutral_ratio': sentiments['neutral'] / len(analyzed) * 100 if analyzed else 0,
        'negative_ratio': sentiments['negative'] / len(analyzed) * 100 if analyzed else 0,
        'rating_counts': Counter(int(round(x)) for x in ratings),
        'average_rating': sum(ratings) / len(ratings) if ratings else None,
        'average_confidence': sum(conf) / len(conf) if conf else None,
        'rating_sentiment_agreement': agree / len(rating_sent) * 100 if rating_sent else None,
        'low_rating_ratio': sum(x <= 2 for x in ratings) / len(ratings) * 100 if ratings else 0,
    }


def top_words(rows, sentiment=None, n=8):
    cnt = Counter()
    for r in rows:
        if sentiment and r.get('sentiment') != sentiment:
            continue
        text = r.get('original_text') or r.get('review_text') or ''
        words = [w for w in re.findall(r"[a-z][a-z'-]+", text.lower()) if len(w) > 2 and w not in STOP]
        cnt.update(words)
    return cnt.most_common(n)


def theme_counts(rows, sentiment=None):
    # English + Korean theme aliases allow the same business insights for both languages.
    themes = {
        '연결/블루투스': ['bluetooth', 'connection', 'connect', 'disconnect', 'pairing', 'wifi', 'wi-fi', '블루투스', '연결', '끊김', '끊겨'],
        '배터리': ['battery', 'charge', 'charging', '배터리', '충전'],
        '음질/사운드': ['sound', 'audio', 'bass', 'speaker', 'volume', '음질', '사운드', '스피커', '볼륨'],
        '화면/터치': ['screen', 'display', 'touch', 'picture', '화면', '디스플레이', '터치'],
        '버튼/리모컨': ['remote', 'button', 'buttons', '리모컨', '버튼'],
        '설정/설치': ['setup', 'set up', 'install', 'program', '설정', '설치'],
        '가격': ['price', 'expensive', 'money', 'value', '가격', '비싸', '가성비'],
        '품질/고장': ['quality', 'broken', 'defective', 'fail', 'failed', 'problem', 'issue', '품질', '고장', '불량', '문제', '오류'],
        '착용감': ['comfortable', 'fit', 'headphones', 'earbuds', '착용', '편안', '이어폰', '헤드폰'],
    }
    counts = Counter()
    for r in rows:
        if sentiment and r.get('sentiment') != sentiment:
            continue
        text = (r.get('original_text') or r.get('review_text') or '').lower()
        for theme, keys in themes.items():
            if any(k in text for k in keys):
                counts[theme] += 1
    return counts


def _parse_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(str(value)[:10], '%Y-%m-%d')
    except (TypeError, ValueError):
        return None


def _period_rows(rows, start, end):
    result = []
    for r in rows:
        d = _parse_date(r.get('review_date'))
        if d and start <= d <= end and r.get('sentiment'):
            result.append(r)
    return result


def _negative_ratio(rows):
    return sum(r.get('sentiment') == 'negative' for r in rows) / len(rows) if rows else None


def _average_rating(rows):
    ratings = []
    for r in rows:
        try:
            if r.get('rating') is not None:
                ratings.append(float(r['rating']))
        except (TypeError, ValueError):
            pass
    return sum(ratings) / len(ratings) if ratings else None


def recent_negative_alert(rows, recent_days=30, threshold=0.30, increase_threshold=0.10):
    """Detect a recent negative-review surge versus the immediately previous N days.

    ``threshold`` is an absolute negative-ratio guardrail and ``increase_threshold``
    is the minimum rise versus the previous period (0.10 == 10 percentage points).
    If no previous-period data exists, the absolute threshold alone is used.
    """
    dates = [_parse_date(r.get('review_date')) for r in rows]
    dates = [d for d in dates if d]
    if not dates:
        return None

    end = max(dates)
    current_start = end - timedelta(days=recent_days - 1)
    previous_end = current_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=recent_days - 1)

    current = _period_rows(rows, current_start, end)
    previous = _period_rows(rows, previous_start, previous_end)
    if not current:
        return None

    current_ratio = _negative_ratio(current)
    previous_ratio = _negative_ratio(previous)
    change = None if previous_ratio is None else current_ratio - previous_ratio

    if previous_ratio is None:
        warning = current_ratio >= threshold
        reason = f'이전 비교기간 데이터가 없어 최근 부정비율 {current_ratio * 100:.1f}%를 절대 기준으로 판단했습니다.'
    else:
        warning = current_ratio >= threshold and change >= increase_threshold
        reason = (
            f'최근 부정비율 {current_ratio * 100:.1f}% / 이전 {previous_ratio * 100:.1f}% / '
            f'증가폭 {change * 100:+.1f}%p'
        )

    current_rating = _average_rating(current)
    previous_rating = _average_rating(previous)
    rating_change = None
    if current_rating is not None and previous_rating is not None:
        rating_change = current_rating - previous_rating

    negative_current = [r for r in current if r.get('sentiment') == 'negative']
    top_products = Counter((r.get('product') or 'Unknown') for r in negative_current).most_common(3)
    top_themes = theme_counts(negative_current).most_common(3)

    hypotheses = []
    if rating_change is not None and rating_change <= -0.3:
        hypotheses.append(f'평균 별점이 이전 기간보다 {abs(rating_change):.2f}점 하락하여 전반적 만족도 저하 가능성이 있습니다.')
    if top_products:
        name, count = top_products[0]
        share = count / max(1, len(negative_current))
        if share >= 0.35:
            hypotheses.append(f"부정 리뷰의 {share * 100:.1f}%가 '{name}'에 집중되어 특정 제품 이슈 가능성이 있습니다.")
    if top_themes:
        theme, count = top_themes[0]
        hypotheses.append(f"최근 부정 리뷰에서 '{theme}' 테마가 {count}건으로 가장 많아 우선 원인 후보로 확인할 수 있습니다.")
    if not hypotheses:
        hypotheses.append('특정 단일 원인은 확인되지 않았으며 제품/기간/별점/테마별 세부 비교가 필요합니다.')

    follow_up = [
        '최근 기간과 이전 기간의 제품별 부정비율을 비교합니다.',
        '신제품 출시일·펌웨어/앱 버전 변경일과 부정 리뷰 증가 시점을 대조합니다.',
        '가격/프로모션 변경 전후의 별점과 부정 키워드 변화를 비교합니다.',
        '최근 부정 리뷰의 TOP 테마 원문을 샘플 검수하여 실제 원인을 확인합니다.',
    ]

    return {
        # Backward-compatible keys
        'start': current_start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d'),
        'count': len(current),
        'negative_ratio': current_ratio,
        'warning': warning,
        # Added comparison/diagnostics
        'previous_start': previous_start.strftime('%Y-%m-%d'),
        'previous_end': previous_end.strftime('%Y-%m-%d'),
        'previous_count': len(previous),
        'previous_negative_ratio': previous_ratio,
        'increase': change,
        'threshold': threshold,
        'increase_threshold': increase_threshold,
        'average_rating': current_rating,
        'previous_average_rating': previous_rating,
        'rating_change': rating_change,
        'top_negative_products': top_products,
        'top_negative_themes': top_themes,
        'reason': reason,
        'hypotheses': hypotheses,
        'follow_up_actions': follow_up,
    }
