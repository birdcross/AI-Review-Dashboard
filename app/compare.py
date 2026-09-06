"""제품/브랜드(카테고리)별 리뷰 비교 분석.

이미 있는 storage.filtered() / analytics.stats() / analytics.theme_counts() 를
여러 대상에 대해 반복 실행해서 나란히 비교할 수 있는 표, 인사이트, 차트,
리포트를 만든다. 새로운 데이터 저장 방식이나 AI 호출은 추가하지 않는다
(비교 자체는 이미 분석·저장된 clean_reviews.jsonl 데이터만 읽는다).
"""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from .analytics import stats, theme_counts
from .dashboard import apply_font
from .shop import product_catalog, short_name


# 제품명에 포함된 키워드로 대략적인 카테고리를 추정한다. 위에서부터 순서대로
# 검사해서 먼저 일치하는 카테고리로 분류한다 (예: "Car Speakers"는 '스피커'가
# 아니라 '차량용 오디오'로 먼저 걸리도록 순서를 잡았다). 원본 CSV의
# categories 컬럼은 파이프라인에 저장되지 않고, 태그도 뒤섞여 있어서
# 정확히 파싱하기 어렵기 때문에, 이미 있는 theme_counts()와 같은 방식으로
# "제품명 키워드 매칭"을 사용한다. 100% 정확한 분류는 아니고 참고용 추정치다.
CATEGORY_KEYWORDS = [
    ('거치대/마운트', ['mount', 'bracket', 'ceiling plate', ' stand']),
    ('차량용 오디오', ['car speaker', 'car audio', 'marine cd', 'cd receiver',
                    'digital media receiver', 'coaxial car', 'amphitheater']),
    ('헤드폰/이어폰', ['headphone', 'earbud', 'earphone']),
    ('스피커', ['speaker', 'soundbar', 'home theater', 'subwoofer', 'mini-system']),
    ('TV/디스플레이', [' tv ', 'led tv', 'smart led', 'lcd/projector']),
    ('키보드/마우스', ['keyboard', 'mouse', 'type cover']),
    ('저장장치/메모리', ['hard drive', 'hdd', 'memory', 'ddr', 'dimm', 'nas ']),
    ('네트워크/통신', ['wi-fi', 'wifi', 'hotspot', 'lte', 'usb adapter']),
    ('배터리/충전기', ['battery', 'charger', 'charging', 'power bank', 'ac adapter']),
    ('카메라/영상', ['camera', 'camcorder', 'video cassette']),
    ('라디오/리모컨', ['radio', 'remote', 'ringer']),
]
UNCATEGORIZED = '기타'


def guess_category(product_name):
    """제품명 키워드로 카테고리를 추정한다. 일치하는 게 없으면 '기타'."""
    text = f' {(product_name or "").lower()} '
    for category, keywords in CATEGORY_KEYWORDS:
        if any(kw in text for kw in keywords):
            return category
    return UNCATEGORIZED


def list_categories(storage):
    """카테고리별 제품 개수를 센다. {카테고리명: 제품 개수} 형태로 반환."""
    catalog = product_catalog(storage.get_clean())
    counts = {}
    for item in catalog:
        cat = guess_category(item['product'])
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def resolve_category(name, available=None):
    """입력한 카테고리 이름을 실제 카테고리 라벨로 정규화한다.

    반환값: (resolved_label 또는 None, candidates 목록)
    """
    labels = available if available is not None else [c for c, _ in CATEGORY_KEYWORDS] + [UNCATEGORIZED]
    name = (name or '').strip()
    if not name:
        return None, []
    exact = next((l for l in labels if l.lower() == name.lower()), None)
    if exact:
        return exact, []
    candidates = [l for l in labels if name.lower() in l.lower()]
    if len(candidates) == 1:
        return candidates[0], []
    return None, candidates


def resolve_targets(storage, kind, names):
    """사용자가 입력한 이름 목록을 실제 product/brand 값으로 정규화한다.

    kind: 'product' 또는 'brand'
    반환값: (resolved, errors)
      resolved -> [{'key': 실제 값, 'input': 사용자가 입력한 원본 문자열}, ...]
      errors   -> [{'input': 원본 문자열, 'reason': 설명, 'candidates': [후보들]}]
    """
    rows = storage.get_clean()

    if kind == 'product':
        pool = sorted({item['product'] for item in product_catalog(rows)})
    else:
        pool = sorted({r['brand'] for r in rows if r.get('brand')})

    resolved = []
    errors = []
    seen_keys = set()

    for raw in names:
        name = (raw or '').strip()
        if not name:
            continue

        exact = next((v for v in pool if v.lower() == name.lower()), None)
        if exact is None:
            candidates = [v for v in pool if name.lower() in v.lower()]
            if len(candidates) == 1:
                exact = candidates[0]
            elif len(candidates) == 0:
                errors.append({'input': raw, 'reason': '일치하는 항목이 없습니다.', 'candidates': []})
                continue
            else:
                errors.append({
                    'input': raw,
                    'reason': '여러 항목과 일치합니다. 더 구체적으로 입력해주세요.',
                    'candidates': candidates[:5],
                })
                continue

        if exact in seen_keys:
            continue
        seen_keys.add(exact)
        resolved.append({'key': exact, 'input': raw})

    return resolved, errors


def build_compare_rows(storage, kind, keys):
    """대상별로 stats()/theme_counts() 를 돌려서 비교용 데이터를 만든다."""
    items = []
    for key in keys:
        rows = storage.filtered({kind: key})
        st = stats(rows)
        items.append({
            'name': key,
            'count': st['total'],
            'analyzed': st['analyzed'],
            'average_rating': st['average_rating'],
            'positive_ratio': st['positive_ratio'],
            'neutral_ratio': st['neutral_ratio'],
            'negative_ratio': st['negative_ratio'],
            'positive_themes': theme_counts(rows, 'positive').most_common(3),
            'negative_themes': theme_counts(rows, 'negative').most_common(3),
        })
    return items


def format_compare_table(items, kind):
    label = '제품' if kind == 'product' else '브랜드'
    name_width = max(10, min(40, max((len(short_name(x['name'], 38)) for x in items), default=10)))
    header = f"{label:<{name_width}} | {'리뷰수':>8} | {'평균별점':>8} | {'긍정%':>7} | {'중립%':>7} | {'부정%':>7}"
    lines = [header, '-' * len(header)]
    for x in items:
        avg = f"{x['average_rating']:.2f}" if x['average_rating'] is not None else '-'
        lines.append(
            f"{short_name(x['name'], name_width):<{name_width}} | {x['count']:>6,}건 | {avg:>8} | "
            f"{x['positive_ratio']:>6.1f}% | {x['neutral_ratio']:>6.1f}% | {x['negative_ratio']:>6.1f}%"
        )
    return '\n'.join(lines)


def build_compare_insights(items, kind):
    label = '제품' if kind == 'product' else '브랜드'
    analyzed_items = [x for x in items if x['analyzed'] > 0]
    if not analyzed_items:
        return ['- 아직 감정 분석이 완료된 리뷰가 없어 비교 인사이트를 만들 수 없습니다. '
                'analyze 명령을 먼저 실행하세요 (테스트용은 --offline 옵션 사용).']

    lines = []
    rated = [x for x in items if x['average_rating'] is not None]
    if rated:
        best_rating = max(rated, key=lambda x: x['average_rating'])
        lines.append(f"- 평균 별점이 가장 높은 {label}: {short_name(best_rating['name'], 40)} "
                     f"({best_rating['average_rating']:.2f})")

    best_pos = max(analyzed_items, key=lambda x: x['positive_ratio'])
    lines.append(f"- 긍정 비율이 가장 높은 {label}: {short_name(best_pos['name'], 40)} "
                 f"({best_pos['positive_ratio']:.1f}%)")

    worst_pos = min(analyzed_items, key=lambda x: x['positive_ratio'])
    lines.append(f"- 긍정 비율이 가장 낮은 {label}: {short_name(worst_pos['name'], 40)} "
                 f"({worst_pos['positive_ratio']:.1f}%) -> 개선 검토 필요")

    worst_neg = max(analyzed_items, key=lambda x: x['negative_ratio'])
    lines.append(f"- 부정 비율이 가장 높은 {label}: {short_name(worst_neg['name'], 40)} "
                 f"({worst_neg['negative_ratio']:.1f}%)")
    return lines


def make_compare_charts(items, out_dir, config, kind):
    apply_font(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = [short_name(x['name'], 18) for x in items]

    # 1) 평균 별점 비교 막대그래프
    ratings = [x['average_rating'] or 0 for x in items]
    plt.figure(figsize=(max(6, len(items) * 1.4), 5))
    plt.bar(labels, ratings, color='#4C72B0')
    plt.title('Average Rating Comparison')
    plt.ylabel('Average Rating')
    plt.ylim(0, 5)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    p1 = out / 'rating_comparison.png'
    plt.savefig(p1, dpi=150)
    plt.close()

    # 2) 감정 비율 비교 누적 막대그래프
    pos = [x['positive_ratio'] for x in items]
    neu = [x['neutral_ratio'] for x in items]
    neg = [x['negative_ratio'] for x in items]
    bottoms = [p + n for p, n in zip(pos, neu)]
    plt.figure(figsize=(max(6, len(items) * 1.4), 5))
    plt.bar(labels, pos, label='positive', color='#55A868')
    plt.bar(labels, neu, bottom=pos, label='neutral', color='#CCB974')
    plt.bar(labels, neg, bottom=bottoms, label='negative', color='#C44E52')
    plt.title('Sentiment Ratio Comparison')
    plt.ylabel('Ratio (%)')
    plt.legend()
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    p2 = out / 'sentiment_comparison.png'
    plt.savefig(p2, dpi=150)
    plt.close()

    return [str(p1), str(p2)]


def build_compare_report(items, kind, category_meta=None):
    label = '제품' if kind == 'product' else '브랜드'
    lines = [
        f'# {label}별 리뷰 비교 분석',
        '',
        f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        '',
    ]
    if category_meta:
        lines += [
            f"카테고리: {category_meta['category']} "
            f"(전체 {category_meta['total_matched']}개 중 상위 {category_meta['shown']}개 비교, "
            f"제품명 키워드 기반 추정 카테고리)",
            '',
        ]
    lines += [f'## 비교 대상 {label}', '']
    lines += [f"- {x['name']} ({x['count']:,}건)" for x in items]
    lines += ['', '## 비교 표', '', '```', format_compare_table(items, kind), '```', '', '## 인사이트']
    lines += build_compare_insights(items, kind)
    lines += ['', f'## {label}별 주요 테마']
    for x in items:
        lines.append(f"### {x['name']}")
        pos = ', '.join(f'{k}({v})' for k, v in x['positive_themes']) or '없음'
        neg = ', '.join(f'{k}({v})' for k, v in x['negative_themes']) or '없음'
        lines.append(f"- 긍정 테마: {pos}")
        lines.append(f"- 부정 테마: {neg}")
        lines.append('')
    return '\n'.join(lines)


def compare(storage, config, kind, names, out_dir=None, make_chart=True):
    """kind('product'|'brand')와 이름 목록을 받아 비교 표/인사이트/차트/리포트를 만든다."""
    resolved, errors = resolve_targets(storage, kind, names)
    if not resolved:
        return {'items': [], 'errors': errors, 'table': '', 'insights': [], 'charts': [], 'report': None, 'meta': {}}

    keys = [r['key'] for r in resolved]
    items = build_compare_rows(storage, kind, keys)
    table = format_compare_table(items, kind)
    insights = build_compare_insights(items, kind)

    charts = []
    report_path = None
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if make_chart:
            charts = make_compare_charts(items, out, config, kind)
        report_text = build_compare_report(items, kind)
        report_path = out / 'compare_report.md'
        report_path.write_text(report_text, encoding='utf-8')

    return {
        'items': items,
        'errors': errors,
        'table': table,
        'insights': insights,
        'charts': charts,
        'report': str(report_path) if report_path else None,
        'meta': {},
    }


def compare_category(storage, config, category_name, limit=5, out_dir=None, make_chart=True):
    """카테고리(스피커, TV/디스플레이 등)로 제품을 자동으로 묶어서 비교한다.

    같은 카테고리 안에서 리뷰가 많은 순으로 정렬한 뒤 상위 `limit`개만 골라
    비교한다 (한 카테고리에 제품이 너무 많으면 표/차트가 지저분해지므로).
    """
    known_labels = [c for c, _ in CATEGORY_KEYWORDS] + [UNCATEGORIZED]
    resolved_label, candidates = resolve_category(category_name, known_labels)
    if not resolved_label:
        reason = '일치하는 카테고리가 없습니다.' if not candidates else '여러 카테고리와 일치합니다. 더 구체적으로 입력해주세요.'
        return {
            'items': [], 'table': '', 'insights': [], 'charts': [], 'report': None, 'meta': {},
            'errors': [{'input': category_name, 'reason': reason, 'candidates': candidates or known_labels}],
        }

    catalog = product_catalog(storage.get_clean())  # 이미 리뷰 많은 순으로 정렬되어 있음
    matched = [item for item in catalog if guess_category(item['product']) == resolved_label]
    total_matched = len(matched)

    if not matched:
        return {
            'items': [], 'table': '', 'insights': [], 'charts': [], 'report': None, 'meta': {},
            'errors': [{'input': category_name, 'reason': f"'{resolved_label}' 카테고리에 해당하는 제품이 없습니다.", 'candidates': []}],
        }

    n = max(1, int(limit)) if limit else len(matched)
    selected = matched[:n]
    keys = [item['product'] for item in selected]

    items = build_compare_rows(storage, 'product', keys)
    table = format_compare_table(items, 'product')
    insights = build_compare_insights(items, 'product')
    meta = {'category': resolved_label, 'total_matched': total_matched, 'shown': len(keys)}

    charts = []
    report_path = None
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if make_chart:
            charts = make_compare_charts(items, out, config, 'product')
        report_text = build_compare_report(items, 'product', category_meta=meta)
        report_path = out / 'compare_report.md'
        report_path.write_text(report_text, encoding='utf-8')

    return {
        'items': items,
        'errors': [],
        'table': table,
        'insights': insights,
        'charts': charts,
        'report': str(report_path) if report_path else None,
        'meta': meta,
    }
