from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import base64
import html as html_lib
import mimetypes

import matplotlib.pyplot as plt
from matplotlib import font_manager

from .analytics import stats, theme_counts, recent_negative_alert


def apply_font(config):
    candidates = config.get('visualization', {}).get('font_candidates', [])
    available = {f.name for f in font_manager.fontManager.ttflist}
    selected = next((x for x in candidates if x in available), 'DejaVu Sans')
    plt.rcParams['font.family'] = selected
    plt.rcParams['axes.unicode_minus'] = False
    return selected


def make_charts(rows, out_dir, config):
    apply_font(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) sentiment distribution
    counts = Counter(r.get('sentiment') or 'unanalyzed' for r in rows)
    labels = ['positive', 'neutral', 'negative', 'unanalyzed']
    vals = [counts[x] for x in labels]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Reviews')
    plt.tight_layout()
    p1 = out / 'sentiment_distribution.png'
    plt.savefig(p1, dpi=150)
    plt.close()

    # 2) monthly sentiment trend
    monthly = defaultdict(Counter)
    for r in rows:
        d = r.get('review_date')
        s = r.get('sentiment')
        if d and s:
            monthly[str(d)[:7]][s] += 1
    months = sorted(monthly)
    plt.figure(figsize=(12, 5))
    for sentiment in ['positive', 'neutral', 'negative']:
        plt.plot(months, [monthly[m][sentiment] for m in months], label=sentiment)
    plt.title('Monthly Sentiment Trend')
    plt.xlabel('Month')
    plt.ylabel('Reviews')
    plt.xticks(rotation=60)
    plt.legend()
    plt.tight_layout()
    p2 = out / 'sentiment_trend.png'
    plt.savefig(p2, dpi=150)
    plt.close()

    # 3) rating vs sentiment matrix
    ratings = [1, 2, 3, 4, 5]
    sentiments = ['positive', 'neutral', 'negative']
    matrix = [[
        sum(
            1 for r in rows
            if r.get('rating') is not None
            and int(round(float(r['rating']))) == rating
            and r.get('sentiment') == sentiment
        )
        for sentiment in sentiments
    ] for rating in ratings]
    import numpy as np
    arr = np.array(matrix)
    plt.figure(figsize=(8, 5))
    plt.imshow(arr, aspect='auto')
    plt.title('Rating vs Sentiment Matrix')
    plt.xlabel('Sentiment')
    plt.ylabel('Rating')
    plt.xticks(range(3), sentiments)
    plt.yticks(range(5), ratings)
    for i in range(5):
        for j in range(3):
            plt.text(j, i, str(arr[i, j]), ha='center', va='center')
    plt.tight_layout()
    p3 = out / 'rating_sentiment_matrix.png'
    plt.savefig(p3, dpi=150)
    plt.close()

    return [str(p1), str(p2), str(p3)]


def _alert_config(config):
    alert_cfg = (config or {}).get('alert', {})
    return (
        int(alert_cfg.get('recent_days', 30)),
        float(alert_cfg.get('negative_ratio_threshold', 0.30)),
        float(alert_cfg.get('increase_threshold', 0.10)),
    )


def build_report(rows, latest_extract=None, top_n=5, config=None):
    st = stats(rows)
    neg = theme_counts(rows, 'negative').most_common(top_n)
    pos = theme_counts(rows, 'positive').most_common(top_n)
    products = Counter((r.get('product') or 'Unknown') for r in rows).most_common(top_n)
    providers = Counter(r.get('analysis_provider') or 'unanalyzed' for r in rows)
    languages = Counter(r.get('analysis_language') or 'unknown' for r in rows if r.get('sentiment'))

    lines = [
        '# 전자제품 고객 리뷰 감정 분석 대시보드',
        '',
        f"생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        '',
        '## 핵심 지표',
        f"- 총 리뷰 수: {st['total']:,}건",
        f"- 분석 완료율: {st['analysis_rate']:.1f}%",
        f"- 긍정 비율: {st['positive_ratio']:.1f}%",
        f"- 부정 비율: {st['negative_ratio']:.1f}%",
        f"- 평균 별점: {st['average_rating']:.2f}" if st['average_rating'] is not None else '- 평균 별점: N/A',
        f"- 평균 감정 신뢰도: {st['average_confidence']:.2f}" if st['average_confidence'] is not None else '- 평균 감정 신뢰도: N/A',
        f"- 별점-감정 일치율: {st['rating_sentiment_agreement']:.1f}%" if st['rating_sentiment_agreement'] is not None else '- 별점-감정 일치율: N/A',
        f"- 저별점(1~2점) 비율: {st['low_rating_ratio']:.1f}%",
        f"- 분석 방식: {dict(providers)}",
        f"- 분석 언어: {dict(languages)}",
        '',
        '## TOP 긍정 테마',
    ]
    lines += [f"{i}. {k}: {v}건" for i, (k, v) in enumerate(pos, 1)] or ['- 없음']
    lines += ['', '## TOP 부정 테마']
    lines += [f"{i}. {k}: {v}건" for i, (k, v) in enumerate(neg, 1)] or ['- 없음']
    lines += ['', f'## TOP {top_n} 리뷰 제품']
    lines += [f"{i}. {name}: {cnt}건" for i, (name, cnt) in enumerate(products, 1)]

    if latest_extract:
        lines += [
            '', '## AI/추출 인사이트',
            f"- 제공자: {latest_extract.get('provider')}",
            '- 긍정 키워드: ' + ', '.join(latest_extract.get('positive_keywords', [])),
            '- 부정 키워드: ' + ', '.join(latest_extract.get('negative_keywords', [])),
            '- 요약: ' + latest_extract.get('summary', ''),
            '- 개선 제안: ' + latest_extract.get('suggestions', ''),
        ]

    if config:
        days, ratio_threshold, increase_threshold = _alert_config(config)
        alert = recent_negative_alert(rows, days, ratio_threshold, increase_threshold)
    else:
        alert = None

    if alert:
        previous_ratio = alert.get('previous_negative_ratio')
        previous_ratio_text = 'N/A' if previous_ratio is None else f'{previous_ratio * 100:.1f}%'
        increase = alert.get('increase')
        increase_text = 'N/A' if increase is None else f'{increase * 100:+.1f}%p'
        lines += [
            '', '## 최근 부정 리뷰 급증 알림',
            f"- 최근 기간: {alert['start']} ~ {alert['end']} ({alert['count']}건)",
            f"- 이전 기간: {alert['previous_start']} ~ {alert['previous_end']} ({alert['previous_count']}건)",
            f"- 최근 부정 비율: {alert['negative_ratio'] * 100:.1f}%",
            f"- 이전 부정 비율: {previous_ratio_text}",
            f"- 증가폭: {increase_text}",
            f"- 경고 기준: 최근 비율 {alert['threshold'] * 100:.1f}% 이상 + 증가폭 {alert['increase_threshold'] * 100:.1f}%p 이상",
            f"- 경고: {'YES' if alert['warning'] else 'NO'}",
            f"- 판단 근거: {alert['reason']}",
            '', '### 원인 가설',
        ]
        lines += [f'- {x}' for x in alert.get('hypotheses', [])]
        lines += ['', '### 후속 분석 절차']
        lines += [f'- {x}' for x in alert.get('follow_up_actions', [])]

    return '\n'.join(lines)


def _image_data_uri(path):
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or 'image/png'
    encoded = base64.b64encode(p.read_bytes()).decode('ascii')
    return f'data:{mime};base64,{encoded}'


def _standalone_html(report, charts):
    # The report and all chart images are embedded. dashboard.html therefore
    # works by itself even if the PNG files are moved or deleted.
    escaped = html_lib.escape(report).replace('\n', '<br>')
    images = ''.join(
        f'<section class="chart"><h3>{html_lib.escape(Path(p).stem.replace("_", " ").title())}</h3>'
        f'<img src="{_image_data_uri(p)}" alt="{html_lib.escape(Path(p).name)}"></section>'
        for p in charts
    )
    return f'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>전자제품 고객 리뷰 대시보드</title>
<style>
body{{font-family:Arial,"Malgun Gothic",sans-serif;max-width:1100px;margin:0 auto;padding:28px;background:#f5f6f8;color:#202124}}
.card{{background:white;border-radius:14px;padding:24px;margin:18px 0;box-shadow:0 2px 10px rgba(0,0,0,.06)}}
h1{{margin-top:0}} .report{{line-height:1.7}} .chart img{{max-width:100%;height:auto;display:block;margin:auto}}
.chart{{background:white;border-radius:14px;padding:18px;margin:18px 0;box-shadow:0 2px 10px rgba(0,0,0,.06)}}
</style>
</head>
<body>
<div class="card"><h1>전자제품 고객 리뷰 대시보드</h1><div class="report">{escaped}</div></div>
{images}
</body>
</html>'''


def dashboard(storage, config, filters=None, out_dir=None, top_n=5, html=True):
    rows = storage.filtered(filters)
    out = Path(out_dir or config.get('visualization', {}).get('output_dir', 'output'))
    out.mkdir(parents=True, exist_ok=True)
    charts = make_charts(rows, out, config)
    report = build_report(rows, storage.latest_extract(filters), top_n, config)

    report_path = out / 'report.md'
    txt_path = out / 'report.txt'
    report_path.write_text(report, encoding='utf-8')
    txt_path.write_text(report.replace('#', ''), encoding='utf-8')

    html_path = None
    if html:
        html_path = out / 'dashboard.html'
        html_path.write_text(_standalone_html(report, charts), encoding='utf-8')

    return {
        'charts': charts,
        'report': str(report_path),
        'txt': str(txt_path),
        'html': str(html_path) if html_path else None,
        'text': report,
    }
