from collections import Counter
from pathlib import Path
import re

from .analytics import stats


def stars(rating):
    if rating is None:
        return '☆☆☆☆☆'
    try:
        n = max(1, min(5, int(round(float(rating)))))
    except Exception:
        return '☆☆☆☆☆'
    return '★' * n + '☆' * (5 - n)


def short_text(text, width=58):
    text = (text or '').replace('\n', ' ').strip()
    return text if len(text) <= width else text[: width - 3] + '...'


def short_name(name, width=48):
    name = (name or 'Unknown').strip()
    return name if len(name) <= width else name[: width - 3] + '...'


def safe_folder_name(name):
    value = re.sub(r'[\\/:*?"<>|]+', '_', name or 'product')
    value = re.sub(r'\s+', '_', value).strip('._')
    return value[:80] or 'product'


def product_catalog(rows, query=None):
    grouped = {}
    q = (query or '').strip().lower()
    for r in rows:
        product = r.get('product') or 'Unknown'
        brand = r.get('brand') or ''
        if q and q not in product.lower() and q not in brand.lower():
            continue
        g = grouped.setdefault(product, {'product': product, 'brand': brand, 'rows': []})
        g['rows'].append(r)
        if not g['brand'] and brand:
            g['brand'] = brand

    result = []
    for product, g in grouped.items():
        st = stats(g['rows'])
        result.append({
            'product': product,
            'brand': g['brand'],
            'count': st['total'],
            'average_rating': st['average_rating'],
            'positive_ratio': st['positive_ratio'],
            'negative_ratio': st['negative_ratio'],
            'analyzed': st['analyzed'],
        })
    result.sort(key=lambda x: (-x['count'], x['product'].lower()))
    return result


def product_rows(storage, product, sentiment=None):
    f = {'product': product}
    if sentiment:
        f['sentiment'] = sentiment
    return storage.filtered(f)


def product_summary(storage, product):
    rows = product_rows(storage, product)
    if not rows:
        return None
    st = stats(rows)
    brands = Counter(r.get('brand') or 'Unknown' for r in rows)
    recent = sorted(rows, key=lambda r: (r.get('review_date') or '', int(r.get('id', 0))), reverse=True)[:5]
    return {
        'product': product,
        'brand': brands.most_common(1)[0][0] if brands else 'Unknown',
        'rows': rows,
        'stats': st,
        'recent': recent,
    }


def keyword_review_search(storage, keyword=None, sentiment=None, rating=None, product=None):
    filters = {}
    if sentiment:
        filters['sentiment'] = sentiment
    if rating is not None:
        filters['rating'] = rating
    if product:
        filters['product'] = product
    rows = storage.filtered(filters)
    q = (keyword or '').strip().lower()
    if q:
        rows = [
            r for r in rows
            if q in (r.get('review_text') or '').lower()
            or q in (r.get('original_text') or '').lower()
            or q in (r.get('product') or '').lower()
            or q in (r.get('brand') or '').lower()
        ]
    rows.sort(key=lambda r: (r.get('review_date') or '', int(r.get('id', 0))), reverse=True)
    return rows


def print_review_line(r, index=None):
    prefix = f'{index:>2}. ' if index is not None else ''
    rating = r.get('rating')
    rating_text = f'{float(rating):.1f}' if rating is not None else '-'
    sentiment = r.get('sentiment') or '미분석'
    label = {'positive': '긍정', 'neutral': '중립', 'negative': '부정'}.get(sentiment, sentiment)
    print(f"{prefix}[ID {r.get('id')}] {stars(rating)} {rating_text} | {r.get('review_date') or '-'} | {label}")
    print(f"    {short_text(r.get('review_text'), 88)}")


def format_percent(value):
    return f'{value:.1f}%'
