import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from app.config import load_config, setup_logging
from app.env import get_api_key
from app.storage import JsonlStorage
from app.importer import import_file
from app.cleaner import clean_all
from app.services import analyze, extract
from app.analytics import stats, theme_counts, recent_negative_alert
from app.dashboard import dashboard
from app.exporter import export
from app.shop import (
    product_catalog,
    product_summary,
    product_rows,
    keyword_review_search,
    print_review_line,
    short_name,
    safe_folder_name,
    stars,
)

ALIASES = {
    '긍정': 'positive', '부정': 'negative', '중립': 'neutral',
    'positive': 'positive', 'negative': 'negative', 'neutral': 'neutral'
}


def sent(v):
    return ALIASES.get(v, v) if v else None


def common_filters(p):
    p.add_argument('--sentiment')
    p.add_argument('--rating', type=float)
    p.add_argument('--date-from')
    p.add_argument('--date-to')
    p.add_argument('--product')
    p.add_argument('--brand')


def filters(args):
    keys = ['sentiment', 'rating', 'rating_min', 'date_from', 'date_to', 'product', 'brand']
    d = {k: getattr(args, k, None) for k in keys if getattr(args, k, None) is not None}
    if d.get('sentiment'):
        d['sentiment'] = sent(d['sentiment'])
    return d


def parser():
    p = argparse.ArgumentParser(description='전자제품 고객 리뷰 AI 분석 CLI')
    p.add_argument('--config', default='config.json')
    sp = p.add_subparsers(dest='command')

    x = sp.add_parser('import')
    x.add_argument('--file', required=True)
    x.add_argument('--duplicate-policy', choices=['skip', 'upsert'])
    x.add_argument('--reset', action='store_true')
    x.add_argument('--limit', type=int)

    x = sp.add_parser('clean')
    x.add_argument('--min-length', type=int)
    x.add_argument('--reset', action='store_true')

    x = sp.add_parser('analyze')
    g = x.add_mutually_exclusive_group()
    g.add_argument('--all', action='store_true')
    g.add_argument('--id', type=int)
    g.add_argument('--unanalyzed', action='store_true')
    x.add_argument('--limit', type=int)
    x.add_argument('--force', action='store_true')
    x.add_argument('--offline', action='store_true', help='API 없이 검증용 베이스라인 분석')

    x = sp.add_parser('extract')
    common_filters(x)
    x.add_argument('--limit', type=int)
    x.add_argument('--offline', action='store_true')

    x = sp.add_parser('list')
    common_filters(x)
    x.add_argument('--page', type=int, default=1)
    x.add_argument('--size', type=int, default=10)
    x.add_argument('--sort', default='id', choices=['id', 'review_date', 'rating', 'confidence', 'product', 'brand', 'sentiment'])
    x.add_argument('--desc', action='store_true')

    x = sp.add_parser('show')
    x.add_argument('id', type=int)

    x = sp.add_parser('stats')
    common_filters(x)

    x = sp.add_parser('dashboard')
    common_filters(x)
    x.add_argument('--output-dir')
    x.add_argument('--top-n', type=int, default=5)
    x.add_argument('--html', action='store_true')

    x = sp.add_parser('export')
    x.add_argument('--format', choices=['csv', 'jsonl', 'xlsx'], required=True)
    x.add_argument('--output')
    x.add_argument('--sentiment')
    x.add_argument('--rating-min', type=float)
    x.add_argument('--date-from')
    x.add_argument('--date-to')
    x.add_argument('--product')
    x.add_argument('--brand')
    return p


def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    elif os.environ.get('TERM'):
        os.system('clear')


def pause(msg='Enter 키를 누르면 계속합니다...'):
    input(f'\n{msg}')


def ask(prompt, default=None):
    suffix = f' [{default}]' if default is not None else ''
    value = input(f'{prompt}{suffix}: ').strip()
    return value if value else default


def ask_int(prompt, default=None, minimum=None):
    while True:
        value = ask(prompt, default)
        if value in (None, ''):
            return None
        try:
            n = int(value)
            if minimum is not None and n < minimum:
                print(f'{minimum} 이상의 숫자를 입력해주세요.')
                continue
            return n
        except ValueError:
            print('숫자로 입력해주세요.')


def yes_no(prompt, default=False):
    d = 'Y' if default else 'N'
    value = input(f'{prompt} (Y/N) [{d}]: ').strip().lower()
    if not value:
        return default
    return value in ('y', 'yes', '예', 'ㅇ')


def print_stats(st, title='리뷰 분석 통계'):
    print(f'\n=== {title} ===')
    print(f"총 리뷰 수     : {st['total']:,}건")
    print(f"분석 완료      : {st['analyzed']:,}건 ({st['analysis_rate']:.1f}%)")
    print(f"긍정           : {st['positive']:,}건 ({st['positive_ratio']:.1f}%)")
    print(f"중립           : {st['neutral']:,}건 ({st['neutral_ratio']:.1f}%)")
    print(f"부정           : {st['negative']:,}건 ({st['negative_ratio']:.1f}%)")
    print(f"평균 별점      : {st['average_rating']:.2f}" if st['average_rating'] is not None else '평균 별점      : N/A')
    print(f"평균 신뢰도    : {st['average_confidence']:.2f}" if st['average_confidence'] is not None else '평균 신뢰도    : N/A')
    print(f"별점-감정 일치 : {st['rating_sentiment_agreement']:.1f}%" if st['rating_sentiment_agreement'] is not None else '별점-감정 일치 : N/A')


def has_api_key(cfg):
    # .env까지 로드해서 실제 설정된 API 키가 있는지 확인한다.
    return bool(get_api_key(cfg))


def ensure_data(storage, cfg):
    if storage.get_clean():
        return
    default_file = Path('data/DatafinitiElectronicsProductData_KO_7299.csv')
    if not default_file.exists():
        return
    print('처음 실행입니다. 상품 리뷰 데이터를 준비하고 있습니다...')
    import_file(storage, default_file, cfg['columns'], cfg.get('duplicate_policy', 'skip'), True, None)
    clean_all(storage, cfg.get('clean', {}).get('min_length', 5), True)
    print('데이터 준비가 완료되었습니다.\n')


def main_header(storage):
    rows = storage.get_clean()
    products = product_catalog(rows)
    print('=' * 78)
    print('                전자제품 고객 리뷰 AI 쇼핑몰 CLI')
    print('=' * 78)
    print(f'  등록 제품 {len(products):>3}종   |   리뷰 {len(rows):>6,}건')
    print('-' * 78)
    print('  1. 제품 둘러보기')
    print('  2. 제품 검색')
    print('  3. 전체 리뷰 AI 대시보드')
    print('  4. 리뷰 검색 / 상세 조회')
    print('  5. 분석 결과 내보내기')
    print('  0. 종료')
    print('=' * 78)


def show_catalog(storage, cfg, query=None):
    items = product_catalog(storage.get_clean(), query)
    if not items:
        print('검색 조건에 맞는 제품이 없습니다.')
        pause()
        return
    page = 1
    size = 10
    while True:
        clear_screen()
        pages = max(1, math.ceil(len(items) / size))
        page = min(max(page, 1), pages)
        start = (page - 1) * size
        visible = items[start:start + size]
        title = f'제품 검색: {query}' if query else '전자제품 목록'
        print('=' * 100)
        print(f' {title}   ({page}/{pages} 페이지, 총 {len(items)}종)')
        print('=' * 100)
        print(f"{'번호':<5}{'브랜드':<14}{'제품명':<52}{'리뷰':>8}{'평균별점':>11}")
        print('-' * 100)
        for i, item in enumerate(visible, 1):
            avg = f"{item['average_rating']:.2f}" if item['average_rating'] is not None else '-'
            print(f"{i:<5}{short_name(item['brand'],12):<14}{short_name(item['product'],48):<52}{item['count']:>7,}건{avg:>10}")
        print('-' * 100)
        print('번호: 제품 선택   N: 다음   P: 이전   0: 뒤로')
        choice = input('선택 > ').strip().lower()
        if choice == '0':
            return
        if choice == 'n' and page < pages:
            page += 1
            continue
        if choice == 'p' and page > 1:
            page -= 1
            continue
        try:
            n = int(choice)
            if 1 <= n <= len(visible):
                product_detail(storage, cfg, visible[n - 1]['product'])
            else:
                print('화면에 표시된 제품 번호를 입력해주세요.')
                pause()
        except ValueError:
            print('번호, N, P, 0 중 하나를 입력해주세요.')
            pause()


def show_reviews(rows, title, size=10):
    if not rows:
        print('표시할 리뷰가 없습니다.')
        pause()
        return
    page = 1
    while True:
        clear_screen()
        pages = max(1, math.ceil(len(rows) / size))
        page = min(max(page, 1), pages)
        start = (page - 1) * size
        visible = rows[start:start + size]
        print('=' * 100)
        print(f' {title}  ({page}/{pages}, 총 {len(rows):,}건)')
        print('=' * 100)
        for i, r in enumerate(visible, start + 1):
            print_review_line(r, i)
            print('-' * 100)
        print('N: 다음   P: 이전   ID 숫자: 상세보기   0: 뒤로')
        choice = input('선택 > ').strip().lower()
        if choice == '0':
            return
        if choice == 'n' and page < pages:
            page += 1
            continue
        if choice == 'p' and page > 1:
            page -= 1
            continue
        try:
            review_id = int(choice)
            r = next((x for x in rows if int(x.get('id', -1)) == review_id), None)
            if r:
                show_review_detail(r)
            else:
                print('해당 ID의 리뷰가 없습니다.')
                pause()
        except ValueError:
            print('N, P, 리뷰 ID, 0 중 하나를 입력해주세요.')
            pause()


def show_review_detail(r):
    clear_screen()
    print('=' * 90)
    print(' 리뷰 상세')
    print('=' * 90)
    print(f"ID       : {r.get('id')}")
    print(f"제품     : {r.get('product')}")
    print(f"브랜드   : {r.get('brand')}")
    print(f"작성일   : {r.get('review_date') or '-'}")
    print(f"별점     : {stars(r.get('rating'))} {r.get('rating') if r.get('rating') is not None else '-'}")
    label = {'positive': '긍정', 'neutral': '중립', 'negative': '부정'}.get(r.get('sentiment'), '미분석')
    print(f"AI 감정  : {label}")
    print(f"신뢰도   : {r.get('confidence') if r.get('confidence') is not None else '-'}")
    print(f"분석방식 : {r.get('analysis_provider') or '-'}")
    print('-' * 90)
    print('[한글 리뷰]')
    print(r.get('review_text') or '')
    if r.get('original_text'):
        print('\n[영문 원문]')
        print(r.get('original_text'))
    pause('Enter 키를 누르면 돌아갑니다...')


def print_insights(rows, item=None):
    st = stats(rows)
    print_stats(st, 'AI 리뷰 분석 결과')
    print('\n[TOP 긍정 테마]')
    pos = theme_counts(rows, 'positive').most_common(5)
    for i, (name, count) in enumerate(pos, 1):
        print(f' {i}. {name:<14} {count:>5,}건')
    if not pos:
        print(' - 아직 분석된 긍정 리뷰가 없습니다.')
    print('\n[TOP 부정 테마]')
    neg = theme_counts(rows, 'negative').most_common(5)
    for i, (name, count) in enumerate(neg, 1):
        print(f' {i}. {name:<14} {count:>5,}건')
    if not neg:
        print(' - 아직 분석된 부정 리뷰가 없습니다.')
    if item:
        print('\n[AI 키워드 / 요약]')
        print(' 긍정 키워드 :', ', '.join(item.get('positive_keywords', [])) or '-')
        print(' 부정 키워드 :', ', '.join(item.get('negative_keywords', [])) or '-')
        print(' 요약          :', item.get('summary', '') or '-')
        print(' 개선 제안     :', item.get('suggestions', '') or '-')


def analyze_product(storage, cfg, product):
    clear_screen()
    rows = product_rows(storage, product)
    if not rows:
        print('제품 리뷰가 없습니다.')
        pause()
        return

    print('=' * 90)
    print(' Gemini AI 제품 리뷰 분석')
    print('=' * 90)
    print('제품:', product)
    print(f'리뷰: {len(rows):,}건')
    providers = Counter(r.get('analysis_provider') or '미분석' for r in rows)
    print('현재 분석 상태:', dict(providers))

    if not has_api_key(cfg):
        env_name = cfg.get('ai', {}).get('api_key_env', 'GEMINI_API_KEY')
        print(f'\n[ERROR] API 키를 찾지 못했습니다.')
        print(f'프로젝트 루트의 .env 파일에 {env_name}=본인_API_KEY 를 입력하세요.')
        print('오프라인 분석으로 대체하지 않고 작업을 중단합니다.')
        pause()
        return

    print('\nGemini API 키가 확인되었습니다.')
    limit = ask_int('Gemini API로 분석할 리뷰 수 (0=전체)', 10, 0)
    limit = None if limit == 0 else limit

    print('\n[1단계] Gemini 감정 분석 중...')
    result = analyze(
        storage, cfg,
        mode='all',
        review_id=None,
        force=True,
        limit=limit,
        offline=False,
        filters={'product': product},
    )
    print('[감정분석 완료]', result)

    if result.get('success', 0) <= 0:
        print('\n감정 분석에 성공한 리뷰가 없어 키워드/요약 분석을 실행하지 않습니다.')
        print('logs/app.log에서 Gemini API 오류 내용을 확인하세요.')
        pause()
        return

    print('\n[2단계] Gemini 키워드 / 요약 / 개선 제안 분석 중...')
    try:
        item, count = extract(
            storage, cfg,
            {'product': product},
            cfg.get('ai', {}).get('extract_limit', 120),
            offline=False,
        )
    except Exception as exc:
        print(f'[ERROR] 키워드/요약 분석 실패: {exc}')
        pause()
        return

    print(f'[인사이트 완료] 분석 리뷰 {count:,}건 참고')
    print(f"저장 provider: {item.get('provider')}")
    print('저장 위치: data/extracts.jsonl')

    rows = product_rows(storage, product)
    print_insights(rows, item)
    pause()


def product_detail(storage, cfg, product):
    while True:
        clear_screen()
        summary = product_summary(storage, product)
        if not summary:
            print('제품 정보를 찾을 수 없습니다.')
            pause()
            return
        st = summary['stats']
        print('=' * 100)
        print(f" {summary['product']}")
        print('=' * 100)
        print(f"브랜드     : {summary['brand']}")
        print(f"평균 별점  : {stars(st['average_rating'])} {st['average_rating']:.2f}" if st['average_rating'] is not None else '평균 별점  : N/A')
        print(f"리뷰 수    : {st['total']:,}건")
        if st['analyzed']:
            print(f"AI 감정    : 긍정 {st['positive_ratio']:.1f}% / 중립 {st['neutral_ratio']:.1f}% / 부정 {st['negative_ratio']:.1f}%")
        else:
            print('AI 감정    : 아직 분석하지 않음')
        print('\n[최근 리뷰]')
        for r in summary['recent'][:3]:
            print_review_line(r)
            print('-' * 100)
        print('  1. 전체 리뷰 보기')
        print('  2. Gemini AI 리뷰 분석 / 고객 인사이트')
        print('  3. 부정 리뷰만 보기')
        print('  4. 긍정 리뷰만 보기')
        print('  5. 제품 대시보드 생성')
        print('  6. 제품 분석 결과 Excel 내보내기')
        print('  0. 제품 목록으로')
        print('=' * 100)
        choice = input('선택 > ').strip()
        if choice == '0':
            return
        if choice == '1':
            rows = sorted(summary['rows'], key=lambda r: (r.get('review_date') or '', int(r.get('id', 0))), reverse=True)
            show_reviews(rows, short_name(product, 70) + ' - 전체 리뷰')
        elif choice == '2':
            analyze_product(storage, cfg, product)
        elif choice == '3':
            rows = sorted(product_rows(storage, product, 'negative'), key=lambda r: (r.get('review_date') or '', int(r.get('id', 0))), reverse=True)
            show_reviews(rows, short_name(product, 70) + ' - 부정 리뷰')
        elif choice == '4':
            rows = sorted(product_rows(storage, product, 'positive'), key=lambda r: (r.get('review_date') or '', int(r.get('id', 0))), reverse=True)
            show_reviews(rows, short_name(product, 70) + ' - 긍정 리뷰')
        elif choice == '5':
            folder = Path(cfg['visualization']['output_dir']) / 'products' / safe_folder_name(product)
            result = dashboard(storage, cfg, {'product': product}, str(folder), 5, True)
            print('\n제품 대시보드가 생성되었습니다.')
            print('HTML  :', result.get('html'))
            print('Report:', result.get('report'))
            for chart in result.get('charts', []):
                print('Chart :', chart)
            pause()
        elif choice == '6':
            folder = Path(cfg['visualization']['output_dir']) / 'products' / safe_folder_name(product)
            out = folder / 'reviews_analysis.xlsx'
            count, path = export(storage, 'xlsx', str(out), {'product': product})
            print(f'\n{count:,}건을 저장했습니다: {path}')
            pause()
        else:
            print('0~6 중 하나를 선택해주세요.')
            pause()


def review_search_menu(storage):
    clear_screen()
    print('=' * 80)
    print(' 리뷰 검색 / 상세 조회')
    print('=' * 80)
    direct = ask('리뷰 ID를 알고 있으면 입력 (검색하려면 Enter)')
    if direct:
        try:
            r = storage.get_review(int(direct))
        except ValueError:
            r = None
        if r:
            show_review_detail(r)
        else:
            print('해당 리뷰 ID가 없습니다.')
            pause()
        return
    keyword = ask('검색어 (리뷰/제품/브랜드)', '') or ''
    sentiment = sent(ask('감정 필터: 긍정/중립/부정 (전체=Enter)', '') or '')
    rating_text = ask('별점 필터 1~5 (전체=Enter)', '') or ''
    try:
        rating = float(rating_text) if rating_text else None
    except ValueError:
        rating = None
    rows = keyword_review_search(storage, keyword, sentiment, rating)
    show_reviews(rows, f'검색 결과: {keyword or "전체"}')


def overall_dashboard_menu(storage, cfg):
    clear_screen()
    rows = storage.get_clean()
    print_stats(stats(rows), '전체 전자제품 리뷰')
    print('\n전체 대시보드를 생성합니다...')
    result = dashboard(storage, cfg, {}, cfg['visualization']['output_dir'], 5, True)
    print('\n[생성 완료]')
    print('HTML  :', result.get('html'))
    print('Report:', result.get('report'))
    for x in result.get('charts', []):
        print('Chart :', x)
    alert = recent_negative_alert(rows, cfg['alert']['recent_days'], cfg['alert']['negative_ratio_threshold'])
    if alert and alert['warning']:
        print(f"\n[경고] 최근 {cfg['alert']['recent_days']}일 부정 리뷰 비율 {alert['negative_ratio']*100:.1f}%")
    pause()


def export_menu(storage, cfg):
    clear_screen()
    print('=' * 80)
    print(' 분석 결과 내보내기')
    print('=' * 80)
    print(' 1. 전체 리뷰')
    print(' 2. 특정 제품')
    print(' 0. 뒤로')
    choice = input('선택 > ').strip()
    if choice == '0':
        return
    f = {}
    name = 'all_reviews'
    if choice == '2':
        q = ask('제품명 검색어')
        matches = product_catalog(storage.get_clean(), q)
        if not matches:
            print('제품을 찾지 못했습니다.')
            pause()
            return
        for i, item in enumerate(matches[:10], 1):
            print(f" {i}. {short_name(item['product'], 65)} ({item['count']:,}건)")
        n = ask_int('제품 번호', 1, 1)
        if not n or n > min(10, len(matches)):
            return
        product = matches[n - 1]['product']
        f['product'] = product
        name = safe_folder_name(product)
    fmt = ask('저장 형식 csv/jsonl/xlsx', 'xlsx')
    if fmt not in ('csv', 'jsonl', 'xlsx'):
        print('지원하지 않는 형식입니다.')
        pause()
        return
    out = Path(cfg['visualization']['output_dir']) / 'exports' / f'{name}.{fmt}'
    count, path = export(storage, fmt, str(out), f)
    print(f'\n{count:,}건 저장 완료: {path}')
    pause()


def interactive_shop(storage, cfg):
    ensure_data(storage, cfg)
    if not storage.get_clean():
        print('분석할 데이터가 없습니다. CLI import/clean 명령으로 먼저 데이터를 준비해주세요.')
        return
    while True:
        clear_screen()
        main_header(storage)
        choice = input('선택 > ').strip()
        try:
            if choice == '0':
                print('프로그램을 종료합니다.')
                return
            elif choice == '1':
                show_catalog(storage, cfg)
            elif choice == '2':
                query = ask('제품명 또는 브랜드 검색어')
                if query:
                    show_catalog(storage, cfg, query)
            elif choice == '3':
                overall_dashboard_menu(storage, cfg)
            elif choice == '4':
                review_search_menu(storage)
            elif choice == '5':
                export_menu(storage, cfg)
            else:
                print('0~5 사이 번호를 선택해주세요.')
                pause()
        except KeyboardInterrupt:
            print('\n작업이 취소되었습니다.')
            pause()
        except Exception as e:
            print(f'\n[ERROR] {type(e).__name__}: {e}')
            pause()


def run_cli(args, cfg, storage):
    if args.command == 'import':
        print(import_file(storage, args.file, cfg['columns'], args.duplicate_policy or cfg['duplicate_policy'], args.reset, args.limit))
    elif args.command == 'clean':
        print(clean_all(storage, args.min_length or cfg['clean']['min_length'], args.reset))
    elif args.command == 'analyze':
        mode = 'id' if args.id is not None else 'all' if args.all else 'unanalyzed'
        print(analyze(storage, cfg, mode, args.id, args.force, args.limit, args.offline))
    elif args.command == 'extract':
        item, n = extract(storage, cfg, filters(args), args.limit or cfg['ai']['extract_limit'], args.offline)
        print(f'추출 대상: {n}건')
        print(item)
    elif args.command == 'list':
        rows, total = storage.list_reviews(filters(args), args.page, args.size, args.sort, args.desc)
        pages = max(1, math.ceil(total / args.size))
        print(f'=== 리뷰 목록 ({args.page}/{pages} 페이지, 총 {total:,}건) ===')
        for r in rows:
            print_review_line(r)
    elif args.command == 'show':
        r = storage.get_review(args.id)
        print(pd.Series(r).to_string() if r else '해당 리뷰 없음')
    elif args.command == 'stats':
        rows = storage.filtered(filters(args))
        print_stats(stats(rows))
        alert = recent_negative_alert(rows, cfg['alert']['recent_days'], cfg['alert']['negative_ratio_threshold'])
        if alert and alert['warning']:
            print(f"\n[WARNING] 최근 {cfg['alert']['recent_days']}일 부정 리뷰 비율 {alert['negative_ratio']*100:.1f}%")
    elif args.command == 'dashboard':
        result = dashboard(storage, cfg, filters(args), args.output_dir, args.top_n, args.html)
        print(result['text'])
        print('\n생성 파일:')
        for x in result['charts'] + [result['report'], result['txt']] + ([result['html']] if result['html'] else []):
            print('-', x)
    elif args.command == 'export':
        ext = {'csv': 'csv', 'jsonl': 'jsonl', 'xlsx': 'xlsx'}[args.format]
        out = args.output or f'output/reviews_export.{ext}'
        print(export(storage, args.format, out, filters(args)))


def main():
    p = parser()
    args = p.parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)
    s = cfg['storage']
    storage = JsonlStorage(s['raw_path'], s['clean_path'], s['extracts_path'])
    if args.command is None:
        interactive_shop(storage, cfg)
    else:
        run_cli(args, cfg, storage)


if __name__ == '__main__':
    main()
