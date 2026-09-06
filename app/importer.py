import hashlib
import logging
from pathlib import Path
import pandas as pd

logger=logging.getLogger(__name__)

def safe_value(v):
    if pd.isna(v): return None
    if hasattr(v, 'item'):
        try: return v.item()
        except: pass
    return v

def import_file(storage, file_path, columns, duplicate_policy='skip', reset=False, limit=None):
    p=Path(file_path)
    if not p.exists(): raise FileNotFoundError(p)
    if p.suffix.lower()=='.csv': df=pd.read_csv(p, low_memory=False)
    elif p.suffix.lower() in {'.xlsx','.xls'}: df=pd.read_excel(p)
    else: raise ValueError('지원 형식: CSV, Excel')
    if limit: df=df.head(limit)
    text_col=columns['text']
    if text_col not in df.columns: raise ValueError(f'필수 리뷰 컬럼 없음: {text_col}')

    rows=[] if reset else storage.get_raw()
    by_key={r.get('duplicate_key'):i for i,r in enumerate(rows)}
    next_id=max((int(r.get('id',0)) for r in rows),default=0)+1
    from datetime import datetime
    now=datetime.now().isoformat(timespec='seconds')
    c={'total':len(df),'inserted':0,'updated':0,'skipped':0,'invalid':0}

    for idx, row in df.iterrows():
        text=safe_value(row.get(text_col))
        if text is None or not str(text).strip():
            c['invalid']+=1; continue
        item={
            'source_row':int(idx)+2,
            'source_id':safe_value(row.get(columns.get('source_id'))),
            'review_text':str(text),
            'original_text':safe_value(row.get(columns.get('original_text'))),
            'rating':safe_value(row.get(columns.get('rating'))),
            'review_date':safe_value(row.get(columns.get('date'))),
            'product':safe_value(row.get(columns.get('product'))),
            'brand':safe_value(row.get(columns.get('brand'))),
        }
        key_src=f"{item['review_text'].strip().lower()}|{item.get('product')}|{item.get('review_date')}"
        key=hashlib.sha256(key_src.encode('utf-8')).hexdigest()
        if key in by_key:
            if duplicate_policy=='skip': c['skipped']+=1; continue
            i=by_key[key]; old_id=rows[i]['id']; rows[i]={**item,'id':old_id,'duplicate_key':key,'imported_at':now}; c['updated']+=1
        else:
            new={**item,'id':next_id,'duplicate_key':key,'imported_at':now}; next_id+=1
            by_key[key]=len(rows); rows.append(new); c['inserted']+=1

    storage.write_jsonl(storage.raw_path, rows)
    logger.info('파일 로드: %s',p)
    logger.info('총 %s건 감지, 삽입 %s, 업데이트 %s, 스킵 %s, 무효 %s', c['total'],c['inserted'],c['updated'],c['skipped'],c['invalid'])
    return c
