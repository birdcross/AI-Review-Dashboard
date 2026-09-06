import hashlib
import logging
import re
import unicodedata
from datetime import datetime
import pandas as pd
logger=logging.getLogger(__name__)

def normalize_text(text):
    text=unicodedata.normalize('NFKC',str(text))
    return re.sub(r'\s+',' ',text).strip()

def normalize_rating(v):
    if v is None or v=='': return None
    try: x=float(v)
    except: return None
    return x if 1<=x<=5 else None

def normalize_date(v):
    if v is None or v=='': return None
    x=pd.to_datetime(v,errors='coerce',utc=True)
    if pd.isna(x): return None
    return x.strftime('%Y-%m-%d')

def clean_all(storage,min_length=5,reset=False):
    existing=[] if reset else storage.get_clean()
    by_raw={int(r['raw_id']):r for r in existing}
    next_id=max((int(r.get('id',0)) for r in existing),default=0)+1
    now=datetime.now().isoformat(timespec='seconds')
    result={'inserted':0,'updated':0,'short':0,'invalid_rating':0,'invalid_date':0}
    out=[]
    seen=set()
    for raw in storage.get_raw():
        text=normalize_text(raw['review_text'])
        if len(text)<min_length:
            result['short']+=1; continue
        rating=normalize_rating(raw.get('rating'))
        if raw.get('rating') not in (None,'') and rating is None: result['invalid_rating']+=1
        date=normalize_date(raw.get('review_date'))
        if raw.get('review_date') not in (None,'') and date is None: result['invalid_date']+=1
        raw_id=int(raw['id']); old=by_raw.get(raw_id)
        item={
            'raw_id':raw_id,'source_row':raw.get('source_row'),'source_id':raw.get('source_id'),
            'review_text':text,'original_text':normalize_text(raw['original_text']) if raw.get('original_text') else None,
            'rating':rating,'review_date':date,
            'product':normalize_text(raw['product']) if raw.get('product') else None,
            'brand':normalize_text(raw['brand']) if raw.get('brand') else None,
            'review_hash':hashlib.sha256(text.lower().encode('utf-8')).hexdigest(),
        }
        if old:
            item={**old,**item,'updated_at':now}; result['updated']+=1
        else:
            item={**item,'id':next_id,'sentiment':None,'confidence':None,'analyzed_at':None,'analysis_provider':None,'created_at':now,'updated_at':now}; next_id+=1; result['inserted']+=1
        out.append(item); seen.add(raw_id)
    storage.write_jsonl(storage.clean_path,out)
    logger.info('정제 완료: %s',result)
    return result
