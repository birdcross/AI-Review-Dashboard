import json
from pathlib import Path
import pandas as pd

def export(storage,fmt,output,filters=None):
    rows=storage.filtered(filters)
    if not rows: raise ValueError('내보낼 데이터가 없습니다.')
    cols=['id','source_row','source_id','product','brand','rating','review_date','review_text','original_text','sentiment','confidence','analysis_provider','analyzed_at']
    data=[{k:r.get(k) for k in cols} for r in rows]
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True); fmt=fmt.lower()
    if fmt=='csv': pd.DataFrame(data).to_csv(p,index=False,encoding='utf-8-sig')
    elif fmt=='jsonl':
        with p.open('w',encoding='utf-8') as f:
            for r in data:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    elif fmt in {'xlsx','excel'}: pd.DataFrame(data).to_excel(p,index=False)
    else: raise ValueError('지원 포맷: csv, jsonl, xlsx')
    return len(data),str(p)
