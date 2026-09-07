import json
from datetime import datetime
from pathlib import Path

class JsonlStorage:
    def __init__(self, raw_path, clean_path, extracts_path):
        self.raw_path = Path(raw_path)
        self.clean_path = Path(clean_path)
        self.extracts_path = Path(extracts_path)
        for p in (self.raw_path, self.clean_path, self.extracts_path):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch(exist_ok=True)

    @staticmethod
    def read_jsonl(path):
        rows=[]
        p=Path(path)
        if not p.exists(): return rows
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    @staticmethod
    def write_jsonl(path, rows):
        with Path(path).open('w', encoding='utf-8') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')

    @staticmethod
    def append_jsonl(path, row):
        with Path(path).open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')

    @staticmethod
    def next_id(rows):
        return max((int(r.get('id',0)) for r in rows), default=0)+1

    def clear_raw(self): self.raw_path.write_text('', encoding='utf-8')
    def clear_clean(self): self.clean_path.write_text('', encoding='utf-8')

    def import_raw(self, item, duplicate_key, policy='skip'):
        rows=self.read_jsonl(self.raw_path)
        now=datetime.now().isoformat(timespec='seconds')
        idx=next((i for i,r in enumerate(rows) if r.get('duplicate_key')==duplicate_key), None)
        if idx is not None:
            if policy=='skip': return rows[idx]['id'], 'skipped'
            old_id=rows[idx]['id']
            rows[idx]={**item, 'id':old_id, 'duplicate_key':duplicate_key, 'imported_at':now}
            self.write_jsonl(self.raw_path, rows)
            return old_id, 'updated'
        new={**item, 'id':self.next_id(rows), 'duplicate_key':duplicate_key, 'imported_at':now}
        self.append_jsonl(self.raw_path, new)
        return new['id'], 'inserted'

    def get_raw(self): return self.read_jsonl(self.raw_path)
    def get_clean(self): return self.read_jsonl(self.clean_path)

    def upsert_clean(self, item):
        rows=self.read_jsonl(self.clean_path)
        now=datetime.now().isoformat(timespec='seconds')
        idx=next((i for i,r in enumerate(rows) if r.get('raw_id')==item['raw_id']), None)
        if idx is None:
            new={**item, 'id':self.next_id(rows), 'created_at':now, 'updated_at':now}
            self.append_jsonl(self.clean_path, new)
            return 'inserted'
        preserved={k:rows[idx].get(k) for k in ['sentiment','confidence','analyzed_at','analysis_provider','analysis_language','prompt_version']}
        rows[idx]={**rows[idx], **item, **preserved, 'updated_at':now}
        self.write_jsonl(self.clean_path, rows)
        return 'updated'

    def save_analysis_many(self, updates):
        by_id={int(x['id']):x for x in updates}
        rows=self.read_jsonl(self.clean_path)
        now=datetime.now().isoformat(timespec='seconds')
        for r in rows:
            u=by_id.get(int(r['id']))
            if u:
                r['sentiment']=u['sentiment']
                r['confidence']=float(u['confidence'])
                r['analysis_provider']=u.get('analysis_provider','api')
                if u.get('analysis_language'):
                    r['analysis_language']=u['analysis_language']
                if u.get('prompt_version'):
                    r['prompt_version']=u['prompt_version']
                r['analyzed_at']=now
                r['updated_at']=now
        self.write_jsonl(self.clean_path, rows)

    @staticmethod
    def matches(row, filters):
        filters=filters or {}
        if filters.get('sentiment') and row.get('sentiment') != filters['sentiment']: return False
        if filters.get('rating') is not None:
            try:
                if float(row.get('rating')) != float(filters['rating']): return False
            except: return False
        if filters.get('rating_min') is not None:
            try:
                if float(row.get('rating')) < float(filters['rating_min']): return False
            except: return False
        if filters.get('date_from'):
            if not row.get('review_date') or row['review_date'] < filters['date_from']: return False
        if filters.get('date_to'):
            if not row.get('review_date') or row['review_date'] > filters['date_to']: return False
        if filters.get('product') and row.get('product') != filters['product']: return False
        if filters.get('brand') and row.get('brand') != filters['brand']: return False
        return True

    def filtered(self, filters=None):
        return [r for r in self.get_clean() if self.matches(r, filters)]

    def list_reviews(self, filters=None, page=1, size=20, sort='id', desc=False):
        rows=self.filtered(filters)
        def key(r):
            v=r.get(sort)
            return (v is None, v if v is not None else '')
        try: rows.sort(key=key, reverse=desc)
        except TypeError: rows.sort(key=lambda r:str(r.get(sort) or ''), reverse=desc)
        total=len(rows)
        start=max(0,page-1)*size
        return rows[start:start+size], total

    def get_review(self, review_id):
        return next((r for r in self.get_clean() if int(r.get('id',-1))==int(review_id)), None)

    def analysis_targets(self, mode='unanalyzed', review_id=None, force=False, limit=None):
        rows=self.get_clean()
        if mode=='id': rows=[r for r in rows if int(r['id'])==int(review_id)]
        elif not force: rows=[r for r in rows if not r.get('sentiment')]
        if limit: rows=rows[:limit]
        return rows

    def save_extract(self, filters, result, provider):
        rows=self.read_jsonl(self.extracts_path)
        item={
            'id':self.next_id(rows), 'filters':filters, 'provider':provider,
            'positive_keywords':result.get('positive_keywords',[]),
            'negative_keywords':result.get('negative_keywords',[]),
            'summary':result.get('summary',''), 'suggestions':result.get('suggestions',''),
            'created_at':datetime.now().isoformat(timespec='seconds')
        }
        self.append_jsonl(self.extracts_path, item)
        return item

    def latest_extract(self, filters=None):
        rows=self.read_jsonl(self.extracts_path)
        if not rows:
            return None
        if not filters:
            return rows[-1]
        for item in reversed(rows):
            saved=item.get('filters') or {}
            if all(saved.get(k)==v for k,v in filters.items() if v is not None):
                return item
        return None
